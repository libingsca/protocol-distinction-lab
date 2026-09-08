from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .core import (
    EPS, Space, align_partition, all_senders, all_tasks, canonical_case,
    canonical_sender, cross_entropy, exact_metrics, maturation_curve,
    minimax_barrier_cross_entropy, optimal_cross_entropy, partition_distance,
    rewrite_distance, smoothed_logits, source_hash,
)
from .independent_checker import brute_force_best_accuracy


CONFIG = {
    "experiment": "CM00-R2",
    "version": "0.1",
    "primary_alpha": 0.5,
    "robustness_alphas": [0.25, 0.5, 1.0],
    "learning_rate": 1.0,
    "steps": [0, 1, 2, 4, 8, 16, 32],
    "numeric_epsilon": 1e-12,
    "strong_barrier_epsilon": 1e-6,
    "spaces": [
        {"name": "R2_A_five_states", "n_x": 5, "n_z": 1, "n_messages": 2, "n_labels": 2},
        {"name": "R2_B_three_labels", "n_x": 4, "n_z": 1, "n_messages": 2, "n_labels": 3},
        {"name": "R2_C_three_messages_side_info", "n_x": 4, "n_z": 2, "n_messages": 3, "n_labels": 2},
    ],
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def qualifies(curve: dict[int, float], base, target) -> bool:
    return (
        not base.has_tie
        and not target.has_tie
        and curve[0] <= EPS
        and curve[32] > EPS
        and target.accuracy - base.accuracy > EPS
        and target.mutual_information - base.mutual_information > EPS
    )


def enumerate_space(space: Space) -> tuple[list[dict], dict, list[dict]]:
    senders = all_senders(space)
    partitions = sorted({canonical_sender(sender, space.n_messages) for sender in senders})
    cases = []
    graph_rows = []
    total_tasks = 0
    solvable_tasks = 0
    candidate_pairs = 0
    weak_accuracy_cases = 0
    audit_states = 0
    audit_mismatches = 0

    for task in all_tasks(space):
        total_tasks += 1
        metrics = {sender: exact_metrics(task, sender, space) for sender in senders}
        for sender, direct in metrics.items():
            brute_accuracy, _, _ = brute_force_best_accuracy(task, sender, space)
            audit_states += 1
            audit_mismatches += abs(direct.accuracy - brute_accuracy) > EPS

        if max(value.accuracy for value in metrics.values()) < 1.0 - EPS:
            continue
        solvable_tasks += 1

        # Search every current sender and every target partition. Target message
        # names are chosen by minimum immediate CE, as preregistered.
        for sender in senders:
            base = metrics[sender]
            base_logits = smoothed_logits(task, sender, space, CONFIG["primary_alpha"])
            base_loss = cross_entropy(task, sender, base_logits, space)
            for target_partition in partitions:
                if canonical_sender(sender, space.n_messages) == target_partition:
                    continue
                candidate_pairs += 1
                rewritten = align_partition(
                    task, sender, target_partition, space,
                    CONFIG["primary_alpha"], primary="cross_entropy",
                )
                target = metrics[rewritten]
                curve = maturation_curve(
                    task, sender, rewritten, space,
                    CONFIG["primary_alpha"], CONFIG["learning_rate"], CONFIG["steps"],
                )
                immediate_accuracy = sum(
                    base.decoder[sender_m * space.n_z + z] == task[x * space.n_z + z]
                    for x, sender_m in enumerate(rewritten)
                    for z in range(space.n_z)
                ) / space.omega_size
                c0_accuracy = immediate_accuracy - base.accuracy
                delta_accuracy = target.accuracy - base.accuracy
                delta_mi = target.mutual_information - base.mutual_information
                weak_accuracy_cases += (
                    c0_accuracy <= EPS and delta_accuracy > EPS and delta_mi > EPS
                )
                if not qualifies(curve, base, target):
                    continue

                robustness = {}
                for alpha in CONFIG["robustness_alphas"]:
                    aligned = align_partition(task, sender, target_partition, space, alpha, primary="cross_entropy")
                    robust_curve = maturation_curve(
                        task, sender, aligned, space, alpha,
                        CONFIG["learning_rate"], CONFIG["steps"],
                    )
                    robust_target = metrics[aligned]
                    robustness[str(alpha)] = {
                        "c0": robust_curve[0],
                        "c32": robust_curve[32],
                        "qualifies": qualifies(robust_curve, base, robust_target),
                    }
                key = canonical_case(task, sender, rewritten, space)
                cases.append({
                    "space": space.name,
                    "task": list(task),
                    "sender": list(sender),
                    "rewritten": list(rewritten),
                    "target_partition": list(target_partition),
                    "canonical_key": json.dumps(key, separators=(",", ":")),
                    "distance": rewrite_distance(sender, rewritten),
                    "partition_distance": partition_distance(sender, rewritten, space.n_messages),
                    "base_accuracy": base.accuracy,
                    "target_accuracy": target.accuracy,
                    "delta_accuracy": delta_accuracy,
                    "base_mi": base.mutual_information,
                    "target_mi": target.mutual_information,
                    "delta_mi": delta_mi,
                    "base_optimal_ce": optimal_cross_entropy(task, sender, space),
                    "target_optimal_ce": optimal_cross_entropy(task, rewritten, space),
                    "c0_ce": curve[0],
                    "c32_ce": curve[32],
                    "t_star": min(step for step in CONFIG["steps"] if step > 0 and curve[step] > EPS),
                    "strict_negative": curve[0] < -CONFIG["strong_barrier_epsilon"],
                    "curve": {str(step): credit for step, credit in curve.items()},
                    "robustness": robustness,
                })

        # Search barriers from every non-sufficient, non-tied canonical state.
        for start in partitions:
            start_metrics = exact_metrics(task, start, space)
            if start_metrics.accuracy >= 1.0 - EPS or start_metrics.has_tie:
                continue
            graph_rows.append({
                "space": space.name,
                "task": list(task),
                "start": list(start),
                "barrier_e1_ce": minimax_barrier_cross_entropy(
                    task, start, space, 1, CONFIG["primary_alpha"]
                ),
                "barrier_e1_e2_ce": minimax_barrier_cross_entropy(
                    task, start, space, 2, CONFIG["primary_alpha"]
                ),
            })

    canonical_cases = {(case["canonical_key"], case["space"]) for case in cases}
    strong_cases = [case for case in cases if case["strict_negative"]]
    robust_cases = [case for case in cases if all(x["qualifies"] for x in case["robustness"].values())]
    b1 = [row["barrier_e1_ce"] for row in graph_rows if row["barrier_e1_ce"] is not None]
    b2 = [row["barrier_e1_e2_ce"] for row in graph_rows if row["barrier_e1_e2_ce"] is not None]
    summary = {
        "space": asdict(space),
        "task_count": total_tasks,
        "solvable_task_count": solvable_tasks,
        "sender_count": len(senders),
        "partition_count": len(partitions),
        "candidate_pairs": candidate_pairs,
        "weak_accuracy_cases": weak_accuracy_cases,
        "primary_raw_cases": len(cases),
        "primary_canonical_cases": len(canonical_cases),
        "strict_negative_raw_cases": len(strong_cases),
        "robust_all_alpha_raw_cases": len(robust_cases),
        "t_star_counts": {str(k): v for k, v in Counter(case["t_star"] for case in cases).items()},
        "graph_states": len(graph_rows),
        "positive_barrier_e1": sum(value > CONFIG["strong_barrier_epsilon"] for value in b1),
        "positive_barrier_e1_e2": sum(value > CONFIG["strong_barrier_epsilon"] for value in b2),
        "unreachable_e1": len(graph_rows) - len(b1),
        "unreachable_e1_e2": len(graph_rows) - len(b2),
        "max_barrier_e1": max(b1) if b1 else None,
        "max_barrier_e1_e2": max(b2) if b2 else None,
        "audit_states": audit_states,
        "audit_mismatches": audit_mismatches,
    }
    return cases, summary, graph_rows


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.lock.json", CONFIG)
    write_json(output / "environment.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    })

    all_cases = []
    summaries = []
    all_graphs = []
    for spec in CONFIG["spaces"]:
        cases, summary, graphs = enumerate_space(Space(**spec))
        all_cases.extend(cases)
        summaries.append(summary)
        all_graphs.extend(graphs)

    write_json(output / "summary.json", summaries)
    write_json(output / "cases.json", all_cases)
    write_json(output / "audit_report.json", {
        "passed": all(summary["audit_mismatches"] == 0 for summary in summaries),
        "spaces": [{"space": s["space"]["name"], "states": s["audit_states"], "mismatches": s["audit_mismatches"]} for s in summaries],
    })
    with (output / "protocol_graphs.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_graphs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["space", "canonical_key", "distance", "partition_distance", "c0_ce", "c32_ce", "t_star", "delta_accuracy", "delta_mi", "strict_negative"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in all_cases:
            writer.writerow({key: case[key] for key in fields})
    source_paths = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("independent_checker.py")]
    write_json(output / "source_hashes.json", source_hash(source_paths))

    lines = ["# CM00-R2 自动汇总", ""]
    for s in summaries:
        lines.extend([
            f"## {s['space']['name']}", "",
            f"- 任务 / 可解任务：{s['task_count']} / {s['solvable_task_count']}",
            f"- Sender / 分区：{s['sender_count']} / {s['partition_count']}",
            f"- 候选对：{s['candidate_pairs']}",
            f"- 准确率弱案例：{s['weak_accuracy_cases']}",
            f"- 无并列交叉熵成熟案例（原始 / 规范化）：{s['primary_raw_cases']} / {s['primary_canonical_cases']}",
            f"- 严格负 C0 案例：{s['strict_negative_raw_cases']}",
            f"- 三个 alpha 全部稳健：{s['robust_all_alpha_raw_cases']}",
            f"- E1 / E1+E2 正交叉熵屏障：{s['positive_barrier_e1']} / {s['positive_barrier_e1_e2']}",
            f"- 独立审计不一致：{s['audit_mismatches']}", "",
        ])
    (output / "RESULTS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()

