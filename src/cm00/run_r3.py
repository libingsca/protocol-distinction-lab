from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path

from .core import (
    EPS, Space, align_partition, all_senders, all_tasks, canonical_sender,
    cross_entropy, exact_metrics, maturation_curve, optimal_cross_entropy,
    partition_distance, smoothed_logits, source_hash,
)
from .independent_checker import brute_force_best_accuracy


CONFIG = {
    "experiment": "CM00-R3",
    "version": "0.1",
    "primary_alpha": 0.5,
    "secondary_alpha": 0.25,
    "learning_rate": 1.0,
    "steps": [0, 1, 2, 4, 8, 16, 32],
    "numeric_epsilon": 1e-12,
    "counterexample_epsilon": 1e-9,
    "spaces": [
        {"name": "R3_A_x6_binary", "n_x": 6, "n_z": 1, "n_messages": 2, "n_labels": 2},
        {"name": "R3_C_x5_side_info", "n_x": 5, "n_z": 2, "n_messages": 2, "n_labels": 2},
        {"name": "R3_D_x5_three_labels", "n_x": 5, "n_z": 1, "n_messages": 2, "n_labels": 3},
        {"name": "R3_B_x7_binary", "n_x": 7, "n_z": 1, "n_messages": 2, "n_labels": 2},
        {"name": "R3_E_x5_m3_y3", "n_x": 5, "n_z": 1, "n_messages": 3, "n_labels": 3},
    ],
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def partitions(space: Space) -> list[tuple[int, ...]]:
    return sorted({canonical_sender(sender, space.n_messages) for sender in all_senders(space)})


def edge_cost_ce(task, source, target, space: Space, alpha: float) -> float:
    logits = smoothed_logits(task, source, space, alpha)
    aligned = align_partition(task, source, target, space, alpha, primary="cross_entropy")
    return max(0.0, cross_entropy(task, aligned, logits, space) - cross_entropy(task, source, logits, space))


def all_minimax_barriers(task, space: Space, max_distance: int, alpha: float) -> dict[tuple[int, ...], float]:
    """Directed minimax distance to any sufficient node, computed in one reverse pass."""
    nodes = partitions(space)
    sufficient = {node for node in nodes if exact_metrics(task, node, space).accuracy >= 1.0 - EPS}
    reverse_edges = {node: [] for node in nodes}
    for source in nodes:
        for target in nodes:
            distance = partition_distance(source, target, space.n_messages)
            if distance == 0 or distance > max_distance:
                continue
            reverse_edges[target].append((source, edge_cost_ce(task, source, target, space, alpha)))

    best = {node: math.inf for node in nodes}
    heap = []
    for node in sufficient:
        best[node] = 0.0
        heapq.heappush(heap, (0.0, node))
    while heap:
        value, current = heapq.heappop(heap)
        if value > best[current] + EPS:
            continue
        for predecessor, edge_cost in reverse_edges[current]:
            candidate = max(value, edge_cost)
            if candidate + EPS < best[predecessor]:
                best[predecessor] = candidate
                heapq.heappush(heap, (candidate, predecessor))
    return best


def decimal_c0(task, source, target, space: Space, alpha: float) -> str:
    """High-precision recomputation for the closest boundary candidate."""
    getcontext().prec = 60
    counts = [[[0 for _ in range(space.n_labels)] for _ in range(space.n_z)] for _ in range(space.n_messages)]
    for x in range(space.n_x):
        for z in range(space.n_z):
            counts[source[x]][z][task[x * space.n_z + z]] += 1

    def loss(sender) -> Decimal:
        total_loss = Decimal(0)
        for x in range(space.n_x):
            for z in range(space.n_z):
                m = sender[x]
                y = task[x * space.n_z + z]
                numerator = Decimal(counts[m][z][y]) + Decimal(str(alpha))
                denominator = Decimal(sum(counts[m][z])) + Decimal(str(alpha)) * space.n_labels
                total_loss -= (numerator / denominator).ln()
        return total_loss / Decimal(space.omega_size)

    return str(loss(source) - loss(target))


def run_space(space: Space) -> tuple[dict, list[dict], list[dict]]:
    nodes = partitions(space)
    raw_senders = all_senders(space)
    task_count = solvable_count = candidate_count = informative_count = 0
    negative_count = zero_count = dynamics_count = 0
    min_c0 = math.inf
    min_secondary_c0 = math.inf
    closest = None
    p1_counterexamples = []
    p2_counterexamples = []
    audit_states = audit_mismatches = 0
    graph_eligible = positive_e1 = positive_e2 = unreachable_e1 = unreachable_e2 = 0
    threshold = CONFIG["counterexample_epsilon"]

    for task in all_tasks(space):
        task_count += 1
        raw_metrics = {sender: exact_metrics(task, sender, space) for sender in raw_senders}
        for sender, direct in raw_metrics.items():
            brute_accuracy, _, _ = brute_force_best_accuracy(task, sender, space)
            audit_states += 1
            audit_mismatches += abs(direct.accuracy - brute_accuracy) > EPS

        node_metrics = {node: raw_metrics[node] for node in nodes}
        if max(value.accuracy for value in node_metrics.values()) < 1.0 - EPS:
            continue
        solvable_count += 1

        for source in nodes:
            base = node_metrics[source]
            base_logits = smoothed_logits(task, source, space, CONFIG["primary_alpha"])
            base_loss = cross_entropy(task, source, base_logits, space)
            for target_partition in nodes:
                if source == target_partition:
                    continue
                candidate_count += 1
                target = node_metrics[target_partition]
                if base.has_tie or target.has_tie:
                    continue
                if target.accuracy - base.accuracy <= EPS:
                    continue
                if target.mutual_information - base.mutual_information <= EPS:
                    continue
                informative_count += 1

                aligned = align_partition(
                    task, source, target_partition, space,
                    CONFIG["primary_alpha"], primary="cross_entropy",
                )
                c0 = base_loss - cross_entropy(task, aligned, base_logits, space)
                if c0 < min_c0:
                    min_c0 = c0
                    closest = {
                        "space": space.name,
                        "task": list(task),
                        "source": list(source),
                        "target_partition": list(target_partition),
                        "aligned_target": list(aligned),
                        "c0_ce": c0,
                        "delta_accuracy": target.accuracy - base.accuracy,
                        "delta_mi": target.mutual_information - base.mutual_information,
                    }

                aligned_secondary = align_partition(
                    task, source, target_partition, space,
                    CONFIG["secondary_alpha"], primary="cross_entropy",
                )
                secondary_logits = smoothed_logits(task, source, space, CONFIG["secondary_alpha"])
                secondary_c0 = (
                    cross_entropy(task, source, secondary_logits, space)
                    - cross_entropy(task, aligned_secondary, secondary_logits, space)
                )
                min_secondary_c0 = min(min_secondary_c0, secondary_c0)

                if abs(c0) <= EPS:
                    zero_count += 1
                if c0 < -threshold:
                    negative_count += 1
                    curve = maturation_curve(
                        task, source, aligned, space, CONFIG["primary_alpha"],
                        CONFIG["learning_rate"], CONFIG["steps"],
                    )
                    if curve[32] > EPS:
                        dynamics_count += 1
                        p1_counterexamples.append({
                            "space": space.name,
                            "task": list(task),
                            "source": list(source),
                            "aligned_target": list(aligned),
                            "c0_ce": c0,
                            "c32_ce": curve[32],
                            "base_optimal_ce": optimal_cross_entropy(task, source, space),
                            "target_optimal_ce": optimal_cross_entropy(task, aligned, space),
                            "curve": {str(k): v for k, v in curve.items()},
                        })

        barriers_e1 = all_minimax_barriers(task, space, 1, CONFIG["primary_alpha"])
        barriers_e2 = all_minimax_barriers(task, space, 2, CONFIG["primary_alpha"])
        for source in nodes:
            base = node_metrics[source]
            if base.has_tie or base.accuracy >= 1.0 - EPS:
                continue
            graph_eligible += 1
            b1 = barriers_e1[source]
            b2 = barriers_e2[source]
            unreachable_e1 += math.isinf(b1)
            unreachable_e2 += math.isinf(b2)
            positive_e1 += not math.isinf(b1) and b1 > threshold
            positive_e2 += not math.isinf(b2) and b2 > threshold
            if (math.isinf(b1) or b1 > threshold) and (math.isinf(b2) or b2 > threshold):
                p2_counterexamples.append({
                    "space": space.name,
                    "task": list(task),
                    "source": list(source),
                    "barrier_e1": None if math.isinf(b1) else b1,
                    "barrier_e1_e2": None if math.isinf(b2) else b2,
                })

    if closest is not None:
        closest["decimal_c0_ce"] = decimal_c0(
            closest["task"], closest["source"], closest["aligned_target"],
            space, CONFIG["primary_alpha"],
        )
    summary = {
        "space": asdict(space),
        "task_count": task_count,
        "solvable_task_count": solvable_count,
        "sender_count": len(raw_senders),
        "partition_count": len(nodes),
        "candidate_pairs": candidate_count,
        "informative_no_tie_pairs": informative_count,
        "minimum_c0_ce_alpha_0_5": None if math.isinf(min_c0) else min_c0,
        "minimum_c0_ce_alpha_0_25": None if math.isinf(min_secondary_c0) else min_secondary_c0,
        "c0_zero_pairs": zero_count,
        "strict_negative_pairs": negative_count,
        "matured_p1_counterexamples": dynamics_count,
        "graph_eligible_starts": graph_eligible,
        "positive_barrier_e1": positive_e1,
        "positive_barrier_e1_e2": positive_e2,
        "unreachable_e1": unreachable_e1,
        "unreachable_e1_e2": unreachable_e2,
        "p2_counterexamples": len(p2_counterexamples),
        "audit_states": audit_states,
        "audit_mismatches": audit_mismatches,
        "closest_case": closest,
    }
    return summary, p1_counterexamples, p2_counterexamples


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.lock.json", CONFIG)
    write_json(output / "environment.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    })
    summaries = []
    p1_cases = []
    p2_cases = []
    for spec in CONFIG["spaces"]:
        space = Space(**spec)
        print(f"START {space.name}", flush=True)
        summary, p1, p2 = run_space(space)
        summaries.append(summary)
        p1_cases.extend(p1)
        p2_cases.extend(p2)
        write_json(output / "summary.partial.json", summaries)
        print(
            f"DONE {space.name}: pairs={summary['candidate_pairs']} "
            f"P1={summary['matured_p1_counterexamples']} P2={summary['p2_counterexamples']}",
            flush=True,
        )

    write_json(output / "summary.json", summaries)
    write_json(output / "p1_counterexamples.json", p1_cases)
    write_json(output / "p2_counterexamples.json", p2_cases)
    write_json(output / "audit_report.json", {
        "passed": all(s["audit_mismatches"] == 0 for s in summaries),
        "total_states": sum(s["audit_states"] for s in summaries),
        "total_mismatches": sum(s["audit_mismatches"] for s in summaries),
    })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "space", "task_count", "solvable_task_count", "candidate_pairs",
            "informative_no_tie_pairs", "minimum_c0_ce_alpha_0_5",
            "minimum_c0_ce_alpha_0_25", "strict_negative_pairs",
            "matured_p1_counterexamples", "p2_counterexamples",
            "audit_states", "audit_mismatches",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({**{key: s.get(key) for key in fields}, "space": s["space"]["name"]})

    source_paths = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("independent_checker.py")]
    write_json(output / "source_hashes.json", source_hash(source_paths))
    lines = ["# CM00-R3 自动汇总", ""]
    for s in summaries:
        lines.extend([
            f"## {s['space']['name']}", "",
            f"- 任务 / 可解任务：{s['task_count']} / {s['solvable_task_count']}",
            f"- 候选对 / 无并列信息增益对：{s['candidate_pairs']} / {s['informative_no_tie_pairs']}",
            f"- 最小 C0_CE（alpha=0.5 / 0.25）：{s['minimum_c0_ce_alpha_0_5']} / {s['minimum_c0_ce_alpha_0_25']}",
            f"- P1 反例：{s['matured_p1_counterexamples']}",
            f"- P2 反例：{s['p2_counterexamples']}",
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

