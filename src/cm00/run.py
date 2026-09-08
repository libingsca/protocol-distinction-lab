from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .core import (
    EPS, Space, align_partition, all_senders, all_tasks, decoder_accuracy, exact_metrics,
    find_h1_cases, maturation_curve, minimax_barrier, source_hash, complement,
)
from .independent_checker import brute_force_best_accuracy


CONFIG = {
    "experiment": "CM00",
    "version": "0.1",
    "alpha": 0.5,
    "learning_rate": 1.0,
    "steps": [0, 1, 2, 4, 8, 16, 32],
    "spaces": [
        {"name": "level1_no_side_info", "n_x": 4, "n_z": 1},
        {"name": "level2_receiver_side_info", "n_x": 4, "n_z": 2},
    ],
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_matched_distance_controls(space: Space) -> dict:
    """Exhaustive control distribution; no sampled random baseline."""
    senders = all_senders(space)
    from .core import canonical_sender
    target_partitions = sorted({canonical_sender(sender, space.n_messages) for sender in senders})
    rows = defaultdict(lambda: defaultdict(int))
    for task in all_tasks(space):
        metrics = {sender: exact_metrics(task, sender, space) for sender in senders}
        if max(value.accuracy for value in metrics.values()) < 1.0 - EPS:
            continue
        for sender in senders:
            base = metrics[sender]
            for target_partition in target_partitions:
                if canonical_sender(sender, space.n_messages) == target_partition:
                    continue
                rewritten = align_partition(task, sender, target_partition, space, CONFIG["alpha"])
                target = metrics[rewritten]
                distance = sum(a != b for a, b in zip(sender, rewritten))
                curve = maturation_curve(
                    task, sender, rewritten, space,
                    CONFIG["alpha"], CONFIG["learning_rate"], CONFIG["steps"],
                )
                matured = curve[0] <= EPS and any(curve[t] > EPS for t in CONFIG["steps"] if t > 0)
                c0_accuracy = decoder_accuracy(task, rewritten, base.decoder, space) - base.accuracy
                informative_h1 = (
                    c0_accuracy <= EPS
                    and target.accuracy - base.accuracy > EPS
                    and target.mutual_information - base.mutual_information > EPS
                )
                irrelevant = target.mutual_information - base.mutual_information <= EPS
                nonbeneficial = (
                    target.accuracy - base.accuracy <= EPS
                    and target.mutual_information - base.mutual_information <= EPS
                )
                row = rows[distance]
                row["all"] += 1
                row["informative_h1"] += informative_h1
                row["irrelevant"] += irrelevant
                row["strict_nonbeneficial"] += nonbeneficial
                row["matured_informative_h1"] += informative_h1 and matured
                row["matured_irrelevant"] += irrelevant and matured
                row["matured_strict_nonbeneficial"] += nonbeneficial and matured
    result = {}
    for distance, counts in sorted(rows.items()):
        item = dict(counts)
        item["maturation_rate_informative_h1"] = (
            item.get("matured_informative_h1", 0) / item.get("informative_h1", 1)
            if item.get("informative_h1", 0) else None
        )
        item["maturation_rate_irrelevant"] = (
            item.get("matured_irrelevant", 0) / item.get("irrelevant", 1)
            if item.get("irrelevant", 0) else None
        )
        item["maturation_rate_strict_nonbeneficial"] = (
            item.get("matured_strict_nonbeneficial", 0) / item.get("strict_nonbeneficial", 1)
            if item.get("strict_nonbeneficial", 0) else None
        )
        result[str(distance)] = item
    return result


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
    audits = []
    curves = []
    graph_rows = []

    for spec in CONFIG["spaces"]:
        space = Space(**spec)
        cases, summary = find_h1_cases(space)
        summary["matched_distance_controls"] = analyze_matched_distance_controls(space)

        # Independent exhaustive receiver audit for every task/sender state.
        mismatch = 0
        checked = 0
        for task in all_tasks(space):
            for sender in all_senders(space):
                direct = exact_metrics(task, sender, space)
                brute_acc, _, _ = brute_force_best_accuracy(task, sender, space)
                checked += 1
                mismatch += abs(direct.accuracy - brute_acc) > EPS
        audits.append({"space": space.name, "states_checked": checked, "accuracy_mismatches": mismatch})

        # Curves for all H1 cases; deduplicate only in reporting, never in search.
        for index, case in enumerate(cases):
            curve = maturation_curve(
                case["task"], case["sender"], case["rewritten"], space,
                CONFIG["alpha"], CONFIG["learning_rate"], CONFIG["steps"],
            )
            positive_steps_after_zero = [step for step, credit in curve.items() if step > 0 and credit > EPS]
            case["c0_cross_entropy"] = curve[0]
            case["dynamic_eligible"] = curve[0] <= EPS
            case["dynamic_t_star"] = (
                min(positive_steps_after_zero)
                if case["dynamic_eligible"] and positive_steps_after_zero else None
            )
            case["c32_cross_entropy"] = curve[max(CONFIG["steps"])]
            for step, credit in curve.items():
                curves.append({
                    "space": space.name,
                    "case_index": index,
                    "canonical_key": case["canonical_key"],
                    "step": step,
                    "credit_cross_entropy": credit,
                })

        summary["dynamic_matured_by_32"] = sum(case["dynamic_t_star"] is not None for case in cases)
        summary["dynamic_matured_canonical_by_32"] = len({
            case["canonical_key"] for case in cases if case["dynamic_t_star"] is not None
        })
        summary["rewrite_kind_counts"] = dict(Counter(case["rewrite_kind"] for case in cases))
        summary["strict_negative_cross_entropy_cases"] = sum(case["c0_cross_entropy"] < -EPS for case in cases)
        summary["neutral_cross_entropy_cases"] = sum(abs(case["c0_cross_entropy"]) <= EPS for case in cases)
        summary["preexisting_positive_cross_entropy_cases"] = sum(case["c0_cross_entropy"] > EPS for case in cases)
        summary["t_star_counts"] = {str(k): v for k, v in Counter(case["dynamic_t_star"] for case in cases).items()}

        # Exact graph barriers for canonical H1 starts. E1 and E1+E2 graphs.
        seen_starts = set()
        for case in cases:
            start_key = (tuple(case["task"]), tuple(case["sender"]))
            if start_key in seen_starts:
                continue
            seen_starts.add(start_key)
            b1 = minimax_barrier(case["task"], case["sender"], space, 1)
            b2 = minimax_barrier(case["task"], case["sender"], space, 2)
            graph_rows.append({
                "space": space.name,
                "task": case["task"],
                "sender": case["sender"],
                "barrier_e1": b1,
                "barrier_e1_e2": b2,
            })
        finite_b1 = [row["barrier_e1"] for row in graph_rows if row["space"] == space.name and row["barrier_e1"] is not None]
        finite_b2 = [row["barrier_e1_e2"] for row in graph_rows if row["space"] == space.name and row["barrier_e1_e2"] is not None]
        summary["graph_start_states"] = len(seen_starts)
        summary["positive_barrier_e1"] = sum(value > EPS for value in finite_b1)
        summary["positive_barrier_e1_e2"] = sum(value > EPS for value in finite_b2)
        summary["unreachable_e1"] = len(seen_starts) - len(finite_b1)
        summary["unreachable_e1_e2"] = len(seen_starts) - len(finite_b2)
        summaries.append(summary)
        all_cases.extend({"space": space.name, **case} for case in cases)

    with (output / "canonical_cases.jsonl").open("w", encoding="utf-8") as handle:
        seen = set()
        for case in all_cases:
            key = (case["space"], case["canonical_key"])
            if key not in seen:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
                seen.add(key)
    write_json(output / "counterexamples.json", all_cases)
    write_json(output / "summary.json", summaries)
    write_json(output / "audit_report.json", {"passed": all(a["accuracy_mismatches"] == 0 for a in audits), "details": audits})
    with (output / "maturation_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["space", "case_index", "canonical_key", "step", "credit_cross_entropy"])
        writer.writeheader(); writer.writerows(curves)
    with (output / "protocol_graphs.jsonl").open("w", encoding="utf-8") as handle:
        for row in graph_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    source_paths = [Path(__file__), Path(__file__).with_name("core.py"), Path(__file__).with_name("independent_checker.py")]
    write_json(output / "source_hashes.json", source_hash(source_paths))

    report = ["# CM00 第一轮实验结果", "", "## 汇总", ""]
    for summary in summaries:
        report.extend([
            f"### {summary['space']['name']}", "",
            f"- 任务数：{summary['task_count']}",
            f"- 其中存在充分协议的任务数：{summary['solvable_task_count']}",
            f"- Sender 数：{summary['sender_count']}",
            f"- 原始候选对：{summary['raw_candidate_pairs']}",
            f"- H1 原始案例：{summary['raw_h1_cases']}",
            f"- H1 规范化案例：{summary['canonical_h1_cases']}",
            f"- 无并列 H1 案例：{summary['no_tie_h1_cases']}",
            f"- 严格负即时信用案例：{summary['strict_negative_h1_cases']}",
            f"- 交叉熵口径严格负即时信用案例：{summary['strict_negative_cross_entropy_cases']}",
            f"- 交叉熵在 t=0 已为正的非成熟案例：{summary['preexisting_positive_cross_entropy_cases']}",
            f"- 32 步内成熟的规范化案例：{summary['dynamic_matured_canonical_by_32']}",
            f"- E1 图正屏障起点：{summary['positive_barrier_e1']}",
            f"- E1+E2 图正屏障起点：{summary['positive_barrier_e1_e2']}", "",
            "同距离穷举对照：", "",
        ])
        for distance, control in summary["matched_distance_controls"].items():
            report.append(
                f"- 距离 {distance}：信息增益 H1 成熟率 "
                f"{control['maturation_rate_informative_h1']}; "
                f"无信息增益改写成熟率 {control['maturation_rate_irrelevant']}; "
                f"严格无组合价值对照成熟率 {control['maturation_rate_strict_nonbeneficial']}"
            )
        report.extend([
            "",
        ])
    report.extend(["## 审计", "", f"独立 receiver 穷举审计：{'通过' if all(a['accuracy_mismatches'] == 0 for a in audits) else '失败'}。", ""])
    (output / "RESULTS_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
