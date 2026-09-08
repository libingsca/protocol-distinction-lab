from __future__ import annotations

import argparse
import itertools
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cm00.core import (
    EPS, Space, align_partition, exact_metrics, maturation_curve,
    message_relabelings, smoothed_logits, cross_entropy, source_hash,
)
from cm00.run_r3 import run_space
from .neural import adapt_curve, best_aligned_mapping, pretrain_receiver


CONFIG = {
    "experiment": "PB01",
    "version": "0.1",
    "alpha": 0.5,
    "secondary_alpha": 0.25,
    "hidden": 8,
    "pretrain_steps": 3000,
    "pretrain_learning_rate": 0.1,
    "adapt_learning_rate": 1.0,
    "adapt_steps": [0, 1, 2, 4, 8, 16, 32],
    "seeds": list(range(30)),
    "task": [0, 1, 2, 0, 1],
    "source": [0, 1, 1, 2, 1],
    "target_partition": [0, 1, 2, 0, 1],
    "spare_target": [0, 1, 3, 2, 1],
    "ablations": [
        {"name": "A1_x4_m3_y3", "n_x": 4, "n_z": 1, "n_messages": 3, "n_labels": 3},
        {"name": "A2_x5_m3_y2", "n_x": 5, "n_z": 1, "n_messages": 3, "n_labels": 2},
        {"name": "A3_x5_m2_y3", "n_x": 5, "n_z": 1, "n_messages": 2, "n_labels": 3},
    ],
    "pass_thresholds": {
        "median_c0_ce_max": -0.05,
        "median_c2_ce_min": 0.02,
        "accuracy_gain_min": 0.10,
        "spare_barrier_reduction_min": 0.50,
        "sign_seed_min": 24,
    },
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def robust_count(space: Space, cases: list[dict]) -> int:
    count = 0
    for case in cases:
        task = tuple(case["task"])
        source = tuple(case["source"])
        target = tuple(case["aligned_target"])
        aligned = align_partition(task, source, target, space, CONFIG["secondary_alpha"], primary="cross_entropy")
        curve = maturation_curve(
            task, source, aligned, space, CONFIG["secondary_alpha"], 1.0,
            CONFIG["adapt_steps"],
        )
        count += curve[0] < -1e-9 and curve[32] > EPS
    return count


def run_ablations():
    results = []
    for spec in CONFIG["ablations"]:
        space = Space(**spec)
        print(f"START {space.name}", flush=True)
        summary, p1, p2 = run_space(space)
        summary["robust_p1_both_alpha"] = robust_count(space, p1)
        summary["p1_raw"] = len(p1)
        summary["p2_raw"] = len(p2)
        results.append(summary)
        print(f"DONE {space.name}: P1={len(p1)} robust={summary['robust_p1_both_alpha']} P2={len(p2)}", flush=True)

    # A4 is a controlled intervention with an unused fourth message.
    task = tuple(CONFIG["task"])
    source = tuple(CONFIG["source"])
    target = tuple(CONFIG["target_partition"])
    spare = tuple(CONFIG["spare_target"])
    a4 = {"name": "A4_spare_fourth_message", "curves": {}}
    for alpha in (CONFIG["alpha"], CONFIG["secondary_alpha"]):
        three = maturation_curve(task, source, target, Space("m3", 5, 1, 3, 3), alpha, 1.0, CONFIG["adapt_steps"])
        four = maturation_curve(task, source, spare, Space("m4", 5, 1, 4, 3), alpha, 1.0, CONFIG["adapt_steps"])
        a4["curves"][str(alpha)] = {
            "occupied_three_messages": {str(k): v for k, v in three.items()},
            "spare_fourth_message": {str(k): v for k, v in four.items()},
            "barrier_reduction": 1.0 - abs(four[0]) / abs(three[0]),
        }
    results.append(a4)
    return results


def same_distance_controls(source, target, n_messages, labels):
    distance = sum(a != b for a, b in zip(source, target))
    base = exact_metrics(labels, source, Space("control", len(labels), 1, n_messages, 3))
    candidates = []
    for mapping in itertools.product(range(n_messages), repeat=len(source)):
        if sum(a != b for a, b in zip(source, mapping)) != distance:
            continue
        if mapping in message_relabelings(target, n_messages):
            continue
        metrics = exact_metrics(labels, mapping, Space("control", len(labels), 1, n_messages, 3))
        candidates.append((mapping, metrics.accuracy - base.accuracy, metrics.mutual_information - base.mutual_information))
    return candidates


def run_neural():
    labels = tuple(CONFIG["task"])
    source = tuple(CONFIG["source"])
    target_partition = tuple(CONFIG["target_partition"])
    spare_target = tuple(CONFIG["spare_target"])
    random_pool = same_distance_controls(source, target_partition, 3, labels)
    noninfo_pool = [item for item in random_pool if item[1] <= EPS and item[2] <= EPS]
    rows = []

    for seed in CONFIG["seeds"]:
        receiver, pretrain_error = pretrain_receiver(
            source, labels, 3, 3, CONFIG["hidden"], seed,
            CONFIG["alpha"], CONFIG["pretrain_steps"], CONFIG["pretrain_learning_rate"],
        )
        aligned_target = best_aligned_mapping(receiver, target_partition, labels, 3)
        source_onehot = np.eye(3)[list(source)]
        target_onehot = np.eye(3)[list(aligned_target)]
        source_fixed_loss = receiver.loss_accuracy(source, labels)[0]
        soft_path = {}
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            mixed = (1.0 - lam) * source_onehot + lam * target_onehot
            loss, accuracy = receiver.loss_accuracy_soft(mixed, labels)
            soft_path[str(lam)] = {"loss": loss, "accuracy": accuracy, "credit": source_fixed_loss - loss}
        left = adapt_curve(receiver.clone(), source, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])
        right = adapt_curve(receiver.clone(), aligned_target, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])
        credits = {step: left[step][0] - right[step][0] for step in CONFIG["adapt_steps"]}

        rng = np.random.default_rng(seed + 10000)
        random_mapping = random_pool[int(rng.integers(len(random_pool)))][0]
        random_curve = adapt_curve(receiver.clone(), random_mapping, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])
        random_c2 = left[2][0] - random_curve[2][0]
        noninfo_mapping = noninfo_pool[int(rng.integers(len(noninfo_pool)))][0]
        noninfo_curve = adapt_curve(receiver.clone(), noninfo_mapping, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])

        spare_receiver, spare_error = pretrain_receiver(
            source, labels, 4, 3, CONFIG["hidden"], seed,
            CONFIG["alpha"], CONFIG["pretrain_steps"], CONFIG["pretrain_learning_rate"],
        )
        spare_left = adapt_curve(spare_receiver.clone(), source, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])
        spare_right = adapt_curve(spare_receiver.clone(), spare_target, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])
        spare_credits = {step: spare_left[step][0] - spare_right[step][0] for step in CONFIG["adapt_steps"]}

        matched_receiver, matched_error = pretrain_receiver(
            aligned_target, labels, 3, 3, CONFIG["hidden"], seed + 50000,
            CONFIG["alpha"], CONFIG["pretrain_steps"], CONFIG["pretrain_learning_rate"],
        )
        matched_loss, matched_accuracy = matched_receiver.loss_accuracy(aligned_target, labels)
        source_metrics = exact_metrics(labels, source, Space("neural", 5, 1, 3, 3))
        target_metrics = exact_metrics(labels, aligned_target, Space("neural", 5, 1, 3, 3))
        rows.append({
            "seed": seed,
            "pretrain_max_probability_error": pretrain_error,
            "spare_pretrain_error": spare_error,
            "matched_pretrain_error": matched_error,
            "aligned_target": list(aligned_target),
            "c0_ce": credits[0],
            "c1_ce": credits[1],
            "c2_ce": credits[2],
            "c32_ce": credits[32],
            "branch_c_accuracy_t2": left[2][1],
            "branch_d_accuracy_t2": right[2][1],
            "accuracy_gain_t2": right[2][1] - left[2][1],
            "source_a_code": source_metrics.accuracy,
            "target_a_code": target_metrics.accuracy,
            "source_mi": source_metrics.mutual_information,
            "target_mi": target_metrics.mutual_information,
            "random_mapping": list(random_mapping),
            "random_c2_ce": random_c2,
            "noninfo_mapping": list(noninfo_mapping),
            "noninfo_c2_ce": left[2][0] - noninfo_curve[2][0],
            "spare_c0_ce": spare_credits[0],
            "spare_c2_ce": spare_credits[2],
            "spare_barrier_reduction": 1.0 - abs(spare_credits[0]) / abs(credits[0]),
            "matched_target_loss": matched_loss,
            "matched_target_accuracy": matched_accuracy,
            "soft_path": soft_path,
            "soft_path_max_loss_increase": max(value["loss"] - source_fixed_loss for value in soft_path.values()),
        })
    return rows


def summarize(rows):
    median = lambda key: statistics.median(row[key] for row in rows)
    thresholds = CONFIG["pass_thresholds"]
    summary = {
        "seed_count": len(rows),
        "median_c0_ce": median("c0_ce"),
        "median_c2_ce": median("c2_ce"),
        "median_c32_ce": median("c32_ce"),
        "median_accuracy_gain_t2": median("accuracy_gain_t2"),
        "median_random_c2_ce": median("random_c2_ce"),
        "median_noninfo_c2_ce": median("noninfo_c2_ce"),
        "median_spare_c0_ce": median("spare_c0_ce"),
        "median_spare_c2_ce": median("spare_c2_ce"),
        "median_spare_barrier_reduction": median("spare_barrier_reduction"),
        "median_pretrain_error": median("pretrain_max_probability_error"),
        "median_soft_path_max_loss_increase": median("soft_path_max_loss_increase"),
        "sign_success_seeds": sum(row["c0_ce"] < 0 < row["c2_ce"] for row in rows),
        "target_a_code": rows[0]["target_a_code"],
        "source_a_code": rows[0]["source_a_code"],
        "target_mi": rows[0]["target_mi"],
        "source_mi": rows[0]["source_mi"],
    }
    checks = {
        "c0": summary["median_c0_ce"] <= thresholds["median_c0_ce_max"],
        "c2": summary["median_c2_ce"] >= thresholds["median_c2_ce_min"],
        "accuracy": summary["median_accuracy_gain_t2"] >= thresholds["accuracy_gain_min"],
        "information": summary["target_a_code"] > summary["source_a_code"] and summary["target_mi"] > summary["source_mi"],
        "random": summary["median_c2_ce"] > summary["median_random_c2_ce"],
        "spare": summary["median_spare_barrier_reduction"] >= thresholds["spare_barrier_reduction_min"],
        "signs": summary["sign_success_seeds"] >= thresholds["sign_seed_min"],
    }
    summary["checks"] = checks
    summary["passed"] = all(checks.values())
    return summary


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.lock.json", CONFIG)
    write_json(output / "environment.json", {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
    })
    ablations = run_ablations()
    write_json(output / "ablations.json", ablations)
    print("START neural_30_seeds", flush=True)
    rows = run_neural()
    summary = summarize(rows)
    write_json(output / "neural_rows.json", rows)
    write_json(output / "summary.json", summary)
    write_json(output / "source_hashes.json", source_hash([Path(__file__), Path(__file__).with_name("neural.py")]))
    print(f"DONE neural_30_seeds: passed={summary['passed']} signs={summary['sign_success_seeds']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
