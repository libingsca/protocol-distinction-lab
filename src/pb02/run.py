from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pb01.neural import MLPReceiver, adapt_curve, conditional_targets, softmax
from pb01.run import same_distance_controls

CONFIG = {
    "experiment": "PB02", "version": "1.0", "seeds": list(range(30, 60)),
    "task": [0, 1, 2, 0, 1], "source": [0, 1, 1, 2, 1],
    "target": [0, 1, 2, 0, 1], "spare_target": [0, 1, 3, 2, 1],
    "alpha": 0.5, "pretrain_steps": 3000, "pretrain_learning_rate": 0.1,
    "adapt_learning_rate": 1.0, "adapt_steps": list(range(33)),
    "T_loss": 2, "T_decision": 4, "primary_architecture": "mlp8",
    "architectures": ["mlp8", "mlp4", "mlp16", "linear"],
    "thresholds": {"sign_count": 24, "c0_max": -0.05, "c2_min": 0.02,
                   "accuracy_count": 24, "accuracy_min": 0.10, "spare_min": 0.50},
    "control_comparison": "median target C2 exceeds both control medians (PB01 convention)",
    "control_rng_offset": 10000, "target_alignment": "fixed preregistered mapping; no search",
    "first_loss_definition": "first integer step with C_CE > 0, through 32; null if censored",
    "first_decision_definition": "first integer step with D-C accuracy >= 0.10; null if censored",
    "linear_definition": "one-hot @ W + b; W ~ N(0,0.3), b=0; same SGD and pretraining",
    "scope": "fixed task labels for controlled intervention, no validation/test data or autonomous sender",
}


class LinearReceiver:
    def __init__(self, n_messages, seed):
        self.w = np.random.default_rng(seed).normal(0, 0.3, (n_messages, 3))
        self.b = np.zeros(3)

    def clone(self):
        other = LinearReceiver(len(self.w), 0)
        other.w, other.b = self.w.copy(), self.b.copy()
        return other

    def forward_distributions(self, x):
        return x, x, softmax(x @ self.w + self.b)

    def forward(self, messages):
        return self.forward_distributions(np.eye(len(self.w))[messages])

    loss_accuracy = MLPReceiver.loss_accuracy
    loss_accuracy_soft = MLPReceiver.loss_accuracy_soft

    def hard_step(self, messages, labels, learning_rate):
        x, _, probs = self.forward(np.asarray(messages))
        probs[np.arange(len(labels)), labels] -= 1
        grad = probs / len(labels)
        self.w -= learning_rate * x.T @ grad
        self.b -= learning_rate * grad.sum(axis=0)

    def soft_target_step(self, targets, weights, learning_rate):
        _, _, probs = self.forward(np.arange(len(self.w)))
        grad = (probs - targets) * (weights / weights.sum())[:, None]
        self.w -= learning_rate * grad
        self.b -= learning_rate * grad.sum(axis=0)


def pretrain(architecture, n_messages, seed):
    if architecture == "linear":
        model = LinearReceiver(n_messages, seed)
    else:
        model = MLPReceiver.random(n_messages, 3, int(architecture[3:]), seed)
    targets, weights = conditional_targets(CONFIG["source"], CONFIG["task"], n_messages, 3, CONFIG["alpha"])
    for _ in range(CONFIG["pretrain_steps"]):
        model.soft_target_step(targets, weights, CONFIG["pretrain_learning_rate"])
    error = float(np.max(np.abs(model.forward(np.arange(n_messages))[2] - targets)))
    return model, error


def run_seed(architecture, seed):
    source, target, labels = CONFIG["source"], CONFIG["target"], CONFIG["task"]
    model, error = pretrain(architecture, 3, seed)
    curve = lambda m, mapping: adapt_curve(m.clone(), mapping, labels, CONFIG["adapt_steps"], CONFIG["adapt_learning_rate"])
    left, right = curve(model, source), curve(model, target)
    credits = [left[t][0] - right[t][0] for t in CONFIG["adapt_steps"]]
    gains = [right[t][1] - left[t][1] for t in CONFIG["adapt_steps"]]
    pool = same_distance_controls(source, target, 3, labels)
    noninfo = [item for item in pool if item[1] <= 1e-12 and item[2] <= 1e-12]
    rng = np.random.default_rng(seed + CONFIG["control_rng_offset"])
    random_mapping = pool[int(rng.integers(len(pool)))][0]
    noninfo_mapping = noninfo[int(rng.integers(len(noninfo)))][0]
    random_curve, noninfo_curve = curve(model, random_mapping), curve(model, noninfo_mapping)
    spare, spare_error = pretrain(architecture, 4, seed)
    spare_left, spare_right = curve(spare, source), curve(spare, CONFIG["spare_target"])
    spare_c0 = spare_left[0][0] - spare_right[0][0]
    soft_path = {}
    for lam in (0., .25, .5, .75, 1.):
        mixed = (1 - lam) * np.eye(3)[source] + lam * np.eye(3)[target]
        loss, accuracy = model.loss_accuracy_soft(mixed, labels)
        soft_path[str(lam)] = {"loss": loss, "accuracy": accuracy, "credit": left[0][0] - loss}
    first_loss = next((t for t, c in enumerate(credits) if c > 0), None)
    first_decision = next((t for t, g in enumerate(gains) if g >= .10), None)
    return {
        "architecture": architecture, "seed": seed, "target": target,
        "pretrain_error": error, "spare_pretrain_error": spare_error,
        "c0_ce": credits[0], "c2_ce": credits[2], "accuracy_gain_t4": gains[4],
        "random_mapping": random_mapping, "noninfo_mapping": noninfo_mapping,
        "random_c2_ce": left[2][0] - random_curve[2][0],
        "noninfo_c2_ce": left[2][0] - noninfo_curve[2][0],
        "spare_c0_ce": spare_c0,
        "spare_barrier_reduction": 1 - abs(spare_c0) / abs(credits[0]) if credits[0] else None,
        "first_loss": first_loss, "first_decision": first_decision,
        "loss_precedes_decision": first_loss is not None and first_decision is not None and first_loss < first_decision,
        "branches": {"C": left, "D": right, "random": random_curve, "noninfo": noninfo_curve,
                     "spare_C": spare_left, "spare_D": spare_right},
        "credits": credits, "accuracy_gains": gains, "soft_path": soft_path,
    }


def summarize(rows):
    if sorted(r["seed"] for r in rows) != CONFIG["seeds"]:
        raise ValueError("Requires exactly the frozen 30 independent seeds")
    med = lambda key: statistics.median(r[key] for r in rows)
    values = {"median_" + key: med(key) for key in (
        "c0_ce", "c2_ce", "accuracy_gain_t4", "random_c2_ce", "noninfo_c2_ce", "spare_barrier_reduction", "pretrain_error")}
    values["sign_count"] = sum(r["c0_ce"] < 0 < r["c2_ce"] for r in rows)
    values["accuracy_count"] = sum(r["accuracy_gain_t4"] >= .10 for r in rows)
    values["loss_precedes_decision_count"] = sum(r["loss_precedes_decision"] for r in rows)
    values["target_beats_both_controls_count"] = sum(r["c2_ce"] > max(r["random_c2_ce"], r["noninfo_c2_ce"]) for r in rows)
    th = CONFIG["thresholds"]
    values["checks"] = {
        "1_sign_count": values["sign_count"] >= th["sign_count"],
        "2_median_c0": values["median_c0_ce"] <= th["c0_max"],
        "3_median_c2": values["median_c2_ce"] >= th["c2_min"],
        "4_accuracy_count_t4": values["accuracy_count"] >= th["accuracy_count"],
        "5_median_accuracy_t4": values["median_accuracy_gain_t4"] >= th["accuracy_min"],
        "6_spare_reduction": values["median_spare_barrier_reduction"] >= th["spare_min"],
        "7_both_controls": values["median_c2_ce"] > max(values["median_random_c2_ce"], values["median_noninfo_c2_ce"]),
    }
    values["passed"] = all(values["checks"].values())
    return values


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hashes(root):
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "tests").glob("*.py"))
    paths += [root / "docs/experiment-plans/pb02-dual-timescale-confirmation.zh-CN.md", root / "pyproject.toml"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    frozen_hashes = hashes(root)
    write_json(output / "config.lock.json", CONFIG)
    write_json(output / "source_hashes.json", frozen_hashes)
    write_json(output / "environment.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "executable": sys.executable, "numpy": np.__version__, "platform": platform.platform(),
        "dependencies": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "command": sys.argv,
    })
    summaries = {}
    for architecture in CONFIG["architectures"]:
        rows = []
        for seed in CONFIG["seeds"]:
            row = run_seed(architecture, seed)
            rows.append(row)
            write_json(output / f"{architecture}_seed_{seed}.json", row)
            print(f"{architecture} seed={seed} C2={row['c2_ce']:.6f} gain4={row['accuracy_gain_t4']:.3f}", flush=True)
        summaries[architecture] = summarize(rows)
    write_json(output / "summary.json", {"primary_passed": summaries["mlp8"]["passed"], "architectures": summaries})
    write_json(output / "execution_audit.json", {"source_unchanged": hashes(root) == frozen_hashes,
        "completed_at": datetime.now(timezone.utc).isoformat(), "row_count": 120})
    if hashes(root) != frozen_hashes:
        raise RuntimeError("Source changed during experiment")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
