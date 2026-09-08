from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


EPS = 1e-12


@dataclass(frozen=True)
class Space:
    name: str
    n_x: int
    n_z: int
    n_messages: int = 2
    n_labels: int = 2

    @property
    def omega_size(self) -> int:
        return self.n_x * self.n_z


@dataclass(frozen=True)
class ExactMetrics:
    accuracy: float
    correct: int
    decoder: tuple[int, ...]
    has_tie: bool
    mutual_information: float


def task_label(task: Sequence[int], x: int, z: int, space: Space) -> int:
    return task[x * space.n_z + z]


def all_tasks(space: Space) -> Iterable[tuple[int, ...]]:
    for task in itertools.product(range(space.n_labels), repeat=space.omega_size):
        if len(set(task)) > 1:
            yield task


def is_surjective(sender: Sequence[int], n_messages: int = 2) -> bool:
    return set(sender) == set(range(n_messages))


def all_senders(space: Space) -> list[tuple[int, ...]]:
    return [
        sender
        for sender in itertools.product(range(space.n_messages), repeat=space.n_x)
        if is_surjective(sender, space.n_messages)
    ]


def complement(sender: Sequence[int]) -> tuple[int, ...]:
    """Binary compatibility helper."""
    return tuple(1 - value for value in sender)


def message_relabelings(sender: Sequence[int], n_messages: int | None = None) -> tuple[tuple[int, ...], ...]:
    sender = tuple(sender)
    if n_messages is None:
        n_messages = max(sender) + 1
    return tuple(
        tuple(permutation[value] for value in sender)
        for permutation in itertools.permutations(range(n_messages))
    )


def canonical_sender(sender: Sequence[int], n_messages: int | None = None) -> tuple[int, ...]:
    return min(message_relabelings(sender, n_messages))


def canonical_case(task: Sequence[int], sender: Sequence[int], rewritten: Sequence[int], space: Space) -> tuple:
    task = tuple(task)
    sender = tuple(sender)
    rewritten = tuple(rewritten)
    candidates = []
    for label_permutation in itertools.permutations(range(space.n_labels)):
        t = tuple(label_permutation[y] for y in task)
        for message_permutation in itertools.permutations(range(space.n_messages)):
            s = tuple(message_permutation[m] for m in sender)
            sp = tuple(message_permutation[m] for m in rewritten)
            candidates.append((t, s, sp))
    return min(candidates)


def counts_for(task: Sequence[int], sender: Sequence[int], space: Space) -> list[list[list[int]]]:
    counts = [
        [[0 for _ in range(space.n_labels)] for _ in range(space.n_z)]
        for _ in range(space.n_messages)
    ]
    for x in range(space.n_x):
        for z in range(space.n_z):
            counts[sender[x]][z][task_label(task, x, z, space)] += 1
    return counts


def mutual_information(task: Sequence[int], sender: Sequence[int], space: Space) -> float:
    # I(Y;M|Z), with uniform (x,z).
    result = 0.0
    for z in range(space.n_z):
        joint = [[0 for _ in range(space.n_labels)] for _ in range(space.n_messages)]
        for x in range(space.n_x):
            joint[sender[x]][task_label(task, x, z, space)] += 1
        for m in range(space.n_messages):
            pm = sum(joint[m]) / space.n_x
            for y in range(space.n_labels):
                pmy = joint[m][y] / space.n_x
                py = sum(joint[mm][y] for mm in range(space.n_messages)) / space.n_x
                if pmy > 0:
                    result += (pmy * math.log2(pmy / (pm * py))) / space.n_z
    return result


def exact_metrics(task: Sequence[int], sender: Sequence[int], space: Space) -> ExactMetrics:
    counts = counts_for(task, sender, space)
    decoder = []
    correct = 0
    has_tie = False
    for m in range(space.n_messages):
        for z in range(space.n_z):
            row = counts[m][z]
            best = max(row)
            winners = [y for y, count in enumerate(row) if count == best]
            has_tie |= len(winners) != 1
            decoder.append(winners[0])
            correct += best
    return ExactMetrics(
        accuracy=correct / space.omega_size,
        correct=correct,
        decoder=tuple(decoder),
        has_tie=has_tie,
        mutual_information=mutual_information(task, sender, space),
    )


def decoder_accuracy(task: Sequence[int], sender: Sequence[int], decoder: Sequence[int], space: Space) -> float:
    correct = 0
    for x in range(space.n_x):
        for z in range(space.n_z):
            prediction = decoder[sender[x] * space.n_z + z]
            correct += prediction == task_label(task, x, z, space)
    return correct / space.omega_size


def rewrite_distance(sender: Sequence[int], rewritten: Sequence[int]) -> int:
    return sum(a != b for a, b in zip(sender, rewritten))


def partition_distance(sender: Sequence[int], rewritten: Sequence[int], n_messages: int | None = None) -> int:
    return min(rewrite_distance(sender, oriented) for oriented in message_relabelings(rewritten, n_messages))


def rewrite_kind(task: Sequence[int], sender: Sequence[int], rewritten: Sequence[int], space: Space) -> str:
    changed = [x for x, (a, b) in enumerate(zip(sender, rewritten)) if a != b]
    if len(changed) == 1:
        return "E1"
    if len(changed) == 2:
        return "E2"
    signatures = [tuple(task_label(task, x, z, space) for z in range(space.n_z)) for x in changed]
    if len(set(signatures)) == 1:
        return "EG"
    return "EP"


def smoothed_logits(task: Sequence[int], sender: Sequence[int], space: Space, alpha: float) -> list[list[list[float]]]:
    counts = counts_for(task, sender, space)
    logits = []
    for m in range(space.n_messages):
        per_z = []
        for z in range(space.n_z):
            total = sum(counts[m][z]) + alpha * space.n_labels
            per_z.append([math.log((count + alpha) / total) for count in counts[m][z]])
        logits.append(per_z)
    return logits


def clone_logits(logits: Sequence[Sequence[Sequence[float]]]) -> list[list[list[float]]]:
    return [[list(row) for row in per_z] for per_z in logits]


def probabilities(row: Sequence[float]) -> list[float]:
    peak = max(row)
    exp_values = [math.exp(value - peak) for value in row]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def cross_entropy(task: Sequence[int], sender: Sequence[int], logits: Sequence[Sequence[Sequence[float]]], space: Space) -> float:
    loss = 0.0
    for x in range(space.n_x):
        for z in range(space.n_z):
            probs = probabilities(logits[sender[x]][z])
            loss -= math.log(max(probs[task_label(task, x, z, space)], 1e-300))
    return loss / space.omega_size


def align_partition(
    task: Sequence[int], current_sender: Sequence[int], target_partition: Sequence[int],
    space: Space, alpha: float = 0.5, primary: str = "accuracy_then_ce",
) -> tuple[int, ...]:
    """Choose the message naming of a target partition most compatible with R(current)."""
    current = exact_metrics(task, current_sender, space)
    base_logits = smoothed_logits(task, current_sender, space, alpha)
    orientations = message_relabelings(target_partition, space.n_messages)
    if primary == "cross_entropy":
        return min(
            orientations,
            key=lambda candidate: (
                cross_entropy(task, candidate, base_logits, space),
                -decoder_accuracy(task, candidate, current.decoder, space),
                candidate,
            ),
        )
    if primary != "accuracy_then_ce":
        raise ValueError(f"unknown alignment primary: {primary}")
    return max(
        orientations,
        key=lambda candidate: (
            decoder_accuracy(task, candidate, current.decoder, space),
            -cross_entropy(task, candidate, base_logits, space),
            tuple(-value for value in candidate),
        ),
    )


def optimal_cross_entropy(task: Sequence[int], sender: Sequence[int], space: Space) -> float:
    """Unregularized Bayes cross entropy H(Y|M,Z), in nats."""
    counts = counts_for(task, sender, space)
    loss = 0.0
    for m in range(space.n_messages):
        for z in range(space.n_z):
            total = sum(counts[m][z])
            for count in counts[m][z]:
                if count:
                    loss -= count * math.log(count / total)
    return loss / space.omega_size


def gradient_step(task: Sequence[int], sender: Sequence[int], logits: list[list[list[float]]], space: Space, learning_rate: float) -> None:
    grad = [
        [[0.0 for _ in range(space.n_labels)] for _ in range(space.n_z)]
        for _ in range(space.n_messages)
    ]
    for x in range(space.n_x):
        for z in range(space.n_z):
            m = sender[x]
            y = task_label(task, x, z, space)
            probs = probabilities(logits[m][z])
            for label in range(space.n_labels):
                grad[m][z][label] += (probs[label] - (label == y)) / space.omega_size
    for m in range(space.n_messages):
        for z in range(space.n_z):
            for y in range(space.n_labels):
                logits[m][z][y] -= learning_rate * grad[m][z][y]


def maturation_curve(
    task: Sequence[int], sender: Sequence[int], rewritten: Sequence[int], space: Space,
    alpha: float, learning_rate: float, steps: Sequence[int],
) -> dict[int, float]:
    base_logits = smoothed_logits(task, sender, space, alpha)
    left = clone_logits(base_logits)
    right = clone_logits(base_logits)
    curve = {}
    maximum = max(steps)
    selected = set(steps)
    for t in range(maximum + 1):
        if t in selected:
            curve[t] = cross_entropy(task, sender, left, space) - cross_entropy(task, rewritten, right, space)
        if t < maximum:
            gradient_step(task, sender, left, space, learning_rate)
            gradient_step(task, rewritten, right, space, learning_rate)
    return curve


def find_h1_cases(space: Space) -> tuple[list[dict], dict]:
    senders = all_senders(space)
    target_partitions = sorted({canonical_sender(sender, space.n_messages) for sender in senders})
    cases = []
    raw_pairs = 0
    canonical_keys = set()
    task_count = 0
    solvable_task_count = 0
    for task in all_tasks(space):
        task_count += 1
        task_metrics = {sender: exact_metrics(task, sender, space) for sender in senders}
        if max(metrics.accuracy for metrics in task_metrics.values()) < 1.0 - EPS:
            continue
        solvable_task_count += 1
        for sender in senders:
            base = task_metrics[sender]
            for target_partition in target_partitions:
                if canonical_sender(sender, space.n_messages) == target_partition:
                    continue
                rewritten = align_partition(task, sender, target_partition, space)
                raw_pairs += 1
                target = task_metrics[rewritten]
                immediate_accuracy = decoder_accuracy(task, rewritten, base.decoder, space)
                c0_accuracy = immediate_accuracy - base.accuracy
                cinf_accuracy = target.accuracy - base.accuracy
                delta_mi = target.mutual_information - base.mutual_information
                if c0_accuracy <= EPS and cinf_accuracy > EPS and delta_mi > EPS:
                    key = canonical_case(task, sender, rewritten, space)
                    canonical_keys.add(key)
                    cases.append({
                        "task": list(task),
                        "sender": list(sender),
                        "rewritten": list(rewritten),
                        "target_partition": list(target_partition),
                        "rewrite_kind": rewrite_kind(task, sender, rewritten, space),
                        "distance": rewrite_distance(sender, rewritten),
                        "partition_distance": partition_distance(sender, rewritten, space.n_messages),
                        "base_accuracy": base.accuracy,
                        "immediate_accuracy": immediate_accuracy,
                        "target_accuracy": target.accuracy,
                        "c0_accuracy": c0_accuracy,
                        "cinf_accuracy": cinf_accuracy,
                        "base_mi": base.mutual_information,
                        "target_mi": target.mutual_information,
                        "delta_mi": delta_mi,
                        "base_decoder": list(base.decoder),
                        "target_decoder": list(target.decoder),
                        "base_tie": base.has_tie,
                        "target_tie": target.has_tie,
                        "canonical_key": json.dumps(key, separators=(",", ":")),
                    })
    summary = {
        "space": asdict(space),
        "task_count": task_count,
        "solvable_task_count": solvable_task_count,
        "sender_count": len(senders),
        "raw_candidate_pairs": raw_pairs,
        "raw_h1_cases": len(cases),
        "canonical_h1_cases": len(canonical_keys),
        "no_tie_h1_cases": sum(not case["base_tie"] and not case["target_tie"] for case in cases),
        "strict_negative_h1_cases": sum(case["c0_accuracy"] < -EPS for case in cases),
    }
    return cases, summary


def minimax_barrier(task: Sequence[int], start: Sequence[int], space: Space, max_partition_distance: int) -> float | None:
    nodes = sorted({canonical_sender(sender, space.n_messages) for sender in all_senders(space)})
    start_node = canonical_sender(start, space.n_messages)
    sufficient = {node for node in nodes if exact_metrics(task, node, space).accuracy >= 1.0 - EPS}
    if not sufficient:
        return None
    best = {node: math.inf for node in nodes}
    best[start_node] = 0.0
    unvisited = set(nodes)
    while unvisited:
        current = min(unvisited, key=lambda node: best[node])
        if math.isinf(best[current]):
            break
        unvisited.remove(current)
        if current in sufficient:
            return best[current]
        current_metrics = exact_metrics(task, current, space)
        for target in list(unvisited):
            distance = partition_distance(current, target, space.n_messages)
            if distance == 0 or distance > max_partition_distance:
                continue
            immediate_candidates = [
                decoder_accuracy(task, oriented, current_metrics.decoder, space)
                for oriented in message_relabelings(target, space.n_messages)
            ]
            immediate = max(immediate_candidates)
            edge_cost = max(0.0, current_metrics.accuracy - immediate)
            candidate = max(best[current], edge_cost)
            if candidate < best[target]:
                best[target] = candidate
    return None


def minimax_barrier_cross_entropy(
    task: Sequence[int], start: Sequence[int], space: Space,
    max_partition_distance: int, alpha: float = 0.5,
) -> float | None:
    """Minimum possible maximum immediate CE increase on a path to sufficiency."""
    nodes = sorted({canonical_sender(sender, space.n_messages) for sender in all_senders(space)})
    start_node = canonical_sender(start, space.n_messages)
    sufficient = {node for node in nodes if exact_metrics(task, node, space).accuracy >= 1.0 - EPS}
    if not sufficient:
        return None
    best = {node: math.inf for node in nodes}
    best[start_node] = 0.0
    unvisited = set(nodes)
    while unvisited:
        current = min(unvisited, key=lambda node: best[node])
        if math.isinf(best[current]):
            break
        unvisited.remove(current)
        if current in sufficient:
            return best[current]
        logits = smoothed_logits(task, current, space, alpha)
        current_loss = cross_entropy(task, current, logits, space)
        for target in list(unvisited):
            distance = partition_distance(current, target, space.n_messages)
            if distance == 0 or distance > max_partition_distance:
                continue
            aligned = align_partition(task, current, target, space, alpha, primary="cross_entropy")
            edge_cost = max(0.0, cross_entropy(task, aligned, logits, space) - current_loss)
            candidate = max(best[current], edge_cost)
            if candidate < best[target]:
                best[target] = candidate
    return None


def source_hash(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
