from __future__ import annotations

import itertools
from typing import Sequence

from .core import Space, task_label


def brute_force_best_accuracy(task: Sequence[int], sender: Sequence[int], space: Space) -> tuple[float, tuple[int, ...], bool]:
    """Independent definition: enumerate every receiver table instead of using counts."""
    best_correct = -1
    winners = []
    receiver_cells = space.n_messages * space.n_z
    for decoder in itertools.product(range(space.n_labels), repeat=receiver_cells):
        correct = 0
        for x in range(space.n_x):
            for z in range(space.n_z):
                prediction = decoder[sender[x] * space.n_z + z]
                correct += prediction == task_label(task, x, z, space)
        if correct > best_correct:
            best_correct = correct
            winners = [decoder]
        elif correct == best_correct:
            winners.append(decoder)
    return best_correct / space.omega_size, winners[0], len(winners) > 1

