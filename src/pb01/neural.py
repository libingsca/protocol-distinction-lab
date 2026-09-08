from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class MLPReceiver:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray

    @classmethod
    def random(cls, n_messages: int, n_labels: int, hidden: int, seed: int) -> "MLPReceiver":
        rng = np.random.default_rng(seed)
        return cls(
            w1=rng.normal(0.0, 0.3, (n_messages, hidden)),
            b1=np.zeros(hidden),
            w2=rng.normal(0.0, 0.3, (hidden, n_labels)),
            b2=np.zeros(n_labels),
        )

    def clone(self) -> "MLPReceiver":
        return MLPReceiver(*(value.copy() for value in (self.w1, self.b1, self.w2, self.b2)))

    def forward(self, messages: np.ndarray):
        x = np.eye(self.w1.shape[0])[messages]
        return self.forward_distributions(x)

    def forward_distributions(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        hidden = np.tanh(x @ self.w1 + self.b1)
        logits = hidden @ self.w2 + self.b2
        return x, hidden, softmax(logits)

    def loss_accuracy(self, messages, labels) -> tuple[float, float]:
        messages = np.asarray(messages, dtype=int)
        labels = np.asarray(labels, dtype=int)
        _, _, probs = self.forward(messages)
        loss = -np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-300)).mean()
        accuracy = (probs.argmax(axis=1) == labels).mean()
        return float(loss), float(accuracy)

    def loss_accuracy_soft(self, message_distributions, labels) -> tuple[float, float]:
        labels = np.asarray(labels, dtype=int)
        _, _, probs = self.forward_distributions(np.asarray(message_distributions, dtype=float))
        loss = -np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-300)).mean()
        accuracy = (probs.argmax(axis=1) == labels).mean()
        return float(loss), float(accuracy)

    def hard_step(self, messages, labels, learning_rate: float) -> None:
        messages = np.asarray(messages, dtype=int)
        labels = np.asarray(labels, dtype=int)
        x, hidden, probs = self.forward(messages)
        grad_logits = probs
        grad_logits[np.arange(len(labels)), labels] -= 1.0
        grad_logits /= len(labels)
        grad_w2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = (grad_logits @ self.w2.T) * (1.0 - hidden * hidden)
        grad_w1 = x.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2

    def soft_target_step(self, targets: np.ndarray, weights: np.ndarray, learning_rate: float) -> float:
        messages = np.arange(self.w1.shape[0])
        x, hidden, probs = self.forward(messages)
        weights = weights / weights.sum()
        grad_logits = (probs - targets) * weights[:, None]
        grad_w2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = (grad_logits @ self.w2.T) * (1.0 - hidden * hidden)
        grad_w1 = x.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)
        self.w1 -= learning_rate * grad_w1
        self.b1 -= learning_rate * grad_b1
        self.w2 -= learning_rate * grad_w2
        self.b2 -= learning_rate * grad_b2
        return float(-(weights[:, None] * targets * np.log(np.maximum(probs, 1e-300))).sum())


def conditional_targets(messages, labels, n_messages: int, n_labels: int, alpha: float):
    counts = np.zeros((n_messages, n_labels), dtype=float)
    for message, label in zip(messages, labels):
        counts[message, label] += 1.0
    targets = np.empty_like(counts)
    weights = np.empty(n_messages, dtype=float)
    for message in range(n_messages):
        total = counts[message].sum()
        targets[message] = (counts[message] + alpha) / (total + alpha * n_labels)
        # An unused slot receives a low-weight uniform calibration target.
        weights[message] = total if total > 0 else 1.0
    return targets, weights


def pretrain_receiver(
    messages, labels, n_messages: int, n_labels: int, hidden: int, seed: int,
    alpha: float, steps: int, learning_rate: float,
) -> tuple[MLPReceiver, float]:
    model = MLPReceiver.random(n_messages, n_labels, hidden, seed)
    targets, weights = conditional_targets(messages, labels, n_messages, n_labels, alpha)
    loss = 0.0
    for _ in range(steps):
        loss = model.soft_target_step(targets, weights, learning_rate)
    _, _, probs = model.forward(np.arange(n_messages))
    max_error = float(np.max(np.abs(probs - targets)))
    return model, max_error


def best_aligned_mapping(model: MLPReceiver, partition, labels, n_messages: int):
    best = None
    for permutation in itertools.permutations(range(n_messages)):
        mapping = tuple(permutation[m] for m in partition)
        loss, accuracy = model.loss_accuracy(mapping, labels)
        candidate = (loss, -accuracy, mapping)
        if best is None or candidate < best:
            best = candidate
    return best[2]


def adapt_curve(model: MLPReceiver, messages, labels, steps, learning_rate: float):
    selected = set(steps)
    result = {}
    maximum = max(steps)
    for step in range(maximum + 1):
        if step in selected:
            result[step] = model.loss_accuracy(messages, labels)
        if step < maximum:
            model.hard_step(messages, labels, learning_rate)
    return result
