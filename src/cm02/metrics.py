from collections import deque

import numpy as np


def matching(edges):
    """Deterministic augmenting paths, with a Hall witness on failure."""
    edges = np.asarray(edges, dtype=bool)
    n, m = edges.shape
    right = [-1] * m

    def augment(v, seen):
        for j in np.flatnonzero(edges[v]):
            if j in seen:
                continue
            seen.add(j)
            if right[j] < 0 or augment(right[j], seen):
                right[j] = v
                return True
        return False

    for v in range(n):
        augment(v, set())
    left = [-1] * n
    for j, v in enumerate(right):
        if v >= 0:
            left[v] = j
    reachable_v = {v for v in range(n) if left[v] < 0}
    reachable_m = set()
    queue = deque(sorted(reachable_v))
    while queue:
        v = queue.popleft()
        for j in np.flatnonzero(edges[v]):
            reachable_m.add(int(j))
            if right[j] >= 0 and right[j] not in reachable_v:
                reachable_v.add(right[j])
                queue.append(right[j])
    k = sum(j >= 0 for j in left)
    if k < n:
        assert len(reachable_v) > len(reachable_m)
    return {'K': k, 'assignment': left, 'hall_V': sorted(reachable_v),
            'hall_messages': sorted(reachable_m)}


def bottleneck(regret):
    levels = np.unique(regret)
    lo, hi = 0, len(levels) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if matching(regret <= levels[mid])['K'] == len(regret):
            hi = mid
        else:
            lo = mid + 1
    delta = float(levels[lo])
    return delta, matching(regret <= delta)['assignment']


def min_assignment(cost):
    """Subset DP; reconstruct lexicographically first optimal assignment."""
    n = len(cost)
    full = (1 << n) - 1
    dp = np.full(full + 1, np.inf)
    dp[full] = 0.0
    choice = np.full(full + 1, -1, dtype=int)
    for mask in range(full - 1, -1, -1):
        row = mask.bit_count()
        for col in range(n):
            if not (mask >> col) & 1:
                value = float(cost[row, col]) + dp[mask | (1 << col)]
                if value < dp[mask]:
                    dp[mask], choice[mask] = value, col
    assignment, mask = [], 0
    for _ in range(n):
        col = int(choice[mask])
        assignment.append(col)
        mask |= 1 << col
    return float(dp[0]), assignment


def analyze(loss, accuracy, counts, epsilon=1e-4, tolerance=1e-10):
    regret = loss - loss.min(axis=1, keepdims=True)
    edges = regret <= epsilon
    match = matching(edges)
    lower = matching(regret < epsilon - tolerance)['K']
    upper = matching(regret <= epsilon + tolerance)['K']
    uncertain = (lower < 16) != (upper < 16)
    weights = counts.sum(axis=1) / counts.sum()
    total, assignment = min_assignment(weights[:, None] * loss)
    delta, bottle_assignment = bottleneck(regret)
    f = float(counts[edges].sum() / counts.sum())
    return {
        **match, 'K_exclude_boundary': lower, 'K_include_boundary': upper,
        'boundary_edges': np.argwhere(np.abs(regret - epsilon) <= tolerance).tolist(),
        'numerically_uncertain': uncertain,
        'positive': match['K'] < 16 and not uncertain, 'F': f,
        'behavior_consistent': match['K'] < 16 and f >= .99 and not uncertain,
        'epsilon_distance_min': float(np.min(np.abs(regret - epsilon))),
        'delta_star': delta, 'bottleneck_assignment': bottle_assignment,
        'minimum_injective_BCE': total, 'minimum_cost_assignment': assignment,
        'independent_minimum_BCE': float(weights @ loss.min(axis=1)),
        'minimum_mean_cost': float(total - weights @ loss.min(axis=1)),
        'K_accuracy': matching(accuracy == accuracy.max(axis=1, keepdims=True))['K'],
        'sender_message_count': int((counts.sum(axis=0) > 0).sum()),
        'current_accuracy': float((counts * accuracy).sum() / counts.sum()),
        'A_code': float(counts.max(axis=0).sum() / counts.sum()),
        'current_mean_regret': float((counts * regret).sum() / counts.sum()),
        'near_optimal_edges': edges.astype(int).tolist(),
    }
