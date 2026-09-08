"""Independent scalar BCE, max-flow and Hungarian checks; no metrics imports."""
import json
import math
from collections import deque
from pathlib import Path

import numpy as np


def flow_size(edges):
    n, m = edges.shape
    sink = n + m + 1
    capacity = np.zeros((sink + 1, sink + 1), dtype=int)
    capacity[0, 1:n + 1] = 1
    capacity[1:n + 1, n + 1:sink] = edges
    capacity[n + 1:sink, sink] = 1
    total = 0
    while True:
        parent = {0: -1}
        queue = deque([0])
        while queue and sink not in parent:
            a = queue.popleft()
            for b in range(sink + 1):
                if capacity[a, b] > 0 and b not in parent:
                    parent[b] = a
                    queue.append(b)
        if sink not in parent:
            return total
        b = sink
        while b:
            a = parent[b]
            capacity[a, b] -= 1
            capacity[b, a] += 1
            b = a
        total += 1


def hungarian(cost):
    n = len(cost)
    u, v = [0.] * (n + 1), [0.] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        mins, used = [math.inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = float(cost[i0 - 1, j - 1]) - u[i0] - v[j]
                    if cur < mins[j]:
                        mins[j], way[j] = cur, j0
                    if mins[j] < delta:
                        delta, j1 = mins[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    mins[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return sum(float(cost[i, j]) for i, j in enumerate(assignment)), assignment


def scalar_tables(logits, group_q, group_v, group_y):
    loss = np.zeros((2, 16, 16))
    accuracy = np.zeros_like(loss)
    for q in range(2):
        for v in range(16):
            g = next(i for i in range(len(group_q)) if group_q[i] == q and group_v[i] == v)
            for m in range(16):
                terms, correct = [], 0
                for r in range(16):
                    bits = []
                    for b in range(4):
                        z = float(logits[q, r, m, b])
                        y = int(group_y[g, r, b])
                        terms.append(max(z, 0.) - z * y + math.log1p(math.exp(-abs(z))))
                        bits.append((z >= 0) == bool(y))
                    correct += all(bits)
                loss[q, v, m] = math.fsum(terms) / 64
                accuracy[q, v, m] = correct / 16
    return loss, accuracy


def audit(output):
    output = Path(output)
    reports = []
    for path in sorted(output.glob('*_tables.npz')):
        tag = path.name.removesuffix('_tables.npz')
        report = json.loads((output / f'{tag}.json').read_text())
        with np.load(path, allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        loss, accuracy = scalar_tables(a['logits'], a['group_q'], a['group_v'], a['group_y'])
        alternate_loss, _ = scalar_tables(a['numpy_logits'], a['group_q'], a['group_v'], a['group_y'])
        error = float(np.max(np.abs(loss - a['loss'])))
        alternate_error = float(np.max(np.abs(alternate_loss - a['loss'])))
        assert error <= 1e-6 and alternate_error <= 1e-6
        assert np.array_equal(accuracy, a['accuracy'])
        conditions = []
        for q in range(2):
            expected = report['conditions'][q]
            counts = np.zeros((16, 16), dtype=int)
            for g in range(len(a['group_q'])):
                if a['group_q'][g] == q:
                    counts[a['group_v'][g], a['group_message'][g]] += 1
            assert np.array_equal(counts, a['counts'][q])
            regret = loss[q] - loss[q].min(axis=1, keepdims=True)
            edges = regret <= 1e-4
            assert np.array_equal(edges, expected['near_optimal_edges'])
            alternate_regret = alternate_loss[q] - alternate_loss[q].min(axis=1, keepdims=True)
            assert np.array_equal(edges, alternate_regret <= 1e-4), 'Forward implementations disagree on edges'
            k = flow_size(edges)
            assert k == expected['K']
            assigned = [j for j in expected['assignment'] if j >= 0]
            assert len(assigned) == k and len(set(assigned)) == k
            assert all(j < 0 or edges[i, j] for i, j in enumerate(expected['assignment']))
            if k < 16:
                vs = expected['hall_V']
                neighbors = sorted({j for i in vs for j in range(16) if edges[i, j]})
                assert neighbors == expected['hall_messages'] and len(neighbors) < len(vs)
            # Certify the exact minimum bottleneck using feasible/nonfeasible adjacent levels.
            delta = expected['delta_star']
            original_regret = a['loss'][q] - a['loss'][q].min(axis=1, keepdims=True)
            assert flow_size(original_regret <= delta) == 16
            assert flow_size(original_regret < delta) < 16
            bott = expected['bottleneck_assignment']
            assert len(set(bott)) == 16 and max(original_regret[i, j] for i, j in enumerate(bott)) == delta
            weights = counts.sum(axis=1) / counts.sum()
            optimum, _ = hungarian(weights[:, None] * loss[q])
            assert abs(optimum - expected['minimum_injective_BCE']) <= 1e-10
            allocation = expected['minimum_cost_assignment']
            assert len(set(allocation)) == 16
            assert abs(sum(weights[i] * loss[q, i, j] for i, j in enumerate(allocation)) - optimum) <= 1e-10
            f = counts[edges].sum() / counts.sum()
            assert f == expected['F']
            low = flow_size(regret < 1e-4 - 1e-10)
            high = flow_size(regret <= 1e-4 + 1e-10)
            assert (low, high) == (expected['K_exclude_boundary'], expected['K_include_boundary'])
            uncertain = (low < 16) != (high < 16)
            assert uncertain == expected['numerically_uncertain']
            assert (k < 16 and not uncertain) == expected['positive']
            assert (k < 16 and f >= .99 and not uncertain) == expected['behavior_consistent']
            assert flow_size(accuracy[q] == accuracy[q].max(axis=1, keepdims=True)) == expected['K_accuracy']
            mask = a['group_q'] == q
            fit = np.mean(a['group_message'][mask] == a['historical_target'][mask])
            assert fit == expected['historical_target_fit']
            for key, value in {
                'A_code': counts.max(axis=0).sum() / counts.sum(),
                'current_accuracy': (counts * accuracy[q]).sum() / counts.sum(),
                'current_mean_regret': (counts * regret).sum() / counts.sum(),
                'minimum_mean_cost': optimum - weights @ loss[q].min(axis=1),
            }.items():
                assert abs(float(value) - expected[key]) <= 1e-10
            # Independent renaming check includes cost, bottleneck and cardinality.
            assert flow_size(edges[:, ::-1]) == k
            renamed_cost, _ = hungarian((weights[:, None] * loss[q])[:, ::-1])
            assert abs(renamed_cost - optimum) <= 1e-10
            assert flow_size(original_regret[:, ::-1] <= delta) == 16
            assert flow_size(original_regret[:, ::-1] < delta) < 16
            conditions.append({'q': q, 'K': k, 'F': float(f), 'hall_verified': True,
                               'assignment_optima_verified': True, 'renaming_passed': True})
        reports.append({'model': tag, 'passed': True, 'scalar_BCE_max_error': error,
                        'numpy_forward_BCE_max_error': alternate_error, 'conditions': conditions})
    assert len(reports) == 6
    result = {'passed': True, 'model_count': 6, 'models': reports,
              'scope': 'Independent algorithms by same agent; not external replication'}
    (output / 'independent_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result
