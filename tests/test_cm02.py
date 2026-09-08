import itertools
import unittest

import numpy as np

from cm02.audit import flow_size, hungarian, scalar_tables
from cm02.metrics import analyze, bottleneck, matching, min_assignment


class CM02Tests(unittest.TestCase):
    def test_sufficient_receiver(self):
        # A synthetic receiver strongly decodes y=(V+r)%16 using m=V.
        logits = np.empty((2, 16, 16, 4))
        labels = []
        for q in range(2):
            for v in range(16):
                labels.append([[(v + r) % 16 >> b & 1 for b in range(4)] for r in range(16)])
                for r in range(16):
                    logits[q, r, v] = [8 if ((v + r) % 16 >> b & 1) else -8 for b in range(4)]
        loss, accuracy = scalar_tables(logits, np.repeat([0, 1], 16), np.tile(np.arange(16), 2), np.array(labels))
        self.assertEqual(matching(loss[0] - loss[0].min(axis=1, keepdims=True) <= 1e-4)['K'], 16)
        np.testing.assert_array_equal(np.diag(accuracy[0]), np.ones(16))

    def test_collapsed_and_tied_tables(self):
        collapsed = np.ones((16, 16)); collapsed[:, 0] = 0
        result = matching(collapsed <= 1e-4)
        self.assertEqual(result['K'], 1)
        self.assertGreater(len(result['hall_V']), len(result['hall_messages']))
        self.assertEqual(matching(np.ones((16, 16), dtype=bool))['K'], 16)

    def test_algorithms_against_small_exhaustive(self):
        rng = np.random.default_rng(42)
        for n in range(2, 6):
            for _ in range(4):
                cost = rng.integers(0, 20, (n, n)).astype(float) / 7
                perms = list(itertools.permutations(range(n)))
                exact = min(sum(cost[i, p[i]] for i in range(n)) for p in perms)
                value, assignment = min_assignment(cost)
                self.assertAlmostEqual(value, exact)
                self.assertAlmostEqual(hungarian(cost)[0], exact)
                self.assertEqual(len(set(assignment)), n)
                delta = min(max(cost[i, p[i]] for i in range(n)) for p in perms)
                self.assertEqual(bottleneck(cost)[0], delta)
                edges = cost <= .7
                k = max(sum(edges[i, p[i]] for i in range(n)) for p in perms)
                self.assertEqual(matching(edges)['K'], k)
                self.assertEqual(flow_size(edges), k)

    def test_boundary_changes_classification(self):
        loss = np.full((16, 16), 1.)
        np.fill_diagonal(loss, 0.)
        loss[1, 0], loss[1, 1] = 0., 1e-4
        result = analyze(loss, np.zeros_like(loss), np.eye(16, dtype=int))
        self.assertTrue(result['numerically_uncertain'])
        self.assertFalse(result['positive'])

    def test_behavior_and_renaming(self):
        loss = np.ones((16, 16)); loss[:, 0] = 0
        counts = np.zeros((16, 16), dtype=int); counts[:, 0] = 1
        result = analyze(loss, np.zeros_like(loss), counts)
        self.assertTrue(result['behavior_consistent'])
        self.assertEqual(result['F'], 1.)
        renamed = analyze(loss[:, ::-1], np.zeros_like(loss), counts[:, ::-1])
        for key in ('K', 'F', 'delta_star', 'minimum_mean_cost'):
            self.assertEqual(result[key], renamed[key])
        counts[:, 0] = 0; counts[:, 1] = 1
        self.assertFalse(analyze(loss, np.zeros_like(loss), counts)['behavior_consistent'])


if __name__ == '__main__':
    unittest.main()
