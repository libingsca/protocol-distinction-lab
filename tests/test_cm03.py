import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cm03.audit import scalar_metrics
from cm03.protocols import BRANCHES, make_random, validate_training
from cm03.training import compare, metrics, state_hash, train_branch


class TinyReceiver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, m, q, r):
        return self.linear(m)


class CM03Tests(unittest.TestCase):
    def test_random_preserves_changed_set_and_sufficiency(self):
        original = np.arange(16) % 4
        target = np.arange(16)
        for seed in (73001, 73002, 73003):
            result, record = make_random(original, target, seed)
            self.assertEqual(sorted(result), list(range(16)))
            np.testing.assert_array_equal(result != original, target != original)
            self.assertFalse(np.array_equal(result, target))
            repeated, repeated_record = make_random(original, target, seed)
            np.testing.assert_array_equal(result, repeated)
            self.assertEqual(record, repeated_record)

    def test_impossible_control_stops(self):
        with self.assertRaises(ValueError):
            make_random(np.arange(16), np.arange(16), 0, max_attempts=3)

    def test_missing_r_rejected(self):
        # Shape-valid data still fails complete-r support validation.
        data = {'x': np.zeros((4096, 8)), 'q': np.zeros(4096),
                'r': np.zeros((4096, 4)), 'y': np.zeros((4096, 4))}
        with self.assertRaises(AssertionError):
            validate_training(data)

    def test_branch_resets_optimizer_and_keeps_origin(self):
        torch.manual_seed(903)
        receiver = TinyReceiver()
        origin = copy.deepcopy(receiver.state_dict())
        original_hash = state_hash(origin)
        m = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 32
        q, r, y = torch.tensor([0, 1]*4), torch.zeros((8, 4)), (m > .5).float()
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / 'weights.pt'
            first, logits1 = train_branch(receiver, origin, m, q, r, y, 3, dest)
            second, logits2 = train_branch(receiver, origin, m, q, r, y, 3)
            self.assertEqual(first, second)
            np.testing.assert_array_equal(logits1, logits2)
            self.assertEqual(state_hash(origin), original_hash)
            self.assertNotEqual(first['curve'][-1]['parameter_hash'], original_hash)
            state = torch.load(dest, weights_only=True)
            self.assertEqual(state['initial_optimizer']['state'], {})
            self.assertTrue(all(int(s['step']) == 3 for s in state['final_optimizer']['state'].values()))

    def test_scalar_metrics_at_extreme_logits(self):
        z = torch.tensor([[100., -100., .01, -.01], [-1., 2., 3., -4.]])
        y = torch.tensor([[1., 0., 0., 1.], [0., 1., 1., 0.]])
        q = torch.tensor([0, 1])
        expected = scalar_metrics(z.numpy(), y.numpy(), q.numpy())
        actual = metrics(z, y, q)
        self.assertAlmostEqual(expected['BCE'], actual['BCE'], places=6)
        self.assertEqual(expected['accuracy'], actual['accuracy'])

    def test_fixed_endpoint_all_gates_and_secondary(self):
        original = [{'BCE': .5, 'accuracy': .2} for _ in range(101)]
        target = [{'BCE': .6, 'accuracy': .1}] + [{'BCE': .4, 'accuracy': .3} for _ in range(100)]
        branches = {name: {'curve': copy.deepcopy(original if name == 'original' else target)} for name in BRANCHES}
        result = compare(branches)
        self.assertTrue(result['target_passes'])
        self.assertFalse(result['selection_advantage_passes'])
        self.assertEqual(result['comparisons']['target']['first_C_positive'], 1)
        branches['target']['curve'][100]['accuracy'] = .2
        self.assertFalse(compare(branches)['target_passes'])
        branches['target']['curve'][100]['accuracy'] = .3
        branches['target']['curve'][0]['BCE'] = .49
        self.assertFalse(compare(branches)['target_passes'])


if __name__ == '__main__':
    unittest.main()
