import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cm04.core import Receiver, compare_seed, features, paired_state, state_hash, summarize, symbolic_check, table, train


class CM04Tests(unittest.TestCase):
    def test_symbolic_and_exact_information(self):
        data = table()
        mapping = [(5*v+3) % 16 for v in range(16)]
        result = symbolic_check(data, {'E0': list(range(16)), 'other': mapping})
        self.assertEqual(result['other']['symbolic_accuracy'], 1.)
        self.assertEqual(result['other']['I_V_M_given_q'], 4.)
        with self.assertRaises(AssertionError):
            symbolic_check(data, {'bad': [0]*16})

    def test_feature_layout(self):
        data = table()
        mapping = list(range(16))
        for rep in ('bit', 'onehot'):
            x = features(data, mapping, rep)
            self.assertEqual(x.shape, (512, 22))
            np.testing.assert_array_equal(x[:, 16:18].sum(1), np.ones(512))
        x = features(data, mapping, 'bit')
        np.testing.assert_array_equal(x[:, 4:8], np.zeros((512, 4)))
        np.testing.assert_array_equal(x[:, 8:12], np.ones((512, 4)))

    def test_permutation_direction_and_seed_copy(self):
        torch.manual_seed(9173)
        origin = copy.deepcopy(Receiver().state_dict())
        torch.manual_seed(9173)
        self.assertEqual(state_hash(origin), state_hash(Receiver().state_dict()))
        mapping = [(5*v+3) % 16 for v in range(16)]
        transformed = paired_state(origin, mapping, 'onehot')
        for v in range(16):
            torch.testing.assert_close(transformed['net.0.weight'][:, mapping[v]], origin['net.0.weight'][:, v], rtol=0, atol=0)
        self.assertEqual(state_hash(paired_state(origin, mapping, 'bit')), state_hash(origin))
        a, b = Receiver(), Receiver()
        a.load_state_dict(origin); b.load_state_dict(transformed)
        with torch.no_grad():
            za = a(torch.from_numpy(features(table(), list(range(16)), 'onehot')))
            zb = b(torch.from_numpy(features(table(), mapping, 'onehot')))
        torch.testing.assert_close(za, zb, atol=1e-6, rtol=0)

    def test_training_reset_and_short_control(self):
        torch.manual_seed(9183)
        origin = copy.deepcopy(Receiver().state_dict())
        origin_hash = state_hash(origin)
        data = table(); q = torch.from_numpy(data['q']); y = torch.from_numpy(data['y'])
        x = torch.from_numpy(features(data, list(range(16)), 'onehot'))
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)/'weights.pt'
            first, z1 = train(origin, x, q, y, 3, dest)
            repeated, z2 = train(origin, x, q, y, 3)
            self.assertEqual(first, repeated)
            np.testing.assert_array_equal(z1, z2)
            self.assertEqual(state_hash(origin), origin_hash)
            ckpt = torch.load(dest, weights_only=True)
            self.assertEqual(ckpt['initial_optimizer']['state'], {})
            self.assertTrue(all(int(s['step']) == 3 for s in ckpt['final_optimizer']['state'].values()))
        mapping = [(7*v+1) % 16 for v in range(16)]
        _, paired_logits = train(paired_state(origin, mapping, 'onehot'), torch.from_numpy(features(data, mapping, 'onehot')), q, y, 3)
        np.testing.assert_allclose(z1, paired_logits, atol=1e-5, rtol=0)

    def test_control_failure_blocks_good_bit_results(self):
        branches, endpoints = {}, {}
        for rep in ('bit', 'onehot'):
            for k in range(4):
                branches[f'{rep}_E{k}'] = {'curve': [{'BCE': .5 + (.1 if rep == 'bit' and k else 0),
                    'accuracy': .8 - (.1 if rep == 'bit' and k else 0)} for _ in range(101)]}
                endpoints[f'{rep}_E{k}'] = np.zeros((2, 2, 4))
        result = compare_seed(branches, endpoints)
        reports = [{'seed': seed, 'result': copy.deepcopy(result)} for seed in range(60, 90)]
        self.assertTrue(summarize(reports)['primary_passed'])
        branches['onehot_E2']['curve'][7]['accuracy'] -= .001
        bad = compare_seed(branches, endpoints)
        reports[0]['result'] = bad
        summary = summarize(reports)
        self.assertTrue(summary['bit_numeric_gate_passed'])
        self.assertFalse(summary['primary_passed'])
        self.assertFalse(summary['mechanism_claim_allowed'])

    def test_every_mapping_and_both_gates_required(self):
        result = {'bit_comparisons': {f'E{k}': {'joint_pass': True} for k in (1, 2, 3)}, 'control_passed': True}
        reports = [{'seed': seed, 'result': copy.deepcopy(result)} for seed in range(60, 90)]
        for r in reports[:7]:
            r['result']['bit_comparisons']['E3']['joint_pass'] = False
        self.assertEqual(summarize(reports)['bit_joint_pass_counts']['E3'], 23)
        self.assertFalse(summarize(reports)['primary_passed'])
        branches = {f'{rep}_E{k}': {'curve': [{'BCE': .5 + (.1 if k else 0), 'accuracy': .8} for _ in range(101)]}
                    for rep in ('bit', 'onehot') for k in range(4)}
        endpoints = {name: np.zeros((2, 2, 4)) for name in branches}
        self.assertFalse(compare_seed(branches, endpoints)['bit_comparisons']['E1']['joint_pass'])


if __name__ == '__main__':
    unittest.main()
