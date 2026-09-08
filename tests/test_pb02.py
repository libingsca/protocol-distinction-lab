import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pb01.run import CONFIG as PB01_CONFIG
from pb02.run import CONFIG, LinearReceiver, run, summarize


class PB02Tests(unittest.TestCase):
    def rows(self):
        return [dict(seed=s, c0_ce=-.1, c2_ce=.08, accuracy_gain_t4=.2,
                     random_c2_ce=-.3, noninfo_c2_ce=-.2, spare_barrier_reduction=.99,
                     pretrain_error=0., loss_precedes_decision=True) for s in range(30, 60)]

    def test_frozen_settings_and_seed_isolation(self):
        self.assertEqual(CONFIG['seeds'], list(range(30, 60)))
        self.assertFalse(set(CONFIG['seeds']) & set(PB01_CONFIG['seeds']))
        for key in ('alpha', 'pretrain_steps', 'pretrain_learning_rate', 'adapt_learning_rate', 'task', 'source'):
            self.assertEqual(CONFIG[key], PB01_CONFIG[key])
        self.assertEqual(CONFIG['target'], [0, 1, 2, 0, 1])
        self.assertEqual((CONFIG['T_loss'], CONFIG['T_decision']), (2, 4))

    def test_every_gate_is_required(self):
        rows = self.rows()
        self.assertTrue(summarize(rows)['passed'])
        for key, value in [('c0_ce', -.01), ('c2_ce', .01), ('accuracy_gain_t4', 0.),
                           ('spare_barrier_reduction', .49), ('random_c2_ce', .09), ('noninfo_c2_ce', .09)]:
            changed = copy.deepcopy(rows)
            for row in changed:
                row[key] = value
            self.assertFalse(summarize(changed)['passed'], key)
        for key, value in [('c0_ce', .1), ('accuracy_gain_t4', 0.)]:
            changed = copy.deepcopy(rows)
            for row in changed[:7]:
                row[key] = value
            self.assertFalse(summarize(changed)['passed'], key)

    def test_rejects_missing_duplicate_and_old_seeds(self):
        for seeds in [list(range(29, 59)), list(range(30, 59)), [30] * 30]:
            rows = self.rows()[:len(seeds)]
            for row, seed in zip(rows, seeds):
                row['seed'] = seed
            with self.assertRaises(ValueError):
                summarize(rows)

    def test_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):
                run(path)
            self.assertEqual((path / 'sentinel').read_text(), 'keep')

    def test_linear_gradient_matches_finite_difference(self):
        model = LinearReceiver(3, 31)
        messages, labels = [0, 1, 1, 2, 1], [0, 1, 2, 0, 1]
        epsilon = 1e-6
        numerical = np.zeros_like(model.w)
        for index in np.ndindex(model.w.shape):
            plus, minus = model.clone(), model.clone()
            plus.w[index] += epsilon
            minus.w[index] -= epsilon
            numerical[index] = (plus.loss_accuracy(messages, labels)[0] - minus.loss_accuracy(messages, labels)[0]) / (2 * epsilon)
        updated = model.clone()
        updated.hard_step(messages, labels, 1.)
        np.testing.assert_allclose(model.w - updated.w, numerical, atol=1e-8)
        self.assertLess(updated.loss_accuracy(messages, labels)[0], model.loss_accuracy(messages, labels)[0])


if __name__ == '__main__':
    unittest.main()
