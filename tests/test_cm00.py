import unittest

from cm00.core import (
    Space, align_partition, canonical_sender, complement, decoder_accuracy, exact_metrics,
    maturation_curve, minimax_barrier_cross_entropy, optimal_cross_entropy,
    partition_distance,
)
from cm00.independent_checker import brute_force_best_accuracy


class CM00Tests(unittest.TestCase):
    def setUp(self):
        self.space = Space("unit", 4, 1)

    def test_preregistered_counterexample(self):
        task = (0, 0, 0, 1)
        sender = (0, 0, 1, 0)
        rewritten = (0, 0, 0, 1)
        base = exact_metrics(task, sender, self.space)
        target = exact_metrics(task, rewritten, self.space)
        self.assertFalse(base.has_tie)
        self.assertFalse(target.has_tie)
        self.assertEqual(base.accuracy, 0.75)
        self.assertEqual(decoder_accuracy(task, rewritten, base.decoder, self.space), 0.75)
        self.assertEqual(target.accuracy, 1.0)
        self.assertGreater(target.mutual_information, base.mutual_information)

    def test_independent_receiver_agrees(self):
        task = (0, 1, 1, 0)
        sender = (0, 0, 1, 1)
        direct = exact_metrics(task, sender, self.space)
        brute, _, _ = brute_force_best_accuracy(task, sender, self.space)
        self.assertEqual(direct.accuracy, brute)

    def test_message_rename_invariance(self):
        task = (0, 0, 0, 1)
        sender = (0, 0, 1, 0)
        self.assertEqual(exact_metrics(task, sender, self.space).accuracy,
                         exact_metrics(task, complement(sender), self.space).accuracy)
        self.assertEqual(canonical_sender(sender, 2), canonical_sender(complement(sender), 2))
        self.assertEqual(partition_distance(sender, complement(sender)), 0)

    def test_dynamic_curve_matures(self):
        curve = maturation_curve(
            (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 0, 1), self.space,
            alpha=0.5, learning_rate=1.0, steps=(0, 1, 2, 4, 8, 16, 32),
        )
        # Accuracy is neutral, while the frozen receiver assigns a worse
        # probability to the rewritten positive state: CE credit is negative.
        self.assertLess(curve[0], 0)
        self.assertGreater(curve[32], 0)

    def test_target_partition_uses_compatible_message_names(self):
        task = (0, 0, 1, 1)
        sender = (0, 0, 0, 1)
        self.assertEqual(
            align_partition(task, sender, (0, 0, 1, 1), self.space),
            (0, 0, 1, 1),
        )

    def test_three_message_relabeling(self):
        sender = (0, 1, 2, 0)
        relabeled = (2, 0, 1, 2)
        self.assertEqual(canonical_sender(sender, 3), canonical_sender(relabeled, 3))
        self.assertEqual(partition_distance(sender, relabeled, 3), 0)

    def test_three_label_receiver(self):
        space = Space("three_labels", 4, 1, 2, 3)
        task = (0, 0, 1, 1)
        sender = (0, 0, 1, 1)
        self.assertEqual(exact_metrics(task, sender, space).accuracy, 1.0)
        self.assertAlmostEqual(optimal_cross_entropy(task, sender, space), 0.0)

    def test_cross_entropy_barrier_zero_for_known_space(self):
        barrier = minimax_barrier_cross_entropy(
            (0, 0, 0, 1), (0, 0, 1, 0), self.space, max_partition_distance=1,
        )
        self.assertIsNotNone(barrier)
        self.assertLessEqual(barrier, 1e-12)


if __name__ == "__main__":
    unittest.main()
