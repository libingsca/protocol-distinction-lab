import unittest

from pb01.neural import pretrain_receiver


class PB01Tests(unittest.TestCase):
    def test_receiver_pretraining_matches_smoothed_conditionals(self):
        model, error = pretrain_receiver(
            (0, 1, 1, 2, 1), (0, 1, 2, 0, 1),
            n_messages=3, n_labels=3, hidden=8, seed=0,
            alpha=0.5, steps=3000, learning_rate=0.1,
        )
        self.assertLess(error, 1e-3)
        loss, accuracy = model.loss_accuracy((0, 1, 1, 2, 1), (0, 1, 2, 0, 1))
        self.assertGreater(loss, 0)
        self.assertEqual(accuracy, 0.8)
        one_hot = __import__("numpy").eye(3)[[0, 1, 1, 2, 1]]
        soft_loss, soft_accuracy = model.loss_accuracy_soft(one_hot, (0, 1, 2, 0, 1))
        self.assertAlmostEqual(loss, soft_loss)
        self.assertEqual(accuracy, soft_accuracy)


if __name__ == "__main__":
    unittest.main()
