import unittest

import torch

from evaluation_framework.metrics.closed_set.f1_score import F1Score


class TestF1Score(unittest.TestCase):
    def test_f1_micro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],

                  [3, 2, 1],

                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = F1Score()

        recall = 3 / 6
        precision = 3 / 6

        f1 = 2 * (recall * precision) / (recall + precision)

        f1_score = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(f1_score, float)
        self.assertAlmostEqual(f1_score, f1, places=5)

        name = metric.get_name()

        self.assertEqual('F1 Score (micro)', name)

    def test_f1_macro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],

                  [3, 2, 1],

                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = F1Score(average='macro')

        rc_0 = 1 / 1
        pc_0 = 1 / 3

        rc_1 = 1 / 3
        pc_1 = 1 / 1

        rc_2 = 1 / 2
        pc_2 = 1 / 2

        f1_0 = 2 * (rc_0 * pc_0) / (rc_0 + pc_0)
        f1_1 = 2 * (rc_1 * pc_1) / (rc_1 + pc_1)
        f1_2 = 2 * (rc_2 * pc_2) / (rc_2 + pc_2)

        f1 = (f1_0 + f1_1 + f1_2) / 3

        f1_score = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(f1_score, float)
        self.assertAlmostEqual(f1_score, f1, places=5)

        name = metric.get_name()

        self.assertEqual('F1 Score (macro)', name)

    def test_f1_weighted(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],

                  [3, 2, 1],

                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = F1Score(average='weighted')

        rc_0 = 1 / 1
        pc_0 = 1 / 3

        rc_1 = 1 / 3
        pc_1 = 1 / 1

        rc_2 = 1 / 2
        pc_2 = 1 / 2

        f1_0 = 2 * (rc_0 * pc_0) / (rc_0 + pc_0)
        f1_1 = 2 * (rc_1 * pc_1) / (rc_1 + pc_1)
        f1_2 = 2 * (rc_2 * pc_2) / (rc_2 + pc_2)

        f1 = (f1_0 * 1+ f1_1 * 3 + f1_2 * 2) / 6

        f1_score = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(f1_score, float)
        self.assertAlmostEqual(f1_score, f1, places=5)

        name = metric.get_name()

        self.assertEqual('F1 Score (weighted)', name)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            F1Score(average='test')

        with self.assertRaises(AssertionError):
            F1Score(average=123)
