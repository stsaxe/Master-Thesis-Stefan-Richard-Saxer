import unittest

import torch

from evaluation_framework.metrics.closed_set.recall import Recall


class TestRecall(unittest.TestCase):
    def test_recall_micro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Recall()

        recall = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(recall, float)
        self.assertAlmostEqual(3 / 5, recall, places=5)

        name = metric.get_name()

        self.assertEqual('Recall (micro)', name)

    def test_recall_macro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],

                  [3, 2, 1],
                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Recall(average='macro')

        recall = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(recall, float)
        self.assertAlmostEqual((1 / 1 + 1 / 3 + 1 / 2) / 3, recall, places=5)

        name = metric.get_name()

        self.assertEqual('Recall (macro)', name)

    def test_recall_weighted(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],

                  [3, 2, 1],
                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Recall(average='weighted')

        recall = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(recall, float)
        self.assertAlmostEqual((1 * 1 / 1 + 3 * 1 / 3 + 2 * 1 / 2) / 6, recall, places=5)

        name = metric.get_name()

        self.assertEqual('Recall (weighted)', name)
    def test_assertions(self):
        with self.assertRaises(AssertionError):
            Recall(average='test')

        with self.assertRaises(AssertionError):
            Recall(average=123)

