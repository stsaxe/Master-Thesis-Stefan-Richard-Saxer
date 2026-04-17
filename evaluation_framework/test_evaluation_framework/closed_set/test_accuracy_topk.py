import unittest

import torch

from evaluation_framework.metrics.closed_set.accuracy_topk import TopKAccuracy


class TestAccuracyTopK(unittest.TestCase):
    def test_accuracy_micro(self):
        logits = [[5, 2, 1],
                  [3, 2, 1],
                  [0, 1, 10],
                  [0, 10, 1],
                  [100, 0, 0],
                  [2, 0, 1],
                  ]

        targets = [2, 1, 0, 1, 0, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = TopKAccuracy(average='micro', topK=2)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(3 / 6, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Top-2 Accuracy', name)

        metric = TopKAccuracy(average='micro', topK=5)
        accuracy = metric.compute(logits=logits, targets=targets)
        self.assertAlmostEqual(1.0, accuracy, places=5)

    def test_accuracy_macro(self):
        logits = [[5, 2, 1],
                  [3, 2, 1],
                  [0, 1, 10],
                  [0, 10, 1],
                  [100, 0, 0],
                  [2, 0, 1],
                  ]

        targets = [2, 1, 0, 1, 0, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = TopKAccuracy(average='macro', topK=2)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((0 + 2 / 3 + 1 / 2) / 3, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Top-2 Accuracy (macro)', name)

    def test_accuracy_weighted(self):
        logits = [[5, 2, 1],
                  [3, 2, 1],
                  [0, 1, 10],
                  [0, 10, 1],
                  [100, 0, 0],
                  [2, 0, 1],
                  ]

        targets = [2, 1, 0, 1, 0, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = TopKAccuracy(average='weighted', topK=2)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((0 + 3 * 2 / 3 + 2 * 1 / 2) / 6, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Top-2 Accuracy (weighted)', name)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            TopKAccuracy(average='test')

        with self.assertRaises(AssertionError):
            TopKAccuracy(average=123)

        with self.assertRaises(AssertionError):
            TopKAccuracy(topK=0)

        with self.assertRaises(AssertionError):
            TopKAccuracy(topK=2.0)

        with self.assertRaises(AssertionError):
            TopKAccuracy(topK='test')
