import unittest

import torch

from evaluation_framework.metrics.closed_set.accuracy import Accuracy


class TestAccuracy(unittest.TestCase):
    def test_accuracy_micro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [100, 0, 0],
                  [0, 100, 0]]

        targets = [2, 1, 0, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Accuracy(average='micro')

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(3 / 4, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy', name)

    def test_accuracy_empty_class(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [100, 0, 0],
                  [0, 100, 0]]

        targets = [1, 1, 0, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Accuracy(average='weighted')

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(2 / 4, accuracy, places=5)

        metric = Accuracy(average='macro')

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((1 / 3 + 1/1 + 0)/3, accuracy, places=5)


    def test_accuracy_macro(self):
        logits = [[5, 2, 3],
                  [3, 2, 1],
                  [100, 0, 0],
                  [0, 100, 0],
                  [100, 0, 0]
                  ]

        targets = [2, 1, 0, 1, 0]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Accuracy(average='macro')

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((0 + 0.5 + 1) / 3, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy (macro)', name)

    def test_accuracy_weighted(self):
        logits = [[5, 2, 3],
                  [3, 2, 1],
                  [100, 0, 0],
                  [0, 100, 0],
                  [100, 0, 0]
                  ]

        targets = [2, 1, 0, 1, 0]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Accuracy(average='weighted')

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((1 * 0 + 2 * 0.5 + 2 * 1) / 5, accuracy, places=5)


        name = metric.get_name()

        self.assertEqual('Accuracy (weighted)', name)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            Accuracy(average='test')
        with self.assertRaises(AssertionError):
            Accuracy(average=123)
