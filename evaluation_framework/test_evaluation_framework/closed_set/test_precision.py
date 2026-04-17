import unittest

import torch

from evaluation_framework.metrics.closed_set.precision import Precision


class TestPrecision(unittest.TestCase):
    def test_precision_micro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Precision()

        precision = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(precision, float)
        self.assertAlmostEqual(3 / 5, precision, places=5)

        name = metric.get_name()

        self.assertEqual('Precision (micro)', name)

    def test_precision_macro(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Precision(average='macro')

        precision = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(precision, float)
        self.assertAlmostEqual((1/2 + 1/1 + 1 / 2)/3, precision, places=5)

        name = metric.get_name()

        self.assertEqual('Precision (macro)', name)

    def test_precision_weighted(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [100, 1, 0],
                  [1, 100, 0],
                  [1, 0, 2]]

        targets = [2, 1, 0, 1, 1]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = Precision(average='weighted')

        precision = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(precision, float)
        self.assertAlmostEqual((1 * 1 / 2 + 3 * 1 / 1 + 1 / 2) / 5, precision, places=5)

        name = metric.get_name()

        self.assertEqual('Precision (weighted)', name)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            Precision(average='test')

        with self.assertRaises(AssertionError):
            Precision(average=123)

