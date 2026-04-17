import unittest

import torch

from evaluation_framework.metrics.closed_set.accuracy import Accuracy
from evaluation_framework.metrics.container.metric_container import MetricContainer


class TestMetricContainer(unittest.TestCase):
    def test_construction(self):
        with self.assertRaises(AssertionError):
            container = MetricContainer(Accuracy(), 123)

        with self.assertRaises(AssertionError):
            container = MetricContainer('ABC', 123)

        with self.assertRaises(AssertionError):
            container = MetricContainer(Accuracy(), 'ABC', 123)

    def test_getters(self):
        metric = Accuracy()
        container = MetricContainer(metric, force_compute=True)

        self.assertEqual(container.get_name(), 'Accuracy')
        self.assertIsInstance(container.get_metric(), Accuracy)
        self.assertTrue(container.is_force_compute())

        get_metric = container.get_metric()
        self.assertFalse(get_metric is metric)

        container = MetricContainer(Accuracy(), "ABC")
        self.assertEqual(container.get_name(), 'ABC')

    def test_metric(self):
        container = MetricContainer(Accuracy(), 'Accuracy')

        logits = torch.tensor([[1, 2, 3],
                               [2, 3, 1],
                               [3, 2, 1]]).float()

        targets = torch.tensor([2, 0, 0])

        accuracy = container.compute(logits=logits, targets=targets)
        self.assertAlmostEqual(accuracy, 2 / 3, places=5)

        logits = torch.tensor([[1, 2, 3],
                               [1, 2, 3]]).float()

        targets = torch.tensor([2, 1])

        self.assertIsInstance(container.reset_score(), MetricContainer)

        accuracy = container.compute(logits=logits, targets=targets)
        self.assertAlmostEqual(accuracy, 0.5, places=5)

    def test_force_compute(self):
        container = MetricContainer(Accuracy(), "Accuracy", )

        self.assertFalse(container.is_force_compute())

        logits = torch.tensor([[1, 2, 3],
                               [2, 3, 1],
                               [3, 2, 1]]).float()

        targets = torch.tensor([2, 0, 0])

        accuracy = container.compute(logits=logits, targets=targets)
        self.assertAlmostEqual(accuracy, 2 / 3, places=5)

        logits = torch.tensor([[1, 2, 3],
                               [1, 2, 3]]).float()

        targets = torch.tensor([2, 1])

        accuracy = container.compute(logits=logits, targets=targets)
        self.assertAlmostEqual(accuracy, 2 / 3, places=5)

        container = MetricContainer(Accuracy(), "Accuracy")
        self.assertFalse(container.is_force_compute())

        container = MetricContainer(Accuracy(), "Accuracy", True)
        self.assertTrue(container.is_force_compute())

