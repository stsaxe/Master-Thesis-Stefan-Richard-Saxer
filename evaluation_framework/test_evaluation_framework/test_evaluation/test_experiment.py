import unittest

import torch

from evaluation_framework.collections.static_metric_collection import StaticMetricCollection
from evaluation_framework.experiments.experiment import Experiment
from evaluation_framework.metrics.closed_set.accuracy import Accuracy
from evaluation_framework.metrics.closed_set.accuracy_topk import TopKAccuracy


class TestExperiment(unittest.TestCase):

    def setUp(self):
        logits = [[1, 2, 3],
                  [3, 2, 1],
                  [1, 2, 3],
                  [0, 1, 0]]

        targets = [2, 1, 0, 1]

        logits_2 = [[1, 2, 3],
                    [3, 2, 1]]

        targets_2 = [2, 0]

        self.logits = torch.Tensor(logits)
        self.targets = torch.Tensor(targets)

        self.logits_2 = torch.Tensor(logits_2)
        self.targets_2 = torch.Tensor(targets_2)

    def test_simple_experiment(self):
        exp = Experiment("TestExperiment")
        exp.add_metrics([Accuracy(), TopKAccuracy(topK=2)])
        exp.add_metrics({'Metric': TopKAccuracy(topK=10)})

        exp.add_data(logits=self.logits, targets=self.targets)
        score = exp.score()

        self.assertAlmostEqual(score['Accuracy'], 1 / 2, places=5)
        self.assertAlmostEqual(score['Top-2 Accuracy'], 3 / 4, places=5)
        self.assertAlmostEqual(score['Metric'], 1, places=5)

        exp.add_data(logits=self.logits_2, targets=self.targets_2)
        score = exp.score()

        self.assertAlmostEqual(score['Top-2 Accuracy'], 5 / 6, places=5)
        self.assertAlmostEqual(score['Accuracy'], 2 / 3, places=5)
        self.assertAlmostEqual(score['Metric'], 1, places=5)

        score = exp['Accuracy'].score()
        self.assertAlmostEqual(score['Accuracy'], 2 / 3, places=5)
        self.assertAlmostEqual(len(score), 1)

        score = exp[['Accuracy', 'Top-2 Accuracy']].score()

        self.assertAlmostEqual(score['Top-2 Accuracy'], 5 / 6, places=5)
        self.assertAlmostEqual(score['Accuracy'], 2 / 3, places=5)

        exp = exp.reset()

        with self.assertRaises(AssertionError) as e:
            acc = exp.score()['Accuracy']

        exp.add_data(logits=self.logits, targets=self.targets)
        score = exp.score()

        self.assertAlmostEqual(score['Accuracy'], 1 / 2, places=5)
        self.assertAlmostEqual(score['Top-2 Accuracy'], 3 / 4, places=5)
        self.assertAlmostEqual(score['Metric'], 1, places=5)

    def test_configure(self):
        config = StaticMetricCollection({'Acc': Accuracy(),
                                         'Top-2': TopKAccuracy(topK=2)})

        exp = Experiment("TestExperiment")

        exp.configure(config)

        exp.add_data(logits=self.logits, targets=self.targets)

        score = exp.score()

        self.assertAlmostEqual(score['Acc'], 1 / 2, places=5)
        self.assertAlmostEqual(score['Top-2'], 3 / 4, places=5)

        exp.add_data(logits=self.logits_2, targets=self.targets_2)

        score = exp.score()
        self.assertAlmostEqual(score['Top-2'], 5 / 6, places=5)
        self.assertAlmostEqual(score['Acc'], 2 / 3, places=5)

    def test_slicing_assertions(self):
        config = StaticMetricCollection({'Acc': Accuracy(),
                                         'Top-2': TopKAccuracy(topK=2)})

        exp = Experiment("TestExperiment")

        exp.configure(config)
        exp.add_data(logits=self.logits, targets=self.targets)

        with self.assertRaises(AssertionError) as e:
            sliced_exp = exp[123]

        with self.assertRaises(AssertionError) as e:
            sliced_exp = exp[['Accuracy', 123]]

        with self.assertRaises(AssertionError) as e:
            sliced_exp = exp[{'Accuracy', 'Top-2'}]

    def test_force_compute(self):
        config = StaticMetricCollection({'Acc': Accuracy(),
                                         'Top-2': TopKAccuracy(topK=2)})

        exp = Experiment("TestExperiment")
        exp.configure(config)

        metrics = exp.get_metrics()

        acc = metrics[0]
