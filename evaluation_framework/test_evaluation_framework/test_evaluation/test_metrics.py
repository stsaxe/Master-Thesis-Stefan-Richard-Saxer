import unittest

import numpy as np
import torch

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class TestMetrics(unittest.TestCase):
    def test_abstract_classification_metric(self):
        class abstract_metric(AbstractMetric):
            def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
                return 0

            def get_name(self) -> str:
                return ""

        logits = [[0, 1e6, 0, 0],
                  [np.log(2), np.log(3), np.log(4), np.log(1)],
                  [0, 0, 0, 0]]

        targets = [1, 2, 3]

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        logits_new, probas, targets_new, = abstract_metric().extract_logits_and_targets(logits=logits, targets=targets)

        real_probas = [[0, 1, 0, 0],
                       [0.2, 0.3, 0.4, 0.1],
                       [0.25, 0.25, 0.25, 0.25]]

        real_probas = torch.Tensor(real_probas)

        # test output of softmax function
        self.assertTrue(torch.allclose(probas, real_probas, atol=1e-6))

        # test assertions
        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets()

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(l=logits, targets=torch.zeros(2))

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=logits, t=targets)

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=torch.zeros((2, 3)), targets=np.zeros(2))

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=np.zeros((2, 3)), targets=torch.zeros(2))

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=torch.zeros((2, 3, 4)), targets=torch.zeros(2))

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=torch.zeros((2, 3)), targets=torch.zeros(3))

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=torch.zeros((2, 3)), targets=torch.zeros(3))

        with self.assertRaises(AssertionError) as e:
            abstract_metric().extract_logits_and_targets(logits=torch.zeros((2, 1)), targets=torch.zeros(2))

    def test_abstract_open_set_metric(self):
        class abstract_metric(AbstractMetric):
            def __init__(self, unknown_label, add_unknowns):
                super().__init__(add_unknowns=add_unknowns, unknown_label=unknown_label)
            def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
                logits, probas, targets = self.extract_logits_and_targets(**kwargs)
                return probas

            def get_name(self) -> str:
                return ""

        with self.assertRaises(AssertionError) as e:
            abstract_metric(unknown_label=None, add_unknowns='True')

        with self.assertRaises(AssertionError) as e:
            abstract_metric(unknown_label=-1, add_unknowns=True)

        with self.assertRaises(AssertionError) as e:
            abstract_metric(unknown_label=None, add_unknowns=True)

        logits = [[60, 10, 20, 10],
                  [10, 10, 60, 20],
                  [60, 10, 10, 20],
                  [0, 40, 50, 10]
                  ]

        targets = [0, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = abstract_metric(unknown_label=2, add_unknowns=True)
        score = metric.compute(logits=logits, targets=targets)

        solution = [[60, 10, 30],
                    [10, 10, 80],
                    [60, 10, 30],
                    [0, 40, 60]]

        solution = torch.tensor(solution).float() / 100

        self.assertTrue(solution.allclose(score, atol=1e-5))

        logits = [[60, 10, 20, 10],
                  [10, 10, 60, 20],
                  [60, 10, 10, 20],
                  [0, 40, 50, 10]
                  ]

        targets = [0, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = abstract_metric(unknown_label=4, add_unknowns=True)
        score = metric.compute(logits=logits, targets=targets)

        solution = [[60, 10, 20, 10],
                    [10, 10, 60, 20],
                    [60, 10, 10, 20],
                    [0, 40, 50, 10]
                    ]

        solution = torch.tensor(solution).float() / 100

        self.assertTrue(solution.allclose(score, atol=1e-5))

        metric = abstract_metric(unknown_label=3, add_unknowns=True)
        score = metric.compute(logits=logits, targets=targets)
        self.assertTrue(solution.allclose(score, atol=1e-5))
