import unittest

import torch

from evaluation_framework.metrics.open_set.precision_at_recall import PrecisionAtRecall


class TestPrecisionAtRecall(unittest.TestCase):
    def test_precision_at_recall_garbage(self):
        logits = [[90, 10, 0],
                  [40, 60, 0],
                  [80, 20, 0],

                  [35, 20, 45],
                  [90, 10, 0],
                  [20, 30, 90],
                  [30, 40, 30],

                  [70, 30, 0],
                  [0, 20, 80],
                  [30, 30, 40],
                  [20, 80, 0]
                  ]

        targets = [0, 0, 0,
                   1, 1, 1, 1,
                   2, 2, 2, 2]

        max_vals = torch.tensor(logits)[:, :2].max(dim=1)[0]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionAtRecall(unknown_label=2, recall=0.8)

        result = metric.compute(logits=logits, targets=targets)

        y_true = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        y_score = max_vals

        self.assertAlmostEqual(0.75, result, places=5)

        name = metric.get_name()
        self.assertEqual(name, 'Precision at 80.0% Recall')

        metric = PrecisionAtRecall(unknown_label=2, recall=0.7)

        result = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(0.75, result, places=5)
