import unittest

import torch

from evaluation_framework.metrics.open_set.recall_at_precision import RecallAtPrecision


class TestRecallAtPrecision(unittest.TestCase):
    def test_recall_at_precision_garbage(self):
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

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = RecallAtPrecision(unknown_label=2, precision=0.6)

        result = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(result, 1.0)
        name = metric.get_name()
        self.assertEqual(name, 'Recall at 60.0% Precision')

        metric = RecallAtPrecision(unknown_label=2, precision=0.7)

        result = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(result, 0.85714, places=3)
