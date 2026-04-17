import unittest

import torch
from sklearn.metrics import precision_recall_curve

from evaluation_framework.metrics.open_set.pr_curve_open_set import PRCurveOpenSet


class TestPRCurveOpenSet(unittest.TestCase):
    def test_pr_curve_thresholding(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [35, 65],
                  [90, 10],
                  [20, 80],
                  [30, 70],

                  [70, 30],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0,
                   1, 1, 1, 1,
                   2, 2, 2]

        max_vals = torch.tensor(logits).max(dim=1)[0]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PRCurveOpenSet(unknown_label=2)

        result = metric.compute(logits=logits, targets=targets)

        precision = result[:, 0].float()
        recall = result[:, 1].float()

        y_true = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        y_score = max_vals

        p, r, _ = precision_recall_curve(y_true, y_score)

        precision_solution = torch.tensor(p).float()
        recall_solution = torch.tensor(r).float()

        self.assertTrue(precision.allclose(precision_solution))
        self.assertTrue(recall.allclose(recall_solution))

        name = metric.get_name()

        self.assertEqual('Precision Recall Curve Open Set', name)

    def test_pr_curve_garbage(self):
        logits = [[90, 10, 0],
                  [40, 60, 0],
                  [80, 20, 0],

                  [35, 20, 45],
                  [90, 10, 0],
                  [20, 30, 50],
                  [10, 70, 20],

                  [70, 30, 0],
                  [0, 20, 80],
                  [30, 30, 40]
                  ]

        targets = [0, 0, 0,
                   1, 1, 1, 1,
                   2, 2, 2]

        max_vals = torch.tensor(logits)[:, :2].max(dim=1)[0]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PRCurveOpenSet(unknown_label=2)

        result = metric.compute(logits=logits, targets=targets)

        precision = result[:, 0].float()
        recall = result[:, 1].float()

        y_true = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        y_score = max_vals

        p, r, _ = precision_recall_curve(y_true, y_score)

        precision_solution = torch.tensor(p).float()
        recall_solution = torch.tensor(r).float()

        self.assertTrue(precision.allclose(precision_solution))
        self.assertTrue(recall.allclose(recall_solution))


