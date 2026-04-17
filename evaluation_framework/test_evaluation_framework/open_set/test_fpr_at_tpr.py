import unittest

import torch
from sklearn.metrics import roc_curve

from evaluation_framework.metrics.open_set.fpr_at_tpr import FPRAtTPR


class TestFPRAtTPR(unittest.TestCase):
    def test_fpr_at_tpr_garbage(self):
        logits = [[90, 10, 0],
                  [40, 60, 0],
                  [80, 20, 0],

                  [35, 20, 45],
                  [90, 10, 0],
                  [20, 30, 90],
                  [30, 40, 30],

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

        metric = FPRAtTPR(unknown_label=2, tpr=0.8)

        result = metric.compute(logits=logits, targets=targets)

        y_true = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        y_score = max_vals

        fpr, tpr, _ = roc_curve(y_true, y_score)

        solution = 0

        for idx, value in enumerate(tpr):
            if value > 0.8:
                solution = float(fpr[idx])
                break

        self.assertAlmostEqual(solution, result, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'FPR at 80.0% TPR')
