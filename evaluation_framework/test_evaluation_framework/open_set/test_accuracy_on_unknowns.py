import unittest

import torch

from evaluation_framework.metrics.open_set.accuracy_on_unknowns import AccuracyOnUnknowns


class TestAccuracyOnUnknowns(unittest.TestCase):
    def test_name(self):
        metric = AccuracyOnUnknowns(unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'Accuracy on Unknowns')

    def test_micro_garbage(self):
        logits = [[70, 20, 10, 0],
                  [10, 60, 30, 0],
                  [20, 0, 0, 80],

                  [10, 50, 0, 40],

                  [0, 10, 40, 50],

                  [0, 10, 50, 40],
                  [0, 0, 20, 80],
                  [60, 30, 0, 10]
                  ]

        targets = [0, 0, 0, 1, 2, 3, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        solution = 1 / 3

        acc = AccuracyOnUnknowns(unknown_label=3).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(solution, acc, places=4)

        targets = torch.zeros(8)

        acc = AccuracyOnUnknowns(unknown_label=3).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(0.0, acc, places=4)

    def test_micro_thresholding(self):
        logits = [[70, 20, 10],
                  [10, 60, 30],
                  [20, 0, 80],

                  [10, 50, 40],

                  [10, 30, 60],

                  [30, 40, 30],
                  [50, 30, 20],
                  [20, 70+0.01, 10],
                  [20, 20, 60]
                  ]

        targets = [0, 0, 0, 1, 2, 3, 3, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        acc = AccuracyOnUnknowns(unknown_label=3, precision=1).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(acc[0], 0, places=4)
        self.assertAlmostEqual(float(acc[5]), 1/4, places=4)
        self.assertAlmostEqual(float(acc[6]),2/4 , places=4)
        self.assertAlmostEqual(float(acc[7]), 3 / 4, places=4)
        self.assertAlmostEqual(float(acc[8]), 4 / 4, places=4)
        self.assertAlmostEqual(acc[10], 1, places=4)

        targets = torch.zeros(9)

        acc = AccuracyOnUnknowns(unknown_label=3, precision=1).compute(logits=logits, targets=targets)

        solution = torch.zeros(11)

        self.assertTrue(solution.allclose(acc))
