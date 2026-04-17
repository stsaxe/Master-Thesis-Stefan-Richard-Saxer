import unittest

import torch

from evaluation_framework.metrics.open_set.accuracy_on_knowns import AccuracyOnKnowns


class TestAccuracyOnKnowns(unittest.TestCase):
    def test_name(self):
        metric = AccuracyOnKnowns(unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'Accuracy on Knowns (micro)')

        metric = AccuracyOnKnowns(average='macro', unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'Accuracy on Knowns (macro)')

        metric = AccuracyOnKnowns(average='weighted', unknown_label=1)
        name = metric.get_name()
        self.assertEqual(name, 'Accuracy on Knowns (weighted)')

    def test_micro_garbage(self):
        logits = [[70, 20, 10, 0],
                  [10, 60, 30, 0],
                  [90, 10, 0, 0],
                  [20, 0, 0, 80],

                  [10, 50, 0, 40],

                  [0, 30, 50, 20],
                  [0, 10, 40, 50],

                  [0, 10, 50, 40],
                  [0, 0, 20, 80],

                  ]

        targets = [0, 0, 0, 0, 1, 2, 2, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        solution = (2 + 1 + 1) / 7

        acc = AccuracyOnKnowns(unknown_label=3).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(solution, acc, places=4)

    def test_macro_garbage(self):
        logits = [[70, 20, 10, 0],
                  [10, 60, 30, 0],
                  [90, 10, 0, 0],
                  [20, 0, 0, 80],

                  [10, 50, 0, 40],

                  [0, 30, 50, 20],
                  [0, 10, 40, 50],

                  [0, 10, 50, 40],
                  [0, 0, 20, 80],

                  ]

        targets = [0, 0, 0, 0, 1, 2, 2, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        solution = (2 / 4 + 1 / 1 + 1 / 2) / 3

        acc = AccuracyOnKnowns(unknown_label=3, average='macro').compute(logits=logits, targets=targets)

        self.assertAlmostEqual(solution, acc, places=4)

    def test_weighted_garbage(self):
        logits = [[70, 20, 10, 0],
                  [10, 60, 30, 0],
                  [90, 10, 0, 0],
                  [20, 0, 0, 80],

                  [10, 50, 0, 40],

                  [0, 30, 50, 20],
                  [0, 10, 40, 50],

                  [0, 10, 50, 40],
                  [0, 0, 20, 80],

                  ]

        targets = [0, 0, 0, 0, 1, 2, 2, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        solution = (4 * 2 / 4 + 1 * 1 / 1 + 2 * 1 / 2) / 7

        acc = AccuracyOnKnowns(unknown_label=3, average='weighted').compute(logits=logits, targets=targets)

        self.assertAlmostEqual(solution, acc, places=4)

    def test_micro_thresholding(self):
        logits = [[71, 19, 10],
                  [10, 60, 30],
                  [90, 10, 0],
                  [80, 20, 0],

                  [30, 40, 30],

                  [0, 60, 40],
                  [0, 40, 60],

                  [10, 20, 70],
                  [40, 30, 30],

                  ]

        targets = [0, 0, 0, 0, 1, 2, 2, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        acc = AccuracyOnKnowns(unknown_label=3, precision=1).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(acc[4], (3 + 1 + 1) / 7, places=4)
        self.assertAlmostEqual(acc[5], (3 + 0 + 1) / 7, places=4)
        self.assertAlmostEqual(acc[6], (3 + 0 + 1) / 7, places=4)
        self.assertAlmostEqual(acc[7], (3 + 0 + 0) / 7, places=4)
        self.assertAlmostEqual(acc[10], 0.0, places=4)

    def test_macro_thresholding(self):
        logits = [[71, 19, 10],
                  [10, 60, 30],
                  [90, 10, 0],
                  [80, 20, 0],

                  [30, 40, 30],

                  [0, 60, 40],
                  [0, 40, 60],

                  [10, 20, 70],
                  [40, 30, 30],

                  ]

        targets = [0, 0, 0, 0, 1, 2, 2, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        acc = AccuracyOnKnowns(unknown_label=3, precision=1, average='macro').compute(logits=logits, targets=targets)

        self.assertAlmostEqual(acc[4], (3 / 4 + 1 + 1 / 2) / 3, places=4)
        self.assertAlmostEqual(acc[5], (3 / 4 + 0 + 1 / 2) / 3, places=4)
        self.assertAlmostEqual(acc[6], (3 / 4 + 0 + 1 / 2) / 3, places=4)
        self.assertAlmostEqual(acc[7], (3 / 4 + 0 + 0) / 3, places=4)
        self.assertAlmostEqual(acc[10], 0.0, places=4)

    def test_weighted_thresholding(self):
        logits = [[71, 19, 10],
                  [10, 60, 30],
                  [90, 10, 0],
                  [80, 20, 0],

                  [30, 40, 30],

                  [0, 60, 40],
                  [0, 40, 60],

                  [10, 20, 70],
                  [40, 30, 30],

                  ]

        targets = [0, 0, 0, 0, 1, 2, 2, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        acc = AccuracyOnKnowns(unknown_label=3, precision=1, average='weighted').compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(acc[4]), (3 + 1 + 1) / 7, places=4)
        self.assertAlmostEqual(float(acc[5]), (3 + 0 + 1) / 7, places=4)
        self.assertAlmostEqual(float(acc[6]), (3 + 0 + 1) / 7, places=4)
        self.assertAlmostEqual(float(acc[7]), (3 + 0 + 0) / 7, places=4)
        self.assertAlmostEqual(float(acc[10]), 0.0, places=4)

    def test_empty_garbage(self):
        logits = [[71, 19, 10],
                  [10, 60, 30],
                  [90, 10, 0]
                  ]

        targets = [2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        acc = AccuracyOnKnowns(unknown_label=2, precision=1).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(0.0, acc, places=4)

    def test_empty_thresholding(self):
        logits = [[71, 19, 10],
                  [10, 60, 30],
                  [90, 10, 0]
                  ]

        targets = [3, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        acc = AccuracyOnKnowns(unknown_label=3, precision=1).compute(logits=logits, targets=targets)

        solution = torch.zeros(11)

        self.assertTrue(solution.allclose(acc))
