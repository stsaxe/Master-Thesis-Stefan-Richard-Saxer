import unittest

import torch

from evaluation_framework.metrics.open_set.accuracy_at_tpr import AccuracyAtTPR


class TestAccuracyAtTPR(unittest.TestCase):

    def test_name(self):
        metric = AccuracyAtTPR(unknown_label=1, tpr = 0.9)
        name = metric.get_name()
        self.assertEqual(name, 'Accuracy at 90.0% TPR')

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            AccuracyAtTPR(unknown_label=1, tpr='micro')

        with self.assertRaises(AssertionError):
            AccuracyAtTPR(unknown_label=1, tpr=1)

        with self.assertRaises(AssertionError):
            AccuracyAtTPR(unknown_label=1, tpr=3.0)

        with self.assertRaises(AssertionError):
            AccuracyAtTPR(unknown_label=-1, tpr=0.5)

    def test_accuracy_at_tpr_thresholding(self):
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

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.7, precision=2).compute(logits=logits, targets=targets)

        # threshold 0.7 > tpr = 5 / 7 > 70%
        # acc = 5 + 1 / 10 = 6

        self.assertAlmostEqual(accuracy, 0.6, places=4)


    def test_accuracy_at_tpr_thresholding_variable_precision(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [30, 70],
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

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.8, precision=1).compute(logits=logits, targets=targets)

        # threshold 0.7 > tpr = 6 / 7 > 80%
        # acc = 6 + 1 / 10 = 6

        self.assertAlmostEqual(accuracy, 0.7, places=4)

    def test_accuracy_at_tpr_thresholding_v2(self):
        logits = [[70, 30],
                  [40, 60],
                  [50, 50],

                  [35, 65],
                  [55, 45],
                  [30, 70],
                  [75, 25],

                  [55, 45],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0,
                   1, 1, 1, 1,
                   2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.4, precision=2).compute(logits=logits, targets=targets)

        # threshold 0.7 > tpr = 5 / 7 > 70%
        # acc = (3 + 2) / 10 = 6

        self.assertAlmostEqual(accuracy, 0.5, places=4)

    def test_accuracy_at_tpr_thresholding_no_positives(self):
        logits = [
            [55, 45],
            [60, 40],
            [90, 10]
        ]

        targets = [2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.4, precision=2).compute(logits=logits, targets=targets)

        self.assertAlmostEqual(accuracy, 0.0, places=4)

    def test_accuracy_at_tpr_thresholding_no_negatives(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [35, 65],
                  [90, 10],
                  [20, 80],
                  [30, 70],
                  ]

        targets = [0, 0, 0,
                   1, 1, 1, 1
                   ]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.5, precision=2).compute(logits=logits, targets=targets)

        # tresh 0.8 -> TPR = 4 / 7  > 0.5

        self.assertAlmostEqual(accuracy, 4 / 7, places=4)

    def test_accuracy_at_tpr_garbage(self):
        logits = [[80, 20, 0],
                  [20, 70, 10],
                  [20, 10, 70],

                  [90, 10, 0],
                  [30, 40, 30],
                  [10, 80, 10],

                  [70, 20, 10],
                  [30, 30, 40]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=1, tpr=0.5, precision=2).compute(logits=logits, targets=targets)

        # thresh 0.8 -> 3 / 5 TPR
        # (3 + 2) / 8 = acc

        self.assertAlmostEqual(accuracy, 5 / 8, places=4)

    def test_accuracy_at_tpr_garbage_v2(self):
        logits = [[80, 20, 0],
                  [20, 70, 10],
                  [20, 10, 70],

                  [90, 10, 0],
                  [30, 40, 30],
                  [10, 80, 10],

                  [70, 20, 10],
                  [30, 30, 40]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.5, precision=2).compute(logits=logits, targets=targets)

        # thresh 0.9 -> 4 / 6 TPR
        # (4 + 1) / 8 = acc

        self.assertAlmostEqual(accuracy, 5 / 8, places=4)

    def test_accuracy_at_tpr_garbage_no_negatives(self):
        logits = [[80, 20, 0],
                  [20, 70, 10],
                  [20, 10, 70],

                  [90, 10, 0],
                  [30, 40, 30],
                  [10, 80, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=2, tpr=0.6, precision=2).compute(logits=logits, targets=targets)

        # thresh 0.9 -> 4 / 6 TPR

        self.assertAlmostEqual(accuracy, 4 / 6, places=4)

    def test_accuracy_at_tpr_garbage_no_positives(self):
        logits = [[80, 20, 0],
                  [20, 70, 10],
                  [20, 10, 70],

                  [90, 10, 0],
                  [30, 40, 30],
                  [10, 80, 10],
                  ]

        targets = [1, 1, 1, 1, 1, 1]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        accuracy = AccuracyAtTPR(unknown_label=1, tpr=0.5, precision=2).compute(logits=logits, targets=targets)

        # thresh 0.9 -> 4 / 6 TPR
        # (4 + 1) / 8 = acc

        self.assertAlmostEqual(accuracy, 0.0, places=4)
