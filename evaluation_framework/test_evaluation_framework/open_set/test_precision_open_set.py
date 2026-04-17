import unittest

import torch

from evaluation_framework.metrics.open_set.precision_open_set import PrecisionOpenSet


class TestPrecisionOpenSet(unittest.TestCase):
    def test_precision_micro_garbage(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='micro')

        precision = metric.compute(logits=logits, targets=targets)

        solution = (2 + 2 + 2) / 10
        self.assertAlmostEqual(solution, precision, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'Precision Open Set (micro)')

    def test_precision_macro_garbage(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='macro')

        precision = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 4
        p_c1 = 2 / 3
        p_c2 = 2 / 3

        solution = (p_c0 + p_c1 + p_c2) / 3

        self.assertAlmostEqual(solution, precision, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'Precision Open Set (macro)')

    def test_precision_macro_garbage_v2(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=1, average='macro')

        precision = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 4
        p_c1 = 2 / 3
        p_c2 = 2 / 3

        solution = (p_c0 + p_c1 + p_c2) / 3

        self.assertAlmostEqual(solution, precision, places=4)

    def test_precision_weighted_garbage(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='weighted')

        precision = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 4
        p_c1 = 2 / 3
        p_c2 = 2 / 3

        solution = (3 * p_c0 + 3 * p_c1 + 4 * p_c2) / 10

        self.assertAlmostEqual(solution, precision, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'Precision Open Set (weighted)')

    def test_precision_balanced_garbage(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='balanced')

        precision = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 4
        p_c1 = 2 / 3
        p_c2 = 2 / 3

        solution = (4 / 7 + p_c2) / 2

        self.assertAlmostEqual(solution, precision, places=4)

    def test_precision_binary_garbage(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='binary')

        precision = metric.compute(logits=logits, targets=targets)

        solution = 5 / 7

        self.assertAlmostEqual(solution, precision, places=4)

    def test_precision_binary_garbage_v2(self):
        logits = [[90, 10, 0],
                  [0, 90, 10],
                  [70, 10, 20],

                  [10, 10, 80],
                  [30, 40, 30],
                  [30, 60, 10],

                  [10, 10, 80],
                  [40, 30, 30],
                  [30, 30, 40],
                  [60, 30, 10],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=1, average='binary')

        precision = metric.compute(logits=logits, targets=targets)

        solution = 6 / (6 + 1)

        self.assertAlmostEqual(solution, precision, places=4)

    def test_precision_micro_threshold(self):
        logits = [[90, 10],
                  [20, 80],
                  [60, 40],

                  [30, 70],
                  [40, 60],

                  [80, 20],
                  [55, 45],
                  [30, 70]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='micro', precision=1)

        precision = metric.compute(logits=logits, targets=targets)

        self.assertEqual(precision[0].float(), 4 / 8)
        self.assertEqual(precision[5].float(), 4 / 8)
        self.assertEqual(precision[6].float(), 5 / 8)
        self.assertEqual(precision[7].float(), 3 / 8)
        self.assertEqual(precision[10].float(), 3 / 8)

    def test_precision_macro_threshold(self):
        logits = [[90, 10],
                  [20, 80],
                  [60, 40],

                  [30, 70],
                  [40, 60],

                  [80, 20],
                  [55, 45],
                  [30, 70]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='macro', precision=1)

        precision = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(precision[0]), (2 / 4 + 2 / 4 + 0) / 3, places=4)
        self.assertAlmostEqual(float(precision[5]), (2 / 4 + 2 / 4 + 0) / 3, places=4)
        self.assertAlmostEqual(float(precision[6]), (2 / 3 + 2 / 4 + 1) / 3, places=4)
        self.assertAlmostEqual(float(precision[10]), (3 / 8) / 3, places=4)

    def test_precision_weighted_threshold(self):
        logits = [[90, 10],
                  [20, 80],
                  [60, 40],

                  [30, 70],
                  [40, 60],

                  [80, 20],
                  [55, 45],
                  [30, 70]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='weighted', precision=1)

        precision = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(precision[0]), (3 * 2 / 4 + 2 * 2 / 4 + 0) / 8, places=4)
        self.assertAlmostEqual(float(precision[5]), (3 * 2 / 4 + 2 * 2 / 4 + 0) / 8, places=4)
        self.assertAlmostEqual(float(precision[6]), (3 * 2 / 3 + 2 * 2 / 4 + 3 * 1) / 8, places=4)
        self.assertAlmostEqual(float(precision[10]), (3 * 3 / 8) / 8, places=4)

    def test_precision_balanced_threshold(self):
        logits = [[90, 10],
                  [20, 80],
                  [60, 40],

                  [30, 70],
                  [40, 60],

                  [80, 20],
                  [55, 45],
                  [30, 70]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='balanced', precision=1)

        precision = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(precision[0]), ((2 / 4 + 2 / 4) / 2 + 0) / 2, places=4)
        self.assertAlmostEqual(float(precision[5]), ((2 / 4 + 2 / 4) / 2 + 0) / 2, places=4)
        self.assertAlmostEqual(float(precision[6]), ((4 / 7) + 1) / 2, places=4)
        self.assertAlmostEqual(float(precision[10]), (0 + 3 / 8) / 2, places=4)

    def test_precision_binary_threshold(self):
        logits = [[90, 10],
                  [20, 80],
                  [60, 40],

                  [30, 70],
                  [40, 60],

                  [80, 20],
                  [55, 45],
                  [30, 70]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = PrecisionOpenSet(unknown_label=2, average='binary', precision=1)

        precision = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(precision[0]), 5 / 8, places=4)
        self.assertAlmostEqual(float(precision[5]), 5 / 8, places=4)
        self.assertAlmostEqual(float(precision[6]), 5 / 7, places=4)
        self.assertAlmostEqual(float(precision[10]), 0, places=4)
