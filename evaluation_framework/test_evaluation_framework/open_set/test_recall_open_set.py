import unittest

import torch

from evaluation_framework.metrics.open_set.recall_open_set import RecallOpenSet


class TestRecallOpenSet(unittest.TestCase):
    def test_recall_micro_garbage(self):
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

        metric = RecallOpenSet(unknown_label=2, average='micro')

        recall = metric.compute(logits=logits, targets=targets)

        solution = (2 + 2 + 2) / 10
        self.assertAlmostEqual(solution, recall, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'Recall Open Set (micro)')

    def test_recall_macro_garbage(self):
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

        metric = RecallOpenSet(unknown_label=2, average='macro')

        recall = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 4
        p_c1 = 2 / 3
        p_c2 = 2 / 3

        solution = (p_c0 + p_c1 + p_c2) / 3

        self.assertAlmostEqual(solution, recall, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'Recall Open Set (macro)')

    def test_recall_weighted_garbage(self):
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

        metric = RecallOpenSet(unknown_label=2, average='weighted')

        recall = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 3
        p_c1 = 2 / 3
        p_c2 = 2 / 4

        solution = (3 * p_c0 + 3 * p_c1 + 4 * p_c2) / 10

        self.assertAlmostEqual(solution, recall, places=4)

        name = metric.get_name()
        self.assertEqual(name, 'Recall Open Set (weighted)')

    def test_recall_balanced_garbage(self):
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

        metric = RecallOpenSet(unknown_label=2, average='balanced')

        recall = metric.compute(logits=logits, targets=targets)

        p_c0 = 2 / 3
        p_c1 = 2 / 3
        p_c2 = 2 / 4

        solution = (4 / 6 + p_c2) / 2

        self.assertAlmostEqual(solution, recall, places=4)

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

        metric = RecallOpenSet(unknown_label=2, average='binary')

        recall = metric.compute(logits=logits, targets=targets)

        solution = 5 / 6

        self.assertAlmostEqual(solution, recall, places=4)

    def test_recall_binary_threshold(self):
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

        metric = RecallOpenSet(unknown_label=2, average='binary', precision=1)

        recall = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(recall[0]), 5 / 5, places=4)
        self.assertAlmostEqual(float(recall[5]), 5 / 5, places=4)
        self.assertAlmostEqual(float(recall[7]), 3 / 5, places=4)
        self.assertAlmostEqual(float(recall[9]), 1 / 5, places=4)
