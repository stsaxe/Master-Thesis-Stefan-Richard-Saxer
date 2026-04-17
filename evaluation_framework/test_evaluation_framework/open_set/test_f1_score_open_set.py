import unittest

import torch
from sklearn.metrics import f1_score

from evaluation_framework import F1ScoreOpenSet


class Testf1ScoreOpenSet(unittest.TestCase):
    def test_name(self):
        metric = F1ScoreOpenSet(unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'F1 Score Open Set (micro)')

        metric = F1ScoreOpenSet(average='weighted', unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'F1 Score Open Set (weighted)')

        metric = F1ScoreOpenSet(average='macro', unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'F1 Score Open Set (macro)')

        metric = F1ScoreOpenSet(unknown_label=1, average='balanced')
        name = metric.get_name()

        self.assertEqual(name, 'F1 Score Open Set (balanced)')

        metric = F1ScoreOpenSet(average='binary', unknown_label=1)
        name = metric.get_name()

        self.assertEqual(name, 'F1 Score Open Set (binary)')

    def test_f1_score_micro(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 0, 0]

        probas = torch.tensor(logits)
        logits = torch.log(torch.Tensor(logits) / 100)

        predictions = probas.argmax(dim=1).numpy()
        targets = torch.Tensor(targets)

        metric = F1ScoreOpenSet(average='micro', unknown_label=2)

        score = metric.compute(logits=logits, targets=targets)

        f1 = f1_score(targets.numpy(), predictions, average='micro')

        self.assertAlmostEqual(f1, score, places=4)

    def test_f1_score_macro(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 0, 0]

        probas = torch.tensor(logits)
        logits = torch.log(torch.Tensor(logits) / 100)

        predictions = probas.argmax(dim=1).numpy()
        targets = torch.Tensor(targets)

        metric = F1ScoreOpenSet(average='macro', unknown_label=2)

        score = metric.compute(logits=logits, targets=targets)

        f1 = f1_score(targets.numpy(), predictions, average='macro')

        self.assertAlmostEqual(f1, score, places=4)

    def test_f1_score_weighted(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 0, 0]

        probas = torch.tensor(logits)
        logits = torch.log(torch.Tensor(logits) / 100)

        predictions = probas.argmax(dim=1).numpy()
        targets = torch.Tensor(targets)

        metric = F1ScoreOpenSet(average='weighted', unknown_label=2)

        score = metric.compute(logits=logits, targets=targets)

        f1 = f1_score(targets.numpy(), predictions, average='weighted')

        self.assertAlmostEqual(f1, score, places=4)

    def test_f1_score_binary(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 30, 50]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 1, 2, 0, 0]

        binary_targets = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1]).long().numpy()
        predictions_binary = torch.tensor([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]).long().numpy()

        logits = torch.log(torch.Tensor(logits) / 100)

        targets = torch.Tensor(targets)

        metric = F1ScoreOpenSet(average='binary', unknown_label=2)

        score = metric.compute(logits=logits, targets=targets)

        f1 = f1_score(binary_targets, predictions_binary)

        self.assertAlmostEqual(f1, score, places=4)

    def test_f1_score_binary_thresholding(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30],

                  [20, 30, 50],
                  [20, 50, 30],
                  [20, 30, 50],
                  [20, 30, 50]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3]

        binary_targets = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]).long().numpy()

        logits = torch.log(torch.Tensor(logits) / 100)

        targets = torch.Tensor(targets)

        metric = F1ScoreOpenSet(average='binary', unknown_label=3, precision=1)

        score = metric.compute(logits=logits, targets=targets)

        pred_binary_07 = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]).long()

        f1_07 = f1_score(binary_targets, pred_binary_07)

        self.assertAlmostEqual(score[0].float(), 0.8, places=5)
        self.assertAlmostEqual(score[7].float(), f1_07, places=5)
        self.assertAlmostEqual(score[10].float(), 0, places = 5)

