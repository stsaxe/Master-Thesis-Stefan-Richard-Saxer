import unittest

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from evaluation_framework.metrics.open_set.auroc_open_set import AUROCOpenSet


class TestAUPR_OpenSet(unittest.TestCase):
    def test_auroc_garbage(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=1).compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 0, 0, 0, 1, 1]
        scores = [70, 20, 80, 30, 50, 10, 50, 30]

        auroc_manual = roc_auc_score(targets_binary, scores)

        self.assertAlmostEqual(auroc, auroc_manual, places=4)

        name = AUROCOpenSet(unknown_label=1, average='micro').get_name()

        self.assertEqual('AUROC Open Set (micro)', name)

    def test_auroc_garbage_v2(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        scores = np.array(logits)[:, :2].max(axis=1)

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=2).compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 1, 1, 1, 0, 0]
        scores = [70, 60, 10, 40, 40, 80, 30, 50]

        auroc_manual = roc_auc_score(targets_binary, scores)

        self.assertTrue(abs(auroc_manual - auroc) < 2*1e-2)

    def test_auroc_garbage_macro(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=1, average='macro').compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 0, 0, 0, 1, 1]
        scores = [70, 20, 80, 30, 50, 10, 50, 30]

        auroc_manual = roc_auc_score(targets_binary, scores, average='macro')

        self.assertAlmostEqual(auroc, auroc_manual, places=4)

        name = AUROCOpenSet(unknown_label=1, average='macro').get_name()

        self.assertEqual('AUROC Open Set (macro)', name)

    def test_auroc_garbage_weighted(self):
        logits = [[70, 20, 10],
                  [20, 60, 20],
                  [10, 10, 80],

                  [30, 40, 30],
                  [10, 40, 50],
                  [10, 80, 10],

                  [20, 30, 50],
                  [20, 50, 30]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=1, average='weighted').compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 0, 0, 0, 1, 1]
        scores = [70, 20, 80, 30, 50, 10, 50, 30]

        auroc_manual = roc_auc_score(targets_binary, scores, average='weighted')

        self.assertAlmostEqual(auroc, auroc_manual, places=4)

        name = AUROCOpenSet(unknown_label=1, average='weighted').get_name()

        self.assertEqual('AUROC Open Set (weighted)', name)

    def test_auroc_thresholding_macro(self):
        logits = [[70, 30],
                  [10, 90],
                  [50, 50],

                  [60, 40],
                  [40, 60],
                  [20, 80],

                  [20, 80],
                  [60, 40]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=2, average='macro').compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 1, 1, 1, 0, 0]
        scores = [70, 90, 50, 60, 60, 80, 80, 60]

        auroc_manual = roc_auc_score(targets_binary, scores, average='macro')

        self.assertAlmostEqual(auroc, auroc_manual, places=4)

    def test_auroc_thresholding_micro(self):
        logits = [[70, 30],
                  [10, 90],
                  [50, 50],

                  [60, 40],
                  [40, 60],
                  [20, 80],

                  [20, 80],
                  [60, 40]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=2, average='micro').compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 1, 1, 1, 0, 0]
        scores = [70, 90, 50, 60, 60, 80, 80, 60]

        auroc_manual = roc_auc_score(targets_binary, scores, average='micro')

        self.assertAlmostEqual(auroc, auroc_manual, places=4)

    def test_auroc_thresholding_weighted(self):
        logits = [[70, 30],
                  [10, 90],
                  [50, 50],

                  [60, 40],
                  [40, 60],
                  [20, 80],

                  [20, 80],
                  [60, 40]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        auroc = AUROCOpenSet(unknown_label=2, average='weighted').compute(logits=logits, targets=targets)

        targets_binary = [1, 1, 1, 1, 1, 1, 0, 0]
        scores = [70, 90, 50, 60, 60, 80, 80, 60]

        auroc_manual = roc_auc_score(targets_binary, scores, average='weighted')

        self.assertAlmostEqual(auroc, auroc_manual, places=4)
