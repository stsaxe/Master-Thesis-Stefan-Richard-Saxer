import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from evaluation_framework.metrics.closed_set.aupr import AUPR


class TestAUPR(unittest.TestCase):
    def test_AUPR_macro(self):
        logits_raw = [[80, 20, 0],
                      [30, 70, 10],
                      [50, 40, 10],

                      [10, 80, 10],
                      [30, 30, 40],

                      [10, 0, 90]]

        targets = np.zeros(500)
        targets[:300] = 0
        targets[300:450] = 1
        targets[450:] = 2

        logits_raw = np.random.uniform(low=0, high=1, size=(500, 3))
        logits_raw = logits_raw / logits_raw.sum(axis=1, keepdims=True)

        logits_raw = np.array(logits_raw)
        logits = torch.Tensor(logits_raw)
        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        aupr = AUPR().compute(logits=logits, targets=targets)

        y_true_c0 = np.zeros(500)
        y_true_c0[:300] = 1

        y_true_c1 = np.zeros(500)
        y_true_c1[300:450] = 1

        y_true_c2 = np.zeros(500)
        y_true_c2[450:] = 1

        aupr_c0 = average_precision_score(y_true=y_true_c0, y_score=logits_raw[:, 0])
        aupr_c1 = average_precision_score(y_true=y_true_c1, y_score=logits_raw[:, 1])
        aupr_c2 = average_precision_score(y_true=y_true_c2, y_score=logits_raw[:, 2])

        aupr_manual = (aupr_c0 + aupr_c1 + aupr_c2) / 3


        self.assertAlmostEqual(aupr - aupr_manual, 0, places=5)

        name = AUPR().get_name()
        self.assertEqual(name, 'AUPR (macro)')

    def test_AUPR_weighted(self):
        logits_raw = [[80, 20, 0],
                      [30, 70, 10],
                      [50, 40, 10],

                      [10, 80, 10],
                      [30, 30, 40],

                      [10, 0, 90]]

        targets = np.zeros(500)
        targets[:300] = 0
        targets[300:450] = 1
        targets[450:] = 2

        logits_raw = np.random.uniform(low=0, high=1, size=(500, 3))
        logits_raw = logits_raw / logits_raw.sum(axis=1, keepdims=True)

        logits_raw = np.array(logits_raw)
        logits = torch.Tensor(logits_raw)
        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        aupr = AUPR(average='weighted').compute(logits=logits, targets=targets)

        y_true_c0 = np.zeros(500)
        y_true_c0[:300] = 1

        y_true_c1 = np.zeros(500)
        y_true_c1[300:450] = 1

        y_true_c2 = np.zeros(500)
        y_true_c2[450:] = 1

        aupr_c0 = average_precision_score(y_true=y_true_c0, y_score=logits_raw[:, 0])
        aupr_c1 = average_precision_score(y_true=y_true_c1, y_score=logits_raw[:, 1])
        aupr_c2 = average_precision_score(y_true=y_true_c2, y_score=logits_raw[:, 2])

        aupr_manual = (300 * aupr_c0 + 150 * aupr_c1 + 50 * aupr_c2) / 500

        self.assertAlmostEqual(aupr - aupr_manual, 0, places=5)


        name = AUPR(average='weighted').get_name()
        self.assertEqual(name, 'AUPR (weighted)')


    def test_assertions(self):
        with self.assertRaises(AssertionError):
            AUPR(average='micro')

        with self.assertRaises(AssertionError):
            AUPR(average=123)
