import unittest

import torch

from evaluation_framework.metrics.open_set.open_auc import OpenAUC


class TestOpenAUC(unittest.TestCase):
    def test_open_auc_garbage(self):
        logits = [[70, 20, 10],
                  [10, 90, 0],
                  [80, 10, 10],

                  [30, 40, 30],
                  [40, 30, 30],
                  [90, 10, 0],

                  [30, 30, 40],
                  [60, 30, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        probas = torch.tensor(logits) / 100
        targets = torch.Tensor(targets)

        metric = OpenAUC(unknown_label=1)

        open_set_scores_known, open_set_scores_unknown, preds_known, targets_known = metric.extract_scores(probas,
                                                                                                           targets)

        open_set_scores_known_solution = torch.tensor([70, 10, 80, 40, 10]) / 100
        open_set_scores_unknown_solution = torch.tensor([40, 30, 10]) / 100
        preds_known_solution = torch.tensor([0, 1, 0, 2, 0]).long()
        targets_known_solution = torch.tensor([0, 0, 0, 2, 2]).long()

        self.assertTrue((open_set_scores_known_solution).isclose(open_set_scores_known).all())
        self.assertTrue((open_set_scores_unknown).isclose(open_set_scores_unknown_solution).all())
        self.assertTrue((preds_known).isclose(preds_known_solution).all())
        self.assertTrue((targets_known).isclose(targets_known_solution).all())

        open_auc = metric.compute(logits=torch.log(torch.Tensor(logits) / 100), targets=targets)
        open_auc_manual = metric.compute_openauc(open_set_scores_known, open_set_scores_unknown, preds_known,
                                                 targets_known)

        self.assertAlmostEqual(open_auc, open_auc_manual, places=4)

        name = OpenAUC(unknown_label=1).get_name()

        self.assertEqual('OpenAUC', name)

    def test_open_auc_garbage_v2(self):
        logits = [[70, 20, 10],
                  [10, 90, 0],
                  [80, 10, 10],

                  [30, 40, 30],
                  [25, 35, 40],
                  [90, 10, 0],

                  [30, 30, 40],
                  [60, 30, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        probas = torch.tensor(logits) / 100
        targets = torch.Tensor(targets)

        metric = OpenAUC(unknown_label=2)

        open_set_scores_known, open_set_scores_unknown, preds_known, targets_known = metric.extract_scores(probas,
                                                                                                           targets)

        open_set_scores_known_solution = torch.tensor([70, 10, 80, 40, 35, 10]) / 100
        open_set_scores_unknown_solution = torch.tensor([40, 10]) / 100
        preds_known_solution = torch.tensor([0, 1, 0, 1, 2., 0]).long()
        targets_known_solution = torch.tensor([0, 0, 0, 1, 1, 1]).long()

        self.assertTrue((open_set_scores_known_solution).isclose(open_set_scores_known).all())
        self.assertTrue((open_set_scores_unknown).isclose(open_set_scores_unknown_solution).all())
        self.assertTrue((preds_known).isclose(preds_known_solution).all())
        self.assertTrue((targets_known).isclose(targets_known_solution).all())

        open_auc = metric.compute(logits=torch.log(torch.Tensor(logits)), targets=targets)
        open_auc_manual = metric.compute_openauc(open_set_scores_known, open_set_scores_unknown, preds_known,
                                                 targets_known)

        self.assertAlmostEqual(open_auc, open_auc_manual, places=4)



    def test_open_auc_garbage_v3(self):
        logits = torch.eye(100, 100)

        targets = [i for i in range(100)]

        probas = torch.tensor(logits)
        targets = torch.Tensor(targets)

        metric = OpenAUC(unknown_label=100 - 1)

        open_set_scores_known, open_set_scores_unknown, preds_known, targets_known = metric.extract_scores(probas,
                                                                                                           targets)


        open_set_scores_known_solution = torch.ones(99)
        open_set_scores_unknown_solution = torch.ones(1)
        preds_known_solution = targets[:-1].long()
        targets_known_solution = targets[:-1].long()

        self.assertTrue((open_set_scores_known_solution).isclose(open_set_scores_known).all())
        self.assertTrue((open_set_scores_unknown).isclose(open_set_scores_unknown_solution).all())
        self.assertTrue((preds_known).isclose(preds_known_solution).all())
        self.assertTrue((targets_known).isclose(targets_known_solution).all())

        open_auc = metric.compute(logits=torch.log(torch.Tensor(logits)), targets=targets)
        open_auc_manual = metric.compute_openauc(open_set_scores_known, open_set_scores_unknown, preds_known,
                                                 targets_known)

        self.assertAlmostEqual(open_auc, open_auc_manual, places=4)

    def test_open_auc_thresholding(self):
        logits = [[70, 20, 10],
                  [10, 90, 0],
                  [80, 10, 10],

                  [30, 40, 30],
                  [40, 30, 30],
                  [90, 10, 0],

                  [30, 30, 40],
                  [60, 30, 10],

                  [30, 30, 40],
                  [60, 30, 10],
                  [90, 0, 10],

                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3]

        probas = torch.tensor(logits) / 100
        targets = torch.Tensor(targets)

        metric = OpenAUC(unknown_label=3)

        open_set_scores_known, open_set_scores_unknown, preds_known, targets_known = metric.extract_scores(probas,
                                                                                                           targets)

        open_set_scores_known_solution = torch.tensor([70, 10, 80, 40, 30, 10, 40, 10]) / 100
        open_set_scores_unknown_solution = 1 + 1 / 3 - torch.tensor([40, 60, 90]) / 100
        preds_known_solution = torch.tensor([0, 1, 0, 1, 0, 0, 2, 0]).long()
        targets_known_solution = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]).long()

        self.assertTrue((open_set_scores_known_solution).isclose(open_set_scores_known).all())
        self.assertTrue((open_set_scores_unknown).isclose(open_set_scores_unknown_solution).all())
        self.assertTrue((preds_known).isclose(preds_known_solution).all())
        self.assertTrue((targets_known).isclose(targets_known_solution).all())

        open_auc = metric.compute(logits=torch.log(torch.Tensor(logits) / 100), targets=targets)
        open_auc_manual = metric.compute_openauc(open_set_scores_known, open_set_scores_unknown, preds_known,
                                                 targets_known)

        self.assertAlmostEqual(open_auc, open_auc_manual, places=4)

    def test_open_auc_thresholding_v2(self):
        logits = [[100, 0, 0],
                  [100, 0, 0],
                  [100, 0, 0],

                  [0, 100, 0],
                  [0, 100, 0],
                  [0, 100, 0],

                  [0, 0, 100],
                  [0, 0, 100],

                  [100 / 3, 100 / 3, 100 / 3],
                  [100 / 3, 100 / 3, 100 / 3],
                  [100 / 3, 100 / 3, 100 / 3],

                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3]

        probas = torch.tensor(logits) / 100
        targets = torch.Tensor(targets)

        metric = OpenAUC(unknown_label=3)

        open_set_scores_known, open_set_scores_unknown, preds_known, targets_known = metric.extract_scores(probas,
                                                                                                           targets)

        open_set_scores_known_solution = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]).float()
        open_set_scores_unknown_solution = 1 + 1 / 3 - torch.tensor([1 / 3, 1 / 3, 1 / 3])
        preds_known_solution = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]).long()
        targets_known_solution = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2]).long()

        self.assertTrue((open_set_scores_known_solution).isclose(open_set_scores_known).all())
        self.assertTrue((open_set_scores_unknown).isclose(open_set_scores_unknown_solution).all())
        self.assertTrue((preds_known).isclose(preds_known_solution).all())
        self.assertTrue((targets_known).isclose(targets_known_solution).all())

        open_auc = metric.compute(logits=torch.log(torch.Tensor(logits) / 100), targets=targets)
        open_auc_manual = metric.compute_openauc(open_set_scores_known, open_set_scores_unknown, preds_known,
                                                 targets_known)

        self.assertAlmostEqual(open_auc, open_auc_manual, places=4)
