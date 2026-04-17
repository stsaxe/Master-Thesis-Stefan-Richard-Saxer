from typing import Any

import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class OpenAUC(AbstractMetric):

    def __init__(self, unknown_label: int, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'

        self.__unknown_label = unknown_label

    def compute_openauc(self, x1: torch.Tensor, x2: torch.Tensor, pred: torch.Tensor, labels: torch.Tensor) -> float:
        """
        :param x1: open set score for each known class sample (B_k,)
        :param x2: open set score for each unknown class sample (B_u,)
        :param pred: predicted class for each known class sample (B_k,)
        :param labels: correct class for each known class sample (B_k,)
        :return: Open Set Classification Rate
        """
        x1 = x1.cpu().numpy()
        x2 = x2.cpu().numpy()
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()

        x1, x2, correct = x1.tolist(), x2.tolist(), (pred == labels).tolist()

        if not isinstance(x1, list):
            x1 = [x1]
        if not isinstance(x2, list):
            x2 = [x2]
        if not isinstance(correct, list):
            correct = [correct]

        m_x2 = max(x2) + 1e-5

        y_score = [value if hit else m_x2 for value, hit in zip(x1, correct)] + x2
        y_true = [0] * len(x1) + [1] * len(x2)
        open_auc = roc_auc_score(y_true, y_score)
        return open_auc

    def extract_scores(self, probas: torch.Tensor, targets: torch.Tensor) -> tuple[
        Tensor, Tensor | Any, Tensor, Tensor]:
        N, C = probas.shape

        assert self.__unknown_label <= C, 'unknown_label must be in range of Number of Classes + 1'

        mask_known = (targets != self.__unknown_label)
        mask_unknown = (targets == self.__unknown_label)

        probas_known = probas[mask_known]
        probas_unknown = probas[mask_unknown]

        targets_known = targets[mask_known]
        preds_known = torch.argmax(probas_known, dim=1)

        open_set_scores_known = probas_known.gather(dim=1, index=targets_known.unsqueeze(1).long()).squeeze()

        if self.__unknown_label < C:
            open_set_scores_unknown = probas_unknown[:, self.__unknown_label].squeeze()

        else:
            open_set_scores_unknown = (1.0 + 1 / C - probas_unknown.max(dim=1)[0]).squeeze()

        return open_set_scores_known, open_set_scores_unknown, preds_known.long(), targets_known.long()

    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = self.extract_logits_and_targets(**kwargs)

            open_set_scores_known, open_set_scores_unknown, preds_known, targets_known = self.extract_scores(probas,
                                                                                                             targets)

            open_auc = self.compute_openauc(open_set_scores_known, open_set_scores_unknown, preds_known, targets_known)

            return open_auc

    def get_name(self) -> str:
        return 'OpenAUC'
