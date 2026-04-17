import numpy as np
import torch
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class AccuracyAtTPR(AbstractMetric):
    def __init__(self, unknown_label: int, tpr: float = 0.95, precision: int = 2, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(tpr, float), 'tpr must be a float'
        assert (0 <= tpr <= 1), 'tpr must be in range from 0 to 1'
        assert isinstance(precision, int), 'precision must be of type integer'

        assert precision >= 0, 'precision must be at least 0'
        assert precision <= 4, 'precision must be at max 4'

        self.__unknown_label = unknown_label
        self.__tpr = tpr
        self.__precision = precision

    @override
    def compute(self, **kwargs: torch.Tensor) -> float:
        with torch.no_grad():
            _, probas, targets = self.extract_logits_and_targets(**kwargs)

            N, C = probas.shape

            assert self.__unknown_label <= C, 'unknown_label must be in range of Number of Classes + 1'

            # in this case, there is a garbage class
            if self.__unknown_label < C:
                known_scores, _ = torch.cat([probas[:, :self.__unknown_label],  probas[:, self.__unknown_label + 1:]], dim=1).max(dim=1)
                known_scores = known_scores.flatten()

            # in this case there is no garbage class
            elif self.__unknown_label == C:
                known_scores, _ = probas.max(dim=1)

            known_scores = known_scores.flatten()
            binary_targets = (targets != self.__unknown_label).int().flatten()

            return self.compute_accuracy_at_tpr(known_scores, binary_targets)

    def compute_accuracy_at_tpr(self, known_scores: torch.Tensor, targets: torch.Tensor) -> float:
        # number of positives / negatives
        # label known = 1
        # label unknown = 0
        P = (targets == 1).sum().item()
        N = (targets == 0).sum().item()

        if P == 0:
            return 0.0

        assert P + N == len(targets)

        thresh = np.linspace(0, 1, num=10 ** self.__precision + 1)

        for thresh in thresh[::-1]:
            preds_known = (known_scores >= thresh).long()

            TP = int(((preds_known == 1) & (targets == 1)).sum().item())
            TN = int(((preds_known == 0) & (targets == 0)).sum().item())

            tpr = TP / P
            acc = (TP + TN) / (P + N)

            if tpr < self.__tpr:
                # still below desired recall
                continue
            else:
                return acc

        return 0.0

    @override
    def get_name(self) -> str:
        return f"Accuracy at {100 * self.__tpr}% TPR"
