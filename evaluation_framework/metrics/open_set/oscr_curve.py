import numpy as np
import torch
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class OSCRCurve(AbstractMetric):
    def __init__(self, unknown_label: int, precision: int = 2, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert isinstance(precision, int), 'precision must be of type integer'

        assert precision >= 0, 'precision must be at least 0'
        assert precision <= 4, 'precision must be at max 4'
        assert unknown_label >= 0, 'unknown label must be positive'

        self.__unknown_label = unknown_label
        self.__precision = precision

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():

            _, probas, targets = self.extract_logits_and_targets(**kwargs)
            targets = targets.long()

            N, C = probas.shape

            assert self.__unknown_label == C or self.__unknown_label == C - 1

            if self.__unknown_label == C - 1:
                probas_known = probas[:, :-1]

            elif self.__unknown_label == C:
                probas_known = probas

            preds_known = probas_known.argmax(dim=1)

            fprs, ccrs, osa = [], [], []

            unknown_count = (targets == self.__unknown_label).sum().int().item()
            known_count = int(len(targets) - unknown_count)

            if known_count == 0 or unknown_count == 0:
                return torch.zeros(3)

            assert unknown_count > 0, 'number of unknowns must be at least 1'
            assert known_count > 0, 'number of knows must be at least 1'

            alpha = known_count / (unknown_count + known_count)

            epsilons = np.linspace(0, 1, num=10 ** self.__precision + 1)

            for eps in epsilons[::-1]:
                fpr = (torch.max(probas_known[targets == self.__unknown_label], dim=1)[
                           0] >= eps).sum().item() / unknown_count

                ccr = (torch.max(probas_known[(preds_known == targets) & (targets != self.__unknown_label)], dim=1)[0] > eps).sum().item() / known_count

                fprs.append(fpr)
                ccrs.append(ccr)
                osa.append(alpha * ccr + (1 - alpha) * (1 - fpr))

            return torch.tensor([fprs, ccrs, osa]).t()

    @override
    def get_name(self) -> str:
        return "OSCR Curve"
