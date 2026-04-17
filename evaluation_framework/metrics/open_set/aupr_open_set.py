import torch
from sklearn.metrics import average_precision_score
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class AUPROpenSet(AbstractMetric):
    def __init__(self, unknown_label: int, average: str = "micro", add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)

        assert average in ["micro", "macro", "weighted"], 'average must be "micro", "macro" or "weighted"'
        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'

        self.__average = average
        self.__unknown_label = unknown_label

    @override
    def compute(self, **kwargs: torch.Tensor) -> float:
        with torch.no_grad():
            _, probas, targets = self.extract_logits_and_targets(**kwargs)

            N, C = probas.shape

            assert self.__unknown_label <= C, 'unknown_label must be in range of Number of Classes + 1'

            is_known = (targets != self.__unknown_label).int().cpu().numpy()

            # in this case, there is a garbage class
            if self.__unknown_label < C:
                prob_known = torch.cat([probas[:, :self.__unknown_label], probas[:, self.__unknown_label + 1:]], dim=1)

                prob_known = prob_known.max(dim=1)[0].cpu().numpy()


            # in this case there is no garbage class
            elif self.__unknown_label == C:
                prob_known = probas.max(dim=1)[0].cpu().numpy()

            return average_precision_score(is_known, prob_known, average=self.__average)

    @override
    def get_name(self) -> str:
        return f"AUPR Open Set ({self.__average})"
