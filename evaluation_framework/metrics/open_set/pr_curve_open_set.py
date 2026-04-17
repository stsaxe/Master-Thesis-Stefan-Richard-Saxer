import torch
from sklearn.metrics import precision_recall_curve
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class PRCurveOpenSet(AbstractMetric):
    def __init__(self, unknown_label: int, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'

        self.__unknown_label = unknown_label

    @override
    def compute(self, **kwargs: torch.Tensor) -> float | torch.Tensor:
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

            precision, recall, _ = precision_recall_curve(is_known, prob_known)

            return torch.tensor([precision.tolist(), recall.tolist()]).t()

    @override
    def get_name(self) -> str:
        return f"Precision Recall Curve Open Set"
