import torch

from evaluation_framework.metrics.open_set.roc_curve_open_set import ROCCurveOpenSet
from evaluation_framework.metrics.abstract_metric import AbstractMetric


class FPRAtTPR(AbstractMetric):
    def __init__(self, unknown_label: int, tpr: float = 0.95, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(tpr, float), 'tpr must be a float'
        assert (0 <= tpr <= 1), 'tpr must be in range from 0 to 1'

        self.__unknown_label = unknown_label
        self.__tpr = tpr

    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        result = ROCCurveOpenSet(unknown_label=self.__unknown_label, add_unknowns=self._add_unknowns).compute(**kwargs)

        fprs, tprs = result[:, 0], result[:, 1]

        assert len(fprs) == len(tprs)
        for idx, tpr in enumerate(tprs):
            if tpr > self.__tpr:
                return float(fprs[idx])

        return 0.0

    def get_name(self) -> str:
        return f"FPR at {100 * self.__tpr}% TPR"

