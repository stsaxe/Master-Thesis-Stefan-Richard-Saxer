import torch

from evaluation_framework import ROCCurveOpenSet, PRCurveOpenSet
from evaluation_framework.metrics.abstract_metric import AbstractMetric


class PrecisionAtRecall(AbstractMetric):
    def __init__(self, unknown_label: int, recall: float = 0.95, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)

        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(recall, float), 'recall must be a float'
        assert (0 <= recall <= 1), 'recall must be in range from 0 to 1'

        self.__unknown_label = unknown_label
        self.__recall = recall

    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        result = PRCurveOpenSet(unknown_label=self.__unknown_label, add_unknowns=self._add_unknowns).compute(**kwargs)

        precision, recall = result[:, 0], result[:, 1]

        assert len(precision) == len(recall)

        best_precision = 0.0

        for idx, r in enumerate(recall.numpy()):
            if float(r) > self.__recall and precision[idx] > best_precision:
                best_precision = precision[idx]

        return float(best_precision)

    def get_name(self) -> str:
        return f"Precision at {100 * self.__recall}% Recall"
