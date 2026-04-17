import torch

from evaluation_framework import ROCCurveOpenSet, PRCurveOpenSet
from evaluation_framework.metrics.abstract_metric import AbstractMetric


class RecallAtPrecision(AbstractMetric):
    def __init__(self, unknown_label: int, precision: float = 0.95, add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)

        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(precision, float), 'tpr must be a float'
        assert (0 <= precision <= 1), 'precision must be in range from 0 to 1'

        self.__unknown_label = unknown_label
        self.__precision = precision

    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        result = PRCurveOpenSet(unknown_label=self.__unknown_label, add_unknowns=self._add_unknowns).compute(**kwargs)

        precision, recall = result[:, 0], result[:, 1]

        assert len(precision) == len(recall)

        best_recall = 0.0

        for idx, p in enumerate(precision):
            if p > self.__precision and recall[idx] > best_recall:
                best_recall = float(recall[idx])

        return best_recall

    def get_name(self) -> str:
        return f"Recall at {100 * self.__precision}% Precision"
