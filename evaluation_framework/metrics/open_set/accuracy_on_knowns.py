import numpy as np
import torch

from evaluation_framework import Accuracy
from evaluation_framework.metrics.abstract_metric import AbstractMetric
from typing_extensions import override


class AccuracyOnKnowns(AbstractMetric):
    def __init__(self, unknown_label: int, precision: float = 2, average: str = 'micro', add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)

        assert average in ['micro', 'macro', 'weighted'], 'invalid average'

        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(precision, int), 'precision must be of type integer'

        assert precision >= 0, 'precision must be at least 0'
        assert precision <= 5, 'precision must be at max 5'

        self.__average = average
        self.__unknown_label = unknown_label
        self.__precision = precision

    def predict(self, probas: torch.Tensor, threshold: float) -> torch.Tensor:
        assert 0 <= threshold <= 1, 'threshold must be in range 0 to  1'

        max_vals, argmaxes = probas.max(dim=1)

        # if max confidence below threshold => unknown_label
        preds = torch.where(max_vals >= threshold, argmaxes, torch.full_like(argmaxes, self.__unknown_label))

        return preds

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = super().extract_logits_and_targets(**kwargs)

            N, C = probas.shape

            assert self.__unknown_label <= C, 'unknown_label must be in range of Number of Classes + 1'

            mask_known = (targets != self.__unknown_label)
            probas_known = probas[mask_known]
            targets_known = targets[mask_known]
            N_known = targets_known.shape[0]

            # in this case, there is a garbage class
            if self.__unknown_label < C:
                predictions_known = torch.argmax(probas_known, dim=1)

                if N_known < 1:
                    return 0.0

                acc = Accuracy(average=self.__average).compute_accuracy(predictions=predictions_known,
                                                                        targets=targets_known, C=C - 1, N=N_known)

                return acc

            # in this case there is no garbage class and thresholding needs to be applied
            elif self.__unknown_label == C:
                result = []

                if N_known < 1:
                    return torch.zeros(10**self.__precision + 1)

                for threshold in np.linspace(0, 1, num=10 ** self.__precision + 1):
                    predictions_known = self.predict(probas_known, threshold)

                    acc = Accuracy(average=self.__average).compute_accuracy(predictions=predictions_known,
                                                                            targets=targets_known, C=C, N=N_known)
                    result.append(float(acc))

                return torch.tensor(result)

    @override
    def get_name(self) -> str:
        return f"Accuracy on Knowns ({self.__average})"
