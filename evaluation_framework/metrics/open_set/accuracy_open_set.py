import numpy as np
import torch
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class AccuracyOpenSet(AbstractMetric):
    def __init__(self, unknown_label: int, precision: int = 2, average: str = 'micro', add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert average in ['micro', 'macro', 'weighted', 'binary', 'balanced'], 'invalid average'

        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(precision, int), 'precision must be of type integer'

        assert precision >= 0, 'precision must be at least 0'
        assert precision <= 5, 'precision must be at max 5'

        self.__average = average
        self.__epsilon = 1e-8
        self.__precision = precision
        self.__unknown_label = unknown_label

    def compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, N: int,
                         C: int) -> torch.Tensor | float:
        total_correct = (predictions == targets).sum().item()

        correct_per_class = torch.zeros(C, dtype=torch.int32)
        support_per_class = torch.zeros(C, dtype=torch.int32)

        for cls in range(C):
            mask = (targets == cls)
            support = mask.sum().item()
            support_per_class[cls] = support

            if support > 0:
                correct = (predictions[mask] == cls).sum().item()
                correct_per_class[cls] = correct

        acc_per_class = correct_per_class.float() / (support_per_class.float() + self.__epsilon)

        micro_acc = total_correct / N
        macro_acc = acc_per_class.mean().item()
        weighted_acc = (acc_per_class * (support_per_class.float() / N)).sum().item()

        correct_per_class_knowns = torch.cat(
            [correct_per_class[:self.__unknown_label], correct_per_class[self.__unknown_label + 1:]])
        support_per_class_knowns = torch.cat(
            [support_per_class[:self.__unknown_label], support_per_class[self.__unknown_label + 1:]])

        correct_unknowns = correct_per_class[self.__unknown_label]
        support_unknowns = support_per_class[self.__unknown_label]

        acc_known = correct_per_class_knowns.sum().item() / (support_per_class_knowns.sum().item() + self.__epsilon)
        acc_unknown = correct_unknowns.sum().item() / (support_unknowns.sum().item() + self.__epsilon)

        balanced_acc = (acc_known + acc_unknown) / 2

        knowns_mask = (targets != self.__unknown_label)
        unknowns_mask = (targets == self.__unknown_label)

        non_rejected_knowns = len(predictions[knowns_mask]) - (
                    predictions[knowns_mask] == self.__unknown_label).sum().item()
        rejected_unknwons = (predictions[unknowns_mask] == self.__unknown_label).sum().item()

        binary_acc = (rejected_unknwons + non_rejected_knowns) / N

        if self.__average == 'weighted':
            return float(weighted_acc)
        elif self.__average == 'micro':
            return float(micro_acc)
        elif self.__average == 'macro':
            return float(macro_acc)
        elif self.__average == 'balanced':
            return float(balanced_acc)
        elif self.__average == 'binary':
            return binary_acc

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

            # in this case, there is a garbage class
            if self.__unknown_label < C:
                predictions = torch.argmax(probas, dim=1)
                return self.compute_accuracy(predictions, targets, N, C)

            # in this case there is no garbage class and thresholding needs to be applied
            elif self.__unknown_label == C:
                result = []

                for threshold in np.linspace(0, 1, num=10 ** self.__precision + 1):
                    predictions = self.predict(probas, threshold)
                    acc = self.compute_accuracy(predictions, targets, N, C+1)
                    result.append(acc)

                return torch.tensor(result)

    @override
    def get_name(self) -> str:
        return f"Accuracy Open Set ({self.__average})"
