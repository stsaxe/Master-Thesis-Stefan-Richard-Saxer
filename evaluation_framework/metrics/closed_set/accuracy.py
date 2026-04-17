import torch
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class Accuracy(AbstractMetric):
    def __init__(self, average: str = 'micro'):
        super().__init__()
        assert average in ['micro', 'macro', 'weighted'], 'invalid average'
        self.__average = average
        self.__epsilon = 1e-8

    def compute_accuracy(self, predictions: torch.Tensor, targets: torch, C: int, N: int) -> float:
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

        if self.__average == 'weighted':
            return float(weighted_acc)
        elif self.__average == 'micro':
            return float(micro_acc)
        else:
            return float(macro_acc)


    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = super().extract_logits_and_targets(**kwargs)

            predictions = torch.argmax(probas, dim=1)
            N, C = probas.shape

            return self.compute_accuracy(predictions, targets, C, N)



    @override
    def get_name(self) -> str:
        if self.__average == 'micro':
            return f"Accuracy"
        else:
            return f"Accuracy ({self.__average})"
