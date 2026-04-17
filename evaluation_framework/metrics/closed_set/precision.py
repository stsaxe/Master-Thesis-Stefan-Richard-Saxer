import torch
from sklearn.metrics import precision_score
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class Precision(AbstractMetric):
    def __init__(self, average: str = "micro"):
        super().__init__()
        assert average in ["micro", "macro", "weighted"]

        self.__average = average

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = self.extract_logits_and_targets(**kwargs)
            preds = torch.argmax(probas, dim=1)

            y_true = targets.cpu().numpy()
            y_pred = preds.cpu().numpy()

            return float(precision_score(y_true, y_pred, average=self.__average, zero_division=0))

    @override
    def get_name(self) -> str:
        return f"Precision ({self.__average})"
