import torch
from sklearn.metrics import f1_score
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric
from evaluation_framework.metrics.closed_set.precision import Precision
from evaluation_framework.metrics.closed_set.recall import Recall


class F1Score(AbstractMetric):
    def __init__(self, average: str = "micro"):
        super().__init__()
        """
        :param average: Averaging method: 'micro', 'macro' or 'weighted'
        average	Best For:
            'macro'	        Class imbalance, equal class importance; doesn’t account for class frequency
            'weighted'	    Imbalanced data, performance overview;	more realistic average F1
            'micro'	        Global view of prediction accuracy;	skips per-class insight
        """

        assert average in ["micro", "macro", "weighted"], 'average must be "micro", "macro" or "weighted"'

        self.__average = average

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            logits, probas, targets = self.extract_logits_and_targets(**kwargs)

            predictions = probas.argmax(dim=1).detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            return f1_score(targets, predictions, average=self.__average, zero_division=0.0)

    @override
    def get_name(self) -> str:
        return f"F1 Score ({self.__average})"
