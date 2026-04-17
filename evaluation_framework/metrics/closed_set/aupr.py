import numpy as np
import torch
from sklearn.metrics import average_precision_score
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class AUPR(AbstractMetric):
    def __init__(self, average: str = "macro"):
        super().__init__()
        assert average in ["macro", "weighted"], 'average must be "micro", "macro" or "weighted"'

        self.__average = average

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = self.extract_logits_and_targets(**kwargs)

        # 1. To NumPy
        y_scores = probas.detach().cpu().numpy()
        y_true = targets.detach().cpu().numpy()
        y_true = y_true.astype(dtype=int)
        N, C = y_scores.shape

        # 2. One-hot encode
        y_onehot = np.eye(C, dtype=int)[y_true]

        # 3. Per-class AUPR via trapezoidal rule
        per_class = []
        for i in range(C):
            score = average_precision_score(y_onehot[:, i], y_scores[:, i])

            per_class.append(score)

        # 4. Macro-average
        macro = float(np.mean(per_class))

        # 5) Weighted-average by class support
        support = y_onehot.sum(axis=0).astype(float)  # shape (C,)
        support_fraction = support / support.sum()  # sum to 1
        weighted = float((per_class * support_fraction).sum())

        if self.__average == "weighted":
            return weighted
        elif self.__average == "macro":
            return macro

    @override
    def get_name(self) -> str:
        return f"AUPR ({self.__average})"
