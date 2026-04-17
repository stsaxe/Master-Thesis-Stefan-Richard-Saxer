import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class AUROC(AbstractMetric):
    def __init__(self, average: str = "macro"):
        super().__init__()
        assert average in ["macro", "weighted"], 'average must be "macro" or "weighted"'

        self.__average = average

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = self.extract_logits_and_targets(**kwargs)
            preds = torch.argmax(probas, dim=1)

            # 1. Move data to NumPy
            y_scores = probas.detach().cpu().numpy()  # shape (N, C)
            y_true = targets.detach().cpu().numpy()  # shape (N,)
            y_true = y_true.astype(dtype=int)
            N, C = y_scores.shape

            # 2. One-hot encode targets: shape (N, C)
            y_onehot = np.eye(C, dtype=int)[y_true]

            # 3. Per-class AUROC
            per_class = []
            support = y_onehot.sum(axis=0).astype(float)  # support[i] = # true samples of class i


            for i in range(C):
                score= roc_auc_score(y_onehot[:, i], y_scores[:, i])

                # Area under ROC curve via trapezoidal rule
                per_class.append(score)

            per_class = np.array(per_class)  # shape (C,)

            # 4. Macro-average (simple mean)
            macro = float(per_class.mean())

            # 5. Weighted-average by support
            weights = support / support.sum()
            weighted = float((per_class * weights).sum())

        if self.__average == "weighted":
            return weighted
        elif self.__average == "macro":
            return macro

    @override
    def get_name(self) -> str:
        return f"AUROC ({self.__average})"
