import torch
from typing_extensions import override

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class TopKAccuracy(AbstractMetric):
    def __init__(self, topK: int = 5, average: str = 'micro'):
        super().__init__()
        assert isinstance(topK, int), 'topK must be of type integer'
        assert topK >= 1, 'topK must be at least 1'
        assert average in ['micro', 'macro', 'weighted'], 'invalid average'

        self.__topK = topK
        self.__average = average
        self.__epsilon = 1e-8

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        with torch.no_grad():
            _, probas, targets = super().extract_logits_and_targets(**kwargs)

            if self.__topK >= probas.shape[1]:
                return 1.0

            N, C = probas.shape

            # 1) Get top K predictions (shape: N x K)
            topk_vals, topk_inds = probas.topk(self.__topK, dim=1, largest=True, sorted=False)

            # 2) For each sample, check if true label is among its top K preds
            #    hit_mask: BooleanTensor of shape (N,)
            hit_mask = (topk_inds == targets.unsqueeze(1)).any(dim=1)

            # 3) Micro Top-K accuracy = overall hit rate
            micro_topk = hit_mask.float().sum().item() / N

            # 4) Per-class counts for support and correct Top-K hits
            support_per_class = torch.zeros(C, dtype=torch.int32)
            correct_per_class = torch.zeros(C, dtype=torch.int32)

            for cls in range(C):
                cls_mask = targets == cls
                support = cls_mask.sum().item()
                support_per_class[cls] = support

                if support > 0:
                    # among samples of this class, how many got Top-K hit
                    correct = hit_mask[cls_mask].sum().item()
                    correct_per_class[cls] = correct

            # 5) Per-class Top-K accuracy
            acc_per_class = correct_per_class.float() / (support_per_class.float() + self.__epsilon)

            # 6) Macro Top-K = unweighted mean of per-class
            macro_topk = acc_per_class.mean().item()

            # 7) Weighted Top-K = support-weighted mean
            weighted_topk = (acc_per_class * (support_per_class.float() / N)).sum().item()

            if self.__average == 'weighted':
                return float(weighted_topk)
            elif self.__average == 'micro':
                return float(micro_topk)
            else:
                return float(macro_topk)

    @override
    def get_name(self) -> str:
        if self.__average == 'micro':
            return f"Top-{self.__topK} Accuracy"
        else:
            return f"Top-{self.__topK} Accuracy ({self.__average})"
