from abc import ABC

import numpy as np
import torch

from evaluation_framework.metrics.abstract_metric import AbstractMetric


class AbstractPrecisionRecallOpenSet(AbstractMetric, ABC):
    def __init__(self, unknown_label: int, precision: int = 2, average: str = 'micro', add_unknowns: bool = False):
        super().__init__(unknown_label=unknown_label, add_unknowns=add_unknowns)
        assert average in ['micro', 'macro', 'weighted', 'binary', 'balanced'], 'invalid average'

        assert isinstance(unknown_label, int), 'unknown label must be of type integer'
        assert unknown_label >= 0, 'unknown_label cannot be negative'
        assert isinstance(precision, int), 'precision must be of type integer'

        assert precision >= 0, 'precision must be at least 0'
        assert precision <= 4, 'precision must be at max 4'

        self.__average = average
        self.__epsilon = 1e-10
        self.__precision = precision
        self.__unknown_label = unknown_label

    def compute_metric(self, predictions: torch.Tensor, targets: torch.Tensor, N: int, C: int) -> torch.Tensor | float:
        # 2) Initialize counts
        TP = torch.zeros(C).float()
        FP = torch.zeros(C).float()
        FN = torch.zeros(C).float()
        support = torch.zeros(C).float()

        TP_known = []
        FP_known = []
        FN_known = []
        support_knowns = []

        TP_unknown = 0
        FP_unknown = 0
        FN_unknown = 0
        support_unknown = 0

        # 3) Accumulate TP, FP, FN, support per class
        for cls in range(C):
            is_true = (targets == cls)
            is_pred = (predictions == cls)

            TP[cls] = (is_true & is_pred).sum().item()
            FP[cls] = (~is_true & is_pred).sum().item()
            FN[cls] = (is_true & ~is_pred).sum().item()
            support[cls] = is_true.sum().item()

            if cls != self.__unknown_label:
                TP_known.append(TP[cls])
                FP_known.append(FP[cls])
                FN_known.append(FN[cls])
                support_knowns.append(support[cls])
            else:
                TP_unknown = TP[cls]
                FP_unknown = FP[cls]
                FN_unknown = FN[cls]
                support_unknown = support[cls]

        TP_known = torch.tensor(TP_known)
        FP_known = torch.tensor(FP_known)
        FN_known = torch.tensor(FN_known)

        # 4) Compute per‐class precision & recall
        precision = TP / (TP + FP + self.__epsilon)
        recall = TP / (TP + FN + self.__epsilon)

        if self.__average == 'macro':
            # 5) Macro‐averages (unweighted)

            macro_precision = float(precision.mean())
            macro_recall = float(recall.mean())

            macro_f1 = (2 * precision * recall) / (self.__epsilon + precision + recall)
            macro_f1 = macro_f1.mean()

            return float(macro_precision), float(macro_recall), float(macro_f1)

        # 6) Micro‐averages (global counts)

        total_TP = TP.sum()
        total_FP = FP.sum()
        total_FN = FN.sum()

        if self.__average == 'micro':
            micro_precision = float(total_TP / (total_TP + total_FP + self.__epsilon))
            micro_recall = float(total_TP / (total_TP + total_FN + self.__epsilon))

            micro_f1 = micro_recall * micro_precision * 2 / (self.__epsilon + micro_recall + micro_precision)

            return float(micro_precision), float(micro_recall), float(micro_f1)

        if self.__average == 'weighted':
            # 7) Weighted‐averages by support
            weights = support / support.sum()
            weighted_precision = float((precision * weights).sum())
            weighted_recall = float((recall * weights).sum())

            f1 = (2 * precision * recall) / (self.__epsilon + precision + recall)
            weighted_f1 = (weights * f1).sum()

            return float(weighted_precision), float(weighted_recall), float(weighted_f1)

        if self.__average == 'balanced':
            # 8) balanced
            precision_knowns = TP_known.sum() / (TP_known.sum() + FP_known.sum() + self.__epsilon)
            recall_knowns = TP_known.sum() / (TP_known.sum() + FN_known.sum() + self.__epsilon)

            precision_unknowns = float(TP_unknown / (TP_unknown + FP_unknown + self.__epsilon))
            recall_unknowns = float(TP_unknown / (TP_unknown + FN_unknown + self.__epsilon))

            precision_balanced = (precision_knowns + precision_unknowns) / 2
            recall_balanced = (recall_knowns + recall_unknowns) / 2

            f1_known = (precision_knowns * recall_knowns * 2) / (self.__epsilon + precision_knowns + recall_knowns)
            f1_unknown = precision_unknowns * recall_unknowns * 2 / (self.__epsilon + precision_unknowns + recall_unknowns)

            f1_balanced = (f1_known + f1_unknown) / 2

            return float(precision_balanced), float(recall_balanced), float(f1_balanced)

        if self.__average == 'binary':
            true_known = (targets != self.__unknown_label)

            pred_known = (predictions != self.__unknown_label)

            # 3) True Positives = model says unknown & it really is unknown
            TP = int((pred_known & true_known).sum().item())
            # 4) False Positives = model says unknown but it's actually known
            FP = int((pred_known & ~true_known).sum().item())
            # 5) False Negatives = model says known but it's actually unknown
            FN = int((~pred_known & true_known).sum().item())

            precision = TP / (TP + FP + self.__epsilon)
            recall = TP / (TP + FN + self.__epsilon)

            f1 = 2 * precision * recall / (self.__epsilon + precision + recall)

            return float(precision), float(recall), float(f1)

    def compute_PR_and_RC(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor | float,  torch.Tensor | float]:
        with torch.no_grad():
            logits, probas, targets = self.extract_logits_and_targets(**kwargs)

            N, C = probas.shape

            assert self.__unknown_label <= C, 'unknown_label must be in range of Number of Classes + 1'

            if self.__unknown_label < C:
                predictions = torch.argmax(logits, dim=1)

                precision, recall, f1_scores = self.compute_metric(predictions, targets, N, C)

            else:
                precision = []
                recall = []
                f1_scores = []

                for threshold in np.linspace(0, 1, num=10 ** self.__precision + 1):
                    max_vals, argmaxes = probas.max(dim=1)

                    # if max confidence below threshold => unknown_label
                    predictions = torch.where(max_vals >= float(threshold), argmaxes,
                                              torch.full_like(argmaxes, self.__unknown_label))

                    p, r, f1 = self.compute_metric(predictions, targets, N, C + 1)

                    precision.append(p)
                    recall.append(r)
                    f1_scores.append(f1)

                precision = torch.tensor(precision)
                recall = torch.tensor(recall)
                f1_scores = torch.tensor(f1_scores)

            return precision, recall, f1_scores
