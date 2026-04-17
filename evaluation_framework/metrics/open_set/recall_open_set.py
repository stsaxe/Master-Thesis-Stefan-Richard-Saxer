import torch

from evaluation_framework.metrics.open_set.abstract_precision_recall import AbstractPrecisionRecallOpenSet


class RecallOpenSet(AbstractPrecisionRecallOpenSet):

    def __init__(self, unknown_label: int, precision: int = 2, average: str = 'micro', add_unknowns: bool = False):
        super().__init__(unknown_label, precision, average, add_unknowns=add_unknowns)
        self.__avg = average

    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        precision, recall, f1 = self.compute_PR_and_RC(**kwargs)

        return recall

    def get_name(self) -> str:
        return f"Recall Open Set ({self.__avg})"
