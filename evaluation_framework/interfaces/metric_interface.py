from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class MetricInterface(ABC):

    @abstractmethod
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
