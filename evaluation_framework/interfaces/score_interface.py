from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation_framework.scoring.score import Score


class ScoreInterface(ABC):
    @abstractmethod
    def score(self) -> Score:
        pass
