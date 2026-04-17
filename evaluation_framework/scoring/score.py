from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from typing_extensions import ItemsView


class Score(Sequence):
    def __init__(self, scores: dict[str,  torch.Tensor | float | Score]):
        self.__scores: dict[str,  torch.Tensor | float | Score] = dict()
        self.__add_scores(scores)

    def __add_score(self, name: str, score: torch.Tensor | float | Score):
        assert isinstance(name, str), 'name must be of type str'
        assert isinstance(score, torch.Tensor) or isinstance(score, float) or isinstance(score, Score), ('score must '
                                                                                                         'be of type '
                                                                                                         'tensor, '
                                                                                                         'float or '
                                                                                                         'score')
        assert name not in self.__scores.keys()

        self.__scores[name] = score

    def __add_scores(self, scores: dict[str,  torch.Tensor | float | Score]):
        assert isinstance(scores, dict), 'scores must be of type dict'

        for name, score in scores.items():
            self.__add_score(name, score)

    def items(self) -> ItemsView[str, torch.Tensor | float] or ItemsView[str, Score]:
        return copy.deepcopy(self.__scores).items()

    def __getitem__(self, key: str | list[str]) -> torch.Tensor | float | Score | dict[
                                                                                  str, torch.Tensor | float | Score]:
        assert isinstance(key, str) or (isinstance(key, list) and all(isinstance(val, str) for val in key)), ('key '
                                                                                                              'must '
                                                                                                              'be of '
                                                                                                              'type '
                                                                                                              'str or '
                                                                                                              'list')

        if isinstance(key, str):
            assert key in self.__scores.keys(), 'key is not in scores'
            return self.__scores[key]

        elif isinstance(key, list):
            new_scores = dict()

            for name in key:
                assert isinstance(name, str), 'all items in the list must be of type str'
                assert name in self.__scores.keys(), 'key is not in scores'
                new_scores[name] = self.__scores[name]

            return new_scores

    def __len__(self):
        return len(self.__scores)
