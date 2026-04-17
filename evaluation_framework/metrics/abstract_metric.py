from abc import ABC

import torch

from evaluation_framework.interfaces.metric_interface import MetricInterface


class AbstractMetric(MetricInterface, ABC):
    def __init__(self, unknown_label: int = None, add_unknowns: bool = False):
        assert isinstance(unknown_label, int) or unknown_label is None, 'unknown_label must be an integer'
        assert isinstance(add_unknowns, bool), 'add_unknowns must be boolean'
        if isinstance(unknown_label, int):
            assert unknown_label >= 0, 'unknown_label cannot be negative'

        if add_unknowns:
            assert not unknown_label is None, 'unknown_label cannot be None when add_unknowns is True'

        self._abstract_unknown_label = unknown_label
        self._add_unknowns = add_unknowns

    def extract_logits(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            assert 'logits' in kwargs.keys(), 'logits is no key in kwargs'

            logits = kwargs['logits']

            assert isinstance(logits, torch.Tensor), 'logits must be of type tensor'

            assert len(logits.shape) == 2, 'number of axis of logits must be 2'
            assert logits.shape[1] >= 2, 'length of axis 1 of logits must be at least 2'
            assert logits.shape[0] >= 1, 'length of axis 0 of logits must be at least 1; logits cannot be empty'

            probas = torch.nn.functional.softmax(logits, dim=1)

            if self._add_unknowns and probas.shape[1] - 1 >= self._abstract_unknown_label:
                probas = torch.cat([probas[:, :self._abstract_unknown_label], probas[:, self._abstract_unknown_label:].sum(dim=1).unsqueeze(-1)], dim=1)
            return logits, probas

    def extract_targets(self, **kwargs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            assert 'targets' in kwargs.keys(), 'targets is no key in kwargs'

            targets = kwargs['targets']

            assert isinstance(targets, torch.Tensor), 'targets must eb of type tensor'
            assert len(targets.shape) == 1, 'number of axis of targets must be 1'
            assert targets.shape[0] >= 1, 'length of axis 0 of targets must be at least 1; targets cannot be empty'

            return targets

    def extract_logits_and_targets(self, **kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, probas = self.extract_logits(**kwargs)
        targets = self.extract_targets(**kwargs)

        assert logits.shape[0] == len(targets), 'length of axis 1 of logits must match length of axis 0 of targets'

        return logits, probas, targets
