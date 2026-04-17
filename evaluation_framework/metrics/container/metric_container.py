import copy

import torch
from typing_extensions import override, Self

from evaluation_framework.interfaces.metric_interface import MetricInterface


class MetricContainer(MetricInterface):
    def __init__(self, metric: MetricInterface, name: str = None, force_compute: bool = False):
        assert isinstance(metric, MetricInterface)
        assert isinstance(force_compute, bool)

        if name is None:
            self.__name = metric.get_name()
        else:
            assert isinstance(name, str), 'name must be of type str'
            self.__name = name
        self.__metric = metric
        self.__score = None
        self.__force_compute = force_compute

    def is_force_compute(self) -> bool:
        return self.__force_compute

    @override
    def get_name(self) -> str:
        return self.__name

    def get_metric(self) -> MetricInterface:
        return copy.deepcopy(self.__metric)

    @override
    def compute(self, **kwargs: torch.Tensor) -> torch.Tensor | float:
        if self.__score is None or self.__force_compute:
            self.__score = self.__metric.compute(**kwargs)
        return self.__score

    def reset_score(self) -> Self:
        self.__score = None
        return self
