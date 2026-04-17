from __future__ import annotations

from abc import abstractmethod
from typing_extensions import Self

from evaluation_framework.interfaces.metric_interface import MetricInterface
from evaluation_framework.interfaces.static_metric_collection_interface import StaticMetricCollectionInterface


class DynamicMetricCollectionInterface(StaticMetricCollectionInterface):
    @abstractmethod
    def add_metric(self, metric: MetricInterface, name: str = None) -> Self:
        pass

    @abstractmethod
    def add_metrics(self, metrics: dict[str:MetricInterface] | list[MetricInterface]) -> Self:
        pass

    @abstractmethod
    def remove_metric(self, name: str) -> Self:
        pass

    @abstractmethod
    def remove_metrics(self, metrics: list[str]) -> Self:
        pass
