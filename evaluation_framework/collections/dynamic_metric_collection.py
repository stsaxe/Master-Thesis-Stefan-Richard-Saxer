from __future__ import annotations

from typing_extensions import override, Self

from evaluation_framework.collections.static_metric_collection import StaticMetricCollection
from evaluation_framework.interfaces.dynamic_metric_collection_interface import DynamicMetricCollectionInterface
from evaluation_framework.interfaces.metric_interface import MetricInterface


class DynamicMetricCollection(StaticMetricCollection, DynamicMetricCollectionInterface):

    @override
    def add_metric(self, metric: MetricInterface, name: str = None) -> Self:
        self._add_metric(metric, name)
        return self

    @override
    def add_metrics(self, metrics: dict[str, MetricInterface] | list[MetricInterface]) -> Self:
        assert isinstance(metrics, dict) or isinstance(metrics, list), 'metrics must be of type dictionary or type list'
        if isinstance(metrics, dict):
            assert all(
                isinstance(metric, MetricInterface) and isinstance(name, str) for name, metric in
                metrics.items()), 'key must be of type string and value of type MetricInterface'

            for name, metric in metrics.items():
                self.add_metric(metric, name)

        elif isinstance(metrics, list):
            all(isinstance(metric, MetricInterface) for metric in metrics), ('all items in the list must be of type '
                                                                             'MetricInterface')
            for metric in metrics:
                self.add_metric(metric)

        return self

    @override
    def remove_metric(self, name: str) -> Self:
        assert isinstance(name, str), 'name must be a string'
        assert name in [cont.get_name() for cont in self._metric_containers], (f'no metric with name {name} exists in '
                                                                               f'this metric collection')

        for i, container in enumerate(self._metric_containers):
            if container.get_name() == name:
                del self._metric_containers[i]
                break

        return self

    @override
    def remove_metrics(self, metrics: list[str]) -> Self:
        assert isinstance(metrics, list), 'metrics must be of type list'
        assert all(isinstance(name, str) for name in metrics), 'all items in the list must be of type string'

        for name in metrics:
            self.remove_metric(name)

        return self
