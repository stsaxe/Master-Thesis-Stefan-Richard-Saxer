import copy
from typing_extensions import Self

from evaluation_framework.interfaces.metric_interface import MetricInterface
from evaluation_framework.interfaces.static_metric_collection_interface import StaticMetricCollectionInterface
from evaluation_framework.metrics.container.metric_container import MetricContainer


class StaticMetricCollection(StaticMetricCollectionInterface):
    def __init__(self, metrics: dict[str, MetricInterface] | list[MetricInterface] = None, force_compute=False):
        self._metric_containers = []
        self._force_compute = force_compute

        if metrics is not None and isinstance(metrics, dict):
            for name, metric in metrics.items():
                self._add_metric(metric, name)

        elif metrics is not None and isinstance(metrics, list):
            for metric in metrics:
                assert isinstance(metric, MetricInterface), 'all list items must be an instance of MetricInterface'
                self._add_metric(metric, metric.get_name())

    def _add_metric(self, metric: MetricInterface, name: str = None):
        assert isinstance(metric, MetricInterface), 'metric must be of type MetricInterface'

        if name is None:
            name = metric.get_name()
        else:
            assert isinstance(name, str), 'name is not of type str'

        for container in self._metric_containers:
            assert container.get_name() != name, (f'a metric with name {name} already exists in '
                                                  f'this metric collection')

        self._metric_containers.append(MetricContainer(metric, name, force_compute=self._force_compute))

    def get_metrics(self) -> list[MetricContainer]:
        return copy.deepcopy(self._metric_containers)

    def reset(self) -> Self:
        for container in self._metric_containers:
            container.reset_score()

        return self
