from __future__ import annotations

from typing_extensions import override, Self

from evaluation_framework.collections.dynamic_data_collection import DynamicDataCollection
from evaluation_framework.collections.dynamic_metric_collection import DynamicMetricCollection
from evaluation_framework.collections.static_metric_collection import StaticMetricCollection
from evaluation_framework.experiments.abstract_experiment import AbstractExperiment
from evaluation_framework.experiments.static_experiment import StaticExperiment


class Experiment(DynamicMetricCollection, DynamicDataCollection, AbstractExperiment):
    def __init__(self, name: str):
        DynamicMetricCollection.__init__(self, metrics=None, force_compute=False)
        DynamicDataCollection.__init__(self, data=None)
        AbstractExperiment.__init__(self, name=name)

    def configure(self, metrics: StaticMetricCollection) -> Self:
        for container in metrics.get_metrics():
            DynamicMetricCollection.add_metric(self, container.get_metric(), container.get_name())

        return self

    @override
    def reset(self) -> Self:
        DynamicDataCollection.reset(self)
        DynamicMetricCollection.reset(self)

        return self

    @override
    def add_data(self, **kwargs) -> Self:
        DynamicDataCollection.add_data(self, **kwargs)
        DynamicMetricCollection.reset(self)

        return self

    def __getitem__(self, key: str | list[str]) -> StaticExperiment:
        assert isinstance(key, str) or (isinstance(key, list) and all(isinstance(val, str) for val in key))

        container_names = [cont.get_name() for cont in self._metric_containers]
        if isinstance(key, str):
            key = [key]

        for cont in key:
            assert cont in container_names

        containers = []

        for cont in self._metric_containers:
            if cont.get_name() in key:
                containers.append(cont)

        return StaticExperiment(self._name, {c.get_name(): c.get_metric() for c in containers}, self.get_data())
