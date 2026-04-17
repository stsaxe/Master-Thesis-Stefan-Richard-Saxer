from __future__ import annotations

import torch

from evaluation_framework.collections.static_data_collection import StaticDataCollection
from evaluation_framework.collections.static_metric_collection import StaticMetricCollection
from evaluation_framework.experiments.abstract_experiment import AbstractExperiment
from evaluation_framework.interfaces.metric_interface import MetricInterface


class StaticExperiment(StaticMetricCollection, StaticDataCollection, AbstractExperiment):
    def __init__(self, name: str, metrics: dict[str, MetricInterface], data: dict[str, torch.Tensor]):
        StaticMetricCollection.__init__(self, metrics, force_compute=False)
        StaticDataCollection.__init__(self, data)
        AbstractExperiment.__init__(self, name)

