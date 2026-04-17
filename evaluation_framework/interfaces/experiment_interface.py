from __future__ import annotations

from abc import ABC, abstractmethod

from evaluation_framework.interfaces.score_interface import ScoreInterface
from evaluation_framework.interfaces.static_data_collection_interface import StaticDataCollectionInterface
from evaluation_framework.interfaces.static_metric_collection_interface import StaticMetricCollectionInterface


class ExperimentInterface(ScoreInterface,
                          StaticMetricCollectionInterface,
                          StaticDataCollectionInterface,
                          ABC):
    @abstractmethod
    def get_name(self):
        pass
