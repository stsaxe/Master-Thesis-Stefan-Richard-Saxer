from abc import ABC, abstractmethod

from evaluation_framework.metrics.container.metric_container import MetricContainer


class StaticMetricCollectionInterface(ABC):
    @abstractmethod
    def get_metrics(self) -> list[MetricContainer]:
        pass

