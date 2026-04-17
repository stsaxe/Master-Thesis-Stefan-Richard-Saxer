from abc import ABC

from typing_extensions import override

from evaluation_framework.interfaces.experiment_interface import ExperimentInterface
from evaluation_framework.scoring.score import Score

import time

class AbstractExperiment(ExperimentInterface, ABC):
    def __init__(self, name: str = None):

        if name is not None:
            assert isinstance(name, str), 'name must be of type str'
            self._name = name
        else:
            self._name = ''

    @override
    def get_name(self) -> str:
        return self._name

    @override
    def score(self) -> Score:
        score = dict()

        for metric in self.get_metrics():
            print(f"{metric.get_name()} started computing...")
            start_time = time.time()
            result = metric.compute(**self.get_data())
            compute_time = time.time()-start_time
            print(f"{metric.get_name()} finished computing in {compute_time/60:.2f} minutes.")
            score[metric.get_name()] = result

        return Score(score)
