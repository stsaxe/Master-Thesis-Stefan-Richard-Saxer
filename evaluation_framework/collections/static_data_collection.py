import copy

import torch

from evaluation_framework.interfaces.static_data_collection_interface import StaticDataCollectionInterface


class StaticDataCollection(StaticDataCollectionInterface):
    def __init__(self, data: dict[str, torch.Tensor] = None):
        self._data = dict()

        if data is not None:
            self._add_data(**data)

    def _add_data(self, **kwargs):
        for name, data in kwargs.items():
            assert isinstance(name, str), 'key of kwargs must be a string'
            assert isinstance(data, torch.Tensor), 'value of kwargs must be a tensor'
            if name in self._data.keys():
                self._data[name] = torch.cat([self._data[name], data], dim=0)
            else:
                self._data[name] = data

    def get_data(self) -> dict[str, torch.Tensor]:
        return copy.deepcopy(self._data)
