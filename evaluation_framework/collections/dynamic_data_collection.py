from typing_extensions import override, Self

from evaluation_framework.collections.static_data_collection import StaticDataCollection
from evaluation_framework.interfaces.dynamic_data_collection_interface import DynamicDataCollectionInterface


class DynamicDataCollection(DynamicDataCollectionInterface, StaticDataCollection):

    @override
    def add_data(self, **kwargs) -> Self:
        StaticDataCollection._add_data(self, **kwargs)
        return self

    @override
    def reset(self) -> Self:
        self._data = dict()
        return self
