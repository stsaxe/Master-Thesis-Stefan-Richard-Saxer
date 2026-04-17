from abc import ABC, abstractmethod
from typing_extensions import Self

from evaluation_framework.interfaces.static_data_collection_interface import StaticDataCollectionInterface


class DynamicDataCollectionInterface(StaticDataCollectionInterface, ABC):

    @abstractmethod
    def add_data(self, **kwargs) -> Self:
        pass

    @abstractmethod
    def reset(self) -> Self:
        pass
