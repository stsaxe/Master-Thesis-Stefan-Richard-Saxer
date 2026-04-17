from abc import abstractmethod, ABC
from typing import Any

from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface


class ActionInterface(ABC):
    @abstractmethod
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        pass

