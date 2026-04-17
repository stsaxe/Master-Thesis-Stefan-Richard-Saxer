from abc import ABC, abstractmethod
from ble.interfaces.NodeInterface import NodeInterface


class ConditionInterface(ABC):
    @abstractmethod
    def apply(self, ctx: NodeInterface) -> bool:
        pass

