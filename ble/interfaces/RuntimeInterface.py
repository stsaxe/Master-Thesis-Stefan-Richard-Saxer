from __future__ import annotations

from abc import abstractmethod, ABC

from ble.fields.AbstractField import AbstractField


from typing import override, TYPE_CHECKING


if TYPE_CHECKING:
    from ble.interfaces.NodeInterface import NodeInterface


class RuntimeInterface(ABC):
    @abstractmethod
    def enter_node(self, node: NodeInterface) -> None:
        pass
    @abstractmethod
    def process_fields(self, fields: list[AbstractField]) -> None:
       pass

    @abstractmethod
    def leave_node(self) -> None:
        pass