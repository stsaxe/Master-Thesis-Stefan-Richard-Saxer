from abc import ABC
from abc import abstractmethod
from ble.interfaces.PrintInterface import PrintInterface


class AbstractField(PrintInterface, ABC):

    @abstractmethod
    def get_value(self, bin: bool = False, prefix: bool = False) -> str:
        pass

    @abstractmethod
    def set_value(self, value: str, bin: bool = False) -> None:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def set_name(self, name: str) -> None:
        pass

    @abstractmethod
    def get_bit_length(self) -> int:
        pass

    @abstractmethod
    def get_target_bit_length(self) -> int | None:
        pass
