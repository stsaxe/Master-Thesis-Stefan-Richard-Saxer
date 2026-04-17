from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from ble.interfaces.PrintInterface import PrintInterface

from ble.parse_policy.ParseRegistry import PARSE_POLICY_REGISTRY


class ParserInterface(ABC):
    @abstractmethod
    def get_length(self, bit: bool = False) -> int:
        pass

    @abstractmethod
    def update(self) -> None:
        pass

    @abstractmethod
    def from_string(self, value: str, parse_mode: str  | ParsePolicyInterface = "normal") -> None:
        pass

    @staticmethod
    def get_parse_policy(parse_mode: str | ParsePolicyInterface) -> ParsePolicyInterface:
        if isinstance(parse_mode, str):
            return PARSE_POLICY_REGISTRY[parse_mode]

        else:
            return parse_mode


    @abstractmethod
    def to_string(self, prefix: bool = False) -> str:
        pass









