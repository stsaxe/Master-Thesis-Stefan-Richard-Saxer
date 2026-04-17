from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from abc import ABC
from typing import override

from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.components.pdu.AbstractAdvPDU import AbstractAdvPDU
from ble.components.MultiNodeContainer import MultiNodeContainer
from ble.utils.HelperMethods import HelperMethods
from ble.utils.StringCursor import StringCursor


class AbstractAdvDataPDU(AbstractAdvPDU, ABC):
    def __init__(self) -> None:
        AbstractAdvPDU.__init__(self)
        self.adv_data: MultiNodeContainer = MultiNodeContainer("Advertising Data", component_type=AbstractAdvDataStruct)

    @override
    def get_length(self, bit: bool = False) -> int:
        length = self.header.get_length(bit=bit) + self.advertising_address.get_length(bit=bit)

        for data in self.adv_data:
            length += data.get_length(bit=bit)

        return length

    @override
    def update(self) -> None:
        self.advertising_address.update()

        for data in self.adv_data:
            data.update()

        self._update_header()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 2 + self.advertising_address.get_target_byte_length(), byte=True)

        cursor = StringCursor(value)
        header_string = cursor.read_bytes(2)

        self.header.from_string(header_string, parse_mode=parse_mode)

        advertising_address = HelperMethods.hex_le_to_be(cursor.read_bytes(6))
        self.advertising_address.from_string(advertising_address, parse_mode=parse_mode)

        self.adv_data.clear()

        while cursor.remaining(bytes=True) >= 2:
            length = cursor.peek_bytes(n=1, shift=0)
            length = HelperMethods.clean_hex_value(length)
            length = int(length, 16)

            adv_type = cursor.peek_bytes(n=1, shift=1)
            adv_type = HelperMethods.clean_hex_value(adv_type)

            adv_struct = policy.extract_adv_struct(cursor, adv_type, length)
            self.adv_data.append(adv_struct)

        policy.hex_string_length(cursor.read_to_end(), 0)
        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        out = self.header.to_string() + HelperMethods.hex_le_to_be(self.advertising_address.to_string())

        for data in self.adv_data:
            out += data.to_string()

        return "0x" * prefix + out

    @override
    def _print(self, ident: int = 0) -> None:
        print("\t" * ident + self.get_name() + ": " + self._get_registry_name())
        self.header._print(ident=ident + 1)
        self.advertising_address._print(ident=ident + 1)
        print("\t" * (ident + 1) + self.adv_data.get_name())
        for data in self.adv_data:
            data._print(ident=ident + 2)
