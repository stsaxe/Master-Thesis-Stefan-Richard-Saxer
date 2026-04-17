from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from typing import override

from ble.utils.StringCursor import StringCursor
from ble.fields.HexDataField import HexDataField
from ble.components.pdu.AbstractAdvPDU import AbstractAdvPDU
from ble.utils.HelperMethods import HelperMethods


class AbstractAdvTargAddr(AbstractAdvPDU):
    def __init__(self) -> None:
        AbstractAdvPDU.__init__(self)
        self.target_address: HexDataField = HexDataField("Target Address", target_byte_length=6)

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.header.get_length(bit=bit) + self.advertising_address.get_length(
            bit=bit) + self.target_address.get_length(bit=bit)

    @override
    def update(self) -> None:
        self.target_address.update()
        self.advertising_address.update()
        self._update_header()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        cursor = StringCursor(value)

        policy.hex_string_length(value, 2 + self.target_address.get_target_byte_length() + self.advertising_address.get_target_byte_length(), byte=True)

        header_string = cursor.read_bytes(2)
        self.header.from_string(header_string, parse_mode=policy)

        advertising_address = HelperMethods.hex_le_to_be(cursor.read_bytes(self.advertising_address.get_target_byte_length()))
        self.advertising_address.from_string(advertising_address, parse_mode=policy)

        target_address = HelperMethods.hex_le_to_be(cursor.read_bytes(self.advertising_address.get_target_byte_length()))
        self.target_address.from_string(target_address, parse_mode=policy)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return "0x" * prefix + self.header.to_string() + HelperMethods.hex_be_to_le(
            self.advertising_address.to_string()) + HelperMethods.hex_be_to_le(self.target_address.to_string())


    @override
    def _print(self, ident: int = 0) -> None:
        print("\t" * ident + self.get_name() + ": " + self._get_registry_name())
        self.header._print(ident=ident + 1)
        self.advertising_address._print(ident=ident + 1)
        self.target_address._print(ident=ident + 1)
