from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from abc import ABC
from typing import override
from ble.utils.StringCursor import StringCursor
from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.fields.HexDataField import HexDataField
from ble.utils.HelperMethods import HelperMethods


class AbstractServiceDataStruct(AbstractAdvDataStruct, ABC):
    def __init__(self, bit:int) -> None:
        AbstractAdvDataStruct.__init__(self)
        assert isinstance(bit, int), 'Bit is not an integer'
        assert bit in [16, 32, 128], 'Bit value must be 16, 32 or 128'

        self.bit: int = bit
        self.uuid: HexDataField = HexDataField(f"{bit} bit UUID", target_byte_length=self.bit // 8 )
        self.data: HexDataField = HexDataField("Data")

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.data.get_length(bit=bit) + self.uuid.get_length(bit=bit) + self.length.get_length(bit=bit) + self.type.get_length(bit=bit)

    @override
    def update(self):
        self._set_type()
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 4 + self.bit // 4, byte=False)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(1), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(1), parse_mode=parse_mode)
        self.uuid.from_string(HelperMethods.hex_le_to_be(cursor.read_bytes(self.bit // 8)), parse_mode=parse_mode)
        self.data.from_string(cursor.read_to_end(byte_aware=True), parse_mode=parse_mode)

        policy.verify(self)


    @override
    def to_string(self, prefix: bool = False) -> str:
        return "0x" * prefix + self.length.to_string() + self.type.to_string() + HelperMethods.hex_be_to_le(self.uuid.to_string()) + self.data.to_string()

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        self.uuid._print(ident=ident + 1)
        self.data._print(ident=ident + 1)
