from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from abc import ABC
from typing import override

from ble.fields.HexDataField import HexDataField
from ble.utils.StringCursor import StringCursor
from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.utils.HelperMethods import HelperMethods
from ble.components.MultiNodeContainer import MultiNodeContainer


class AbstractServiceUUIDListStruct(AbstractAdvDataStruct, ABC):

    def __init__(self, bit: int) -> None:
        AbstractAdvDataStruct.__init__(self)
        assert isinstance(bit, int), 'Bit is not an integer'
        assert bit in [16, 32, 128], 'Bit value must be 16, 32 or 128'

        self.bit: int = bit
        self.uuids: MultiNodeContainer = MultiNodeContainer(f"{self.bit} bit UUIDs", component_type=HexDataField)

    @override
    def get_length(self, bit: bool = False) -> int:
        length = self.length.get_length(bit=bit) + self.type.get_length(bit=bit)

        for uuid in self.uuids:
            length += uuid.get_length(bit=bit)

        return length

    @override
    def update(self):
        self._set_type()
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 2, byte=True)

        target_length = int(2 + (len(value) // 2 - 2) // (self.bit // 8) * (self.bit // 8))

        policy.hex_string_length(value, target_length, byte=True)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(1), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(1), parse_mode=parse_mode)

        self.uuids.clear()

        while cursor.remaining(bytes=True) >= self.bit // 8:
            uuid = cursor.read_bytes(self.bit // 8)

            field = HexDataField(f"{self.bit} bit UUID", target_byte_length=self.bit // 8)
            field.from_string(HelperMethods.hex_le_to_be(uuid), parse_mode=parse_mode)

            self.uuids.append(field)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        out = self.length.to_string() + self.type.to_string()

        for uuid in self.uuids:
            out += HelperMethods.hex_be_to_le(uuid.to_string(prefix=False))

        return prefix * "0x" + out

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        print("\t" * (ident + 1) + self.uuids.get_name())

        for uuid in self.uuids:
            uuid._print(ident=ident + 2)
