from __future__ import annotations
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.fields.HexDataField import HexDataField
from ble.utils.StringCursor import StringCursor
from ble.utils.HelperMethods import HelperMethods


class AbstractLocalNameStruct(AbstractAdvDataStruct):
    def __init__(self) -> None:
        AbstractAdvDataStruct.__init__(self)
        self.device_name: HexDataField = HexDataField("Device Name")

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.length.get_length(bit=bit) + self.type.get_length(bit=bit) + self.device_name.get_length(bit=bit)

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

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(1), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(1), parse_mode=parse_mode)
        self.device_name.from_string(cursor.read_to_end(byte_aware=True), parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return "0x" * prefix + self.length.to_string() + self.type.to_string() + self.device_name.to_string()

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)

        name = bytes.fromhex(self.device_name.get_value())
        try:
            print("\t" * (ident + 1) + f"{self.device_name.get_name()}: {name.decode('utf-8')}")
        except UnicodeDecodeError:
            print("\t" * (ident + 1) + f"{self.device_name.get_name()}: {self.device_name.to_string(prefix=True)}")
