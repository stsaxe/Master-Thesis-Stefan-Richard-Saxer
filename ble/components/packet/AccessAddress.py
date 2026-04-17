from __future__ import annotations
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from ble.fields.HexDataField import HexDataField


class AccessAddress(HexDataField):
    def __init__(self) -> None:
        HexDataField.__init__(self, "Access Address", target_byte_length=4)

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = AccessAddress.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_length(value, self.get_target_byte_length(), byte=True)
        value = AccessAddress.hex_le_to_be(value[:self.get_target_byte_length() * 2])

        self.set_value(value)

        policy.verify(self)

    def to_string(self, prefix: bool = False) -> str:
        return "0x" * prefix + AccessAddress.hex_be_to_le(self.value)
