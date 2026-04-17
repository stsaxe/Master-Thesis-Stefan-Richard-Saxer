
from __future__ import annotations
from typing import TYPE_CHECKING

from typing_extensions import override

from ble.utils.StringCursor import StringCursor
from ble.fields.AbstractField import AbstractField
from ble.utils.HelperMethods import HelperMethods
from ble.interfaces.ParserInterface import ParserInterface

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

class HexDataField(AbstractField, ParserInterface, HelperMethods):
    def __init__(self, name: str, value: str = None, bin: bool = False, target_bit_length: int = None, target_byte_length: int = None) -> None:
        self.name: str = ""
        self.set_name(name)
        self.value: str = ""

        self.target_bit_length: int = None

        if (target_bit_length is not None) and (target_byte_length is not None):
            assert target_bit_length == target_byte_length * 8, "Target bit length must match target byte length if both are set"

        if target_bit_length is not None:
            self._set_target_bit_length(target_bit_length)
        elif target_byte_length is not None:
            self._set_target_bit_length(target_byte_length * 8)

        if not value is None:
            self.set_value(value, bin)



    def _set_target_bit_length(self, target_bit_length: int) -> None:
        if not target_bit_length is None:
            assert isinstance(target_bit_length, int) and target_bit_length >= 0, "Target bit length must be positive integer"
            self.target_bit_length = target_bit_length

    @override
    def get_target_bit_length(self) -> int | None:
        return self.target_bit_length

    def get_target_byte_length(self) -> int | None:
        return self.target_bit_length // 8


    @override
    def get_value(self, bin: bool = False, prefix: bool = False) -> str:
        if self.value == "":
            if bin:
                return prefix * "0b" + ""
            else:
                return prefix * "0x" + ""

        if bin:
            return prefix * "0b" + HexDataField.hex_to_bin(self.value, pad='byte')
        else:
            return prefix * "0x" + self.value

    @override
    def get_bit_length(self) -> int:
        return len(self.value) * 4


    @override
    def set_value(self, value: str, bin: bool = False) -> None:
        if bin:
            value = HexDataField.bin_to_hex(value, pad="byte")

        value = HexDataField.clean_hex_value(value, empty_allowed=True)

        assert len(value) % 2 == 0, "DataField value must be of even length (no partial Bytes allowed)"

        if not self.get_target_bit_length() is None:
            assert len(value) * 4 == self.get_target_bit_length(), "Length of Value does not match Target bit length"

        self.value = value

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def set_name(self, name: str) -> None:
        self.check_valid_string(name)
        self.name = name

    @override
    def get_length(self, bit: bool = False) -> int:
        if bit:
            return self.get_bit_length()
        else:
            return len(self.value) // 2

    @override
    def update(self):
        pass

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value, empty_allowed=True)
        policy = self.get_parse_policy(parse_mode)

        policy.hex_string_is_in_byte_format(value, empty_allowed=True)

        cursor = StringCursor(value)

        self.set_value(cursor.read_to_end(byte_aware=True), bin=False)



    @override
    def to_string(self, prefix: bool = False) -> str:
        return prefix * "0x" + self.value

    @override
    def _print(self, ident: int = 0) -> None:
        print('\t' * ident + f"{self.get_name()}: 0x{self.get_value()}")

