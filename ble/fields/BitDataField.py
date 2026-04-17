from typing_extensions import override

from ble.fields.AbstractField import AbstractField
from ble.utils.HelperMethods import HelperMethods

class BitDataField(AbstractField, HelperMethods):

    def __init__(self, name: str, value: str = None, bin: bool = False, target_bit_length: int = None) -> None:
        self.name: str = ""
        self.set_name(name)
        self.value: str = ""

        self.target_bit_length: int = None
        self._set_target_bit_length(target_bit_length)

        if not value is None:
            self.set_value(value, bin)


    def _set_target_bit_length(self, target_bit_length: int) -> None:
        if not target_bit_length is None:
            assert isinstance(target_bit_length, int) and target_bit_length >= 0, "Target bit length must be positive integer"
            self.target_bit_length = target_bit_length


    def get_target_bit_length(self) -> int | None:
        return self.target_bit_length


    @override
    def get_value(self, bin: bool = False, prefix: bool = False) -> str:
        if self.value == "":
            if bin:
                return "0b" * prefix + ""
            else:
                return "0x" * prefix + ""

        if bin:
            return prefix * "0b" + self.value
        else:
            return prefix * "0x" + HelperMethods.bin_to_hex(self.value, pad=None)

    @override
    def set_value(self, value: str, bin: bool = False) -> None:
        assert isinstance(value, str), "value must be a string"

        if len(value) == 0:
            value = ""

        elif bin:
            value = HelperMethods.clean_bin_value(value, empty_allowed=False)
        else:
            value = HelperMethods.hex_to_bin(value, pad=None)

        if not self.get_target_bit_length() is None:
            assert len(value) == self.get_target_bit_length(), f"Length of Value must be equal to Target bit length, {value}, {self.get_target_bit_length()}"


        self.value = value

    @override
    def get_bit_length(self) -> int:
        return len(self.value)

    @override
    def get_name(self) -> str:
        return self.name

    @override
    def set_name(self, name: str) -> None:
        self.check_valid_string(name)
        self.name = name

    @override
    def _print(self, ident: int = 0):
        print('\t' * ident + self.get_name() + ": 0b" + self.value)

