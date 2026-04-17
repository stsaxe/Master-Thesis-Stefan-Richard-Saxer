from abc import ABC
from typing import Optional, Literal

PadMode = Optional[Literal["nibble", "byte"]]


class HelperMethods(ABC):
    @staticmethod
    def check_valid_string(value: str, empty_allowed: bool = False) -> str:
        assert isinstance(value, str), 'Value is not a string'
        if not empty_allowed:
            assert len(value) > 0, 'Value is an empty string'

        return value

    @staticmethod
    def bin_to_hex(value: str, pad: PadMode = None) -> str:
        assert pad in ['byte', 'nibble', None], "pad must be None, 'nibble', or None"
        bits = HelperMethods.clean_bin_value(value, empty_allowed=True)

        if pad == 'byte':
            rem = len(bits) % 8
            if rem:
                bits = bits.zfill(len(bits) + (8 - rem))

        out = int(bits, 2)
        hex_digits = max(1, (len(bits) + 3) // 4)  # ceil(bits/4)
        return format(out, f"0{hex_digits}X")

    @staticmethod
    def int_to_hex(value: int, pad: PadMode = None) -> str:
        assert isinstance(value, int), 'Value is not an integer'
        assert value >= 0, 'Value is a negative integer'

        return HelperMethods.bin_to_hex(bin(value), pad=pad)

    @staticmethod
    def int_to_bin(value: int, pad: PadMode = None) -> str:
        assert isinstance(value, int), 'Value is not an integer'
        assert value >= 0, 'Value is a negative integer'
        assert pad in ['byte', 'nibble', None], "pad must be None, 'nibble', or None"


        bits = format(value, "b")

        if pad == "nibble":
            width = ((len(bits) + 3) // 4) * 4
            bits = bits.zfill(width)
        elif pad == "byte":
            width = ((len(bits) + 7) // 8) * 8
            bits = bits.zfill(width)

        return f"{bits}"

    @staticmethod
    def hex_to_bin(value: str, pad: PadMode = None) -> str:
        assert pad in ['byte', 'nibble', None], "pad must be None, 'nibble', or None"
        value = HelperMethods.clean_hex_value(value, empty_allowed=False)

        return HelperMethods.int_to_bin(int(value, 16), pad=pad)

    @staticmethod
    def hex_le_to_be(value: str) -> str:
        value = HelperMethods.clean_hex_value(value, empty_allowed=True)

        assert len(value) % 2 == 0, "hex string must have an even number of hex digits (full bytes)."

        if value == "":
            return ""

        b = bytes.fromhex(value)
        b = b[::-1].hex()

        return HelperMethods.clean_hex_value(b)

    @staticmethod
    def hex_be_to_le(value: str) -> str:
        return HelperMethods.hex_le_to_be(value)

    @staticmethod
    def clean_hex_value(value: str, empty_allowed: bool = False, prefix:bool =False) -> str:
        value = HelperMethods.check_valid_string(value, empty_allowed=empty_allowed)

        special_symbols = ["_", ":", " ", "-"]
        value = value.strip().lower()

        for symbol in special_symbols:
            value = value.replace(symbol, "")

        if value.startswith("0x") or value.startswith("0X"):
            value = value[2:]

        value = value.upper()

        if any(c not in "0123456789ABCDEF" for c in value):
            raise ValueError(f"Invalid hex string: {value}")

        return prefix * "0x" + value

    @staticmethod
    def clean_bin_value(value: str, empty_allowed:bool = False, prefix:bool = False) -> str:
        value = HelperMethods.check_valid_string(value, empty_allowed=empty_allowed)

        if value == "":
            return ""

        special_symbols = ["_", ":", " ", "-"]

        for symbol in special_symbols:
            value = value.replace(symbol, "")

        if value.startswith("0b") or value.startswith("0B"):
            value = value[2:]

        if any(c not in "01" for c in value):
            raise ValueError(f"Invalid binary string: {value}")

        return prefix * "0b" + value
