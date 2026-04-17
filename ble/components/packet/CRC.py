from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface


from typing import override

from ble.fields.HexDataField import HexDataField
from ble.utils.HelperMethods import HelperMethods


class CRC(HexDataField):
    def __init__(self) -> None:
        HexDataField.__init__(self, "CRC", target_byte_length=3)

    @override
    def _print(self, ident: int = 0) -> None:
        print('\t' * ident + f"{self.get_name()}: 0x{HelperMethods.clean_hex_value(self._print_format(), empty_allowed=True)}")

    @staticmethod
    def from_pdu(pdu_hex: str) -> str:
        init_hex: str = "555555"

        def swapbits(a: int) -> int:
            v = 0
            if a & 0x80: v |= 0x01
            if a & 0x40: v |= 0x02
            if a & 0x20: v |= 0x04
            if a & 0x10: v |= 0x08
            if a & 0x08: v |= 0x10
            if a & 0x04: v |= 0x20
            if a & 0x02: v |= 0x40
            if a & 0x01: v |= 0x80
            return v

        pdu_hex = HelperMethods.clean_hex_value(pdu_hex, empty_allowed=True)
        pdu_hex = pdu_hex.strip()
        init_hex = init_hex.strip()

        if len(pdu_hex) % 2 != 0:
            raise ValueError("pdu_hex must have an even number of hex characters")
        if len(init_hex) != 6:
            raise ValueError("init_hex must be exactly 6 hex characters")

        pdu = bytes.fromhex(pdu_hex)
        init = int(init_hex, 16)

        state = (
                swapbits(init & 0xFF)
                | (swapbits((init >> 8) & 0xFF) << 8)
                | (swapbits((init >> 16) & 0xFF) << 16)
        )

        lfsr_mask = 0x5A6000

        for byte in pdu:
            cur = byte
            for _ in range(8):
                next_bit = (state ^ cur) & 1
                cur >>= 1
                state >>= 1
                if next_bit:
                    state |= 1 << 23
                    state ^= lfsr_mask

        crc_bytes = struct.pack("<L", state)[:3]
        return crc_bytes.hex().upper()


    def _bitrev8(self, x: int) -> int:
        x &= 0xFF
        x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)
        x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2)
        x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1)
        return x

    def _print_format(self) -> str:
        b = bytes.fromhex(self.get_value())
        return bytes(self._bitrev8(x) for x in b).hex()




