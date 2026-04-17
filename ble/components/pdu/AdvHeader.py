from __future__ import annotations
from typing import TYPE_CHECKING

from ble.interfaces.PrintInterface import PrintInterface

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from typing import override

from ble.fields.BitDataField import BitDataField
from ble.fields.HexDataField import HexDataField
from ble.utils.HelperMethods import HelperMethods
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.ParserInterface import ParserInterface
from ble.utils.StringCursor import StringCursor


class AdvHeader(ParserInterface, NodeInterface, HelperMethods, PrintInterface):
    def __init__(self) -> None:
        NodeInterface.__init__(self, "Packet Header")
        self.pdu_type = BitDataField("PDU Type", target_bit_length=4)
        self.rfu = BitDataField("RFU", target_bit_length=1)
        self.ch_sel = BitDataField("ChSel", target_bit_length=1)
        self.tx_add = BitDataField("TxAdd", target_bit_length=1)
        self.rx_add = BitDataField("RxAdd", target_bit_length=1)
        self.length = HexDataField("Length", target_byte_length=1)

    @override
    def get_length(self, bit: bool = False) -> int:
        length = (self.pdu_type.get_bit_length() +
                  self.rfu.get_bit_length() +
                  self.ch_sel.get_bit_length() +
                  self.tx_add.get_bit_length() +
                  self.rx_add.get_bit_length() + self.length.get_length(bit=True))

        if bit:
            return length
        else:
            return length // 8

    @override
    def update(self) -> None:
        for field in self.get_fields():
            if field.get_name() == "RFU" and isinstance(field, BitDataField) and (field is not self.pdu_type):
                field.set_value("0", bin=True)

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_length(value, 2, byte=True)

        cursor = StringCursor(value)

        bit_values = HelperMethods.hex_to_bin(cursor.read_bytes(1), pad='byte')

        self.pdu_type.set_value(bit_values[4:8], bin=True)
        self.rx_add.set_value(bit_values[0], bin=True)
        self.tx_add.set_value(bit_values[1], bin=True)
        self.ch_sel.set_value(bit_values[2], bin=True)
        self.rfu.set_value(bit_values[3], bin=True)

        self.length.from_string(cursor.read_bytes(self.length.get_target_byte_length()), parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        bit_values = self.rx_add.get_value(bin=True) + self.tx_add.get_value(bin=True) + self.ch_sel.get_value(
            bin=True) + self.rfu.get_value(bin=True) + self.pdu_type.get_value(bin=True)
        return prefix * "0x" + HelperMethods.bin_to_hex(bit_values, pad="byte") + self.length.get_value()

    @override
    def _print(self, ident: int = 0) -> None:
        print('\t' * ident + self.get_name())
        self.pdu_type._print(ident=ident + 1)
        self.rfu._print(ident=ident + 1)
        self.ch_sel._print(ident=ident + 1)
        self.tx_add._print(ident=ident + 1)
        self.rx_add._print(ident=ident + 1)
        self.length._print(ident=ident + 1)
