from __future__ import annotations
from datetime import datetime
from typing import override, TYPE_CHECKING, Self
import datetime

from ble.fields.HexDataField import HexDataField

if TYPE_CHECKING:
    from ble.generation.GenConfig import GenConfig

from ble.interfaces.PrintInterface import PrintInterface
from ble.masking.MaskConfig import MaskConfig

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from ble.components.packet.AccessAddress import AccessAddress
from ble.components.packet.CRC import CRC
from ble.components.pdu.AbstractPDU import AbstractPDU
from ble.components.pdu.AdvertisingPDUs import NullPDU
from ble.utils.HelperMethods import HelperMethods
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.ParserInterface import ParserInterface
from ble.utils.StringCursor import StringCursor


class Packet(ParserInterface, NodeInterface, PrintInterface):

    def __init__(self) -> None:
        NodeInterface.__init__(self, "Packet")
        self.access_address = AccessAddress()
        self.pdu: AbstractPDU = NullPDU()
        self.crc: CRC = CRC()
        self.time: datetime.datetime = datetime.datetime.now()
        self.rssi: HexDataField = HexDataField("RSSI", target_byte_length=1)
        self.channel: HexDataField = HexDataField("Channel", target_byte_length=1)

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.access_address.get_length(bit=bit) + self.pdu.get_length(bit=bit) + self.crc.get_length(bit=bit)

    @override
    def update(self):
        self.access_address.update()
        self.access_address.set_value("8E89BED6")
        self.channel.update()
        self.rssi.update()
        self.pdu.update()
        self.crc.update()
        self.crc.from_string(self.crc.from_pdu(self.pdu.to_string()))

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 7, byte=True)

        cursor = StringCursor(value)

        access_address = cursor.read_bytes(n=self.access_address.get_target_byte_length())
        pdu_string = cursor.read(n=cursor.remaining() - (self.crc.get_target_byte_length() * 2))
        crc = cursor.read_bytes(self.crc.get_target_byte_length())

        assert cursor.remaining() == 0, "string cursor is not empty"

        self.access_address.from_string(access_address, parse_mode=policy)
        self.pdu = policy.extract_pdu(pdu_string)
        self.crc.from_string(crc, parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return prefix * "0x" + self.access_address.to_string() + self.pdu.to_string() + self.crc.to_string()


    @override
    def _print(self, ident: int = 0) -> None:
        print("\t" * ident + self.get_name())
        self.access_address._print(ident=ident + 1)
        self.pdu._print(ident=ident + 1)
        self.crc._print(ident=ident + 1)

    def set_time(self, time: float):
        self.time = datetime.datetime.fromtimestamp(time)

    def get_time(self) -> float:
        return float(self.time.timestamp())

    @staticmethod
    def _convert_field_value(value: str | int) -> str:
        if isinstance(value, str):
            value = HelperMethods.clean_hex_value(value)

        elif isinstance(value, int):
            value = HelperMethods.int_to_hex(value, pad='byte')
        else:
            raise TypeError("value is neither a string nor an integer")

        assert len(value) == 2, "Value must be 1 Byte long"

        return value

    def set_channel(self, channel: str | int):
        channel = self._convert_field_value(channel)
        self.channel.set_value(channel)

    def set_rssi(self, rssi: str | int):
        rssi = self._convert_field_value(rssi)
        self.rssi.set_value(rssi)

    def mask(self, mask_config: MaskConfig) -> MaskConfig:
        mask_config.set_ctx(self)
        self.walk(mask_config)
        return mask_config

    def generate(self, gen_config: GenConfig) -> Self:
        gen_config.set_ctx(self)
        self.walk(gen_config)
        self.update()
        return self

    def get_channel(self, integer: bool = False) -> int | str:
        if integer:
            return int(self.channel.get_value(), 16)
        else:
            return str(self.channel.get_value())

    def get_rssi(self, integer: bool = False) -> int | str:
        if integer:
            return int(self.rssi.get_value(), 16)
        else:
            return str(self.rssi.get_value())
