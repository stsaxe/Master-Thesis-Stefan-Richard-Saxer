from __future__ import annotations
from typing import override, TYPE_CHECKING

from ble.components.pdu.AdvHeader import AdvHeader
from ble.walking.PathSegment import PathSegment

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface

from ble.components.ComponentRegistry import PDURegistry
from ble.components.pdu.AbstractAdvDataPDU import AbstractAdvDataPDU
from ble.components.pdu.AbstractAdvTragAddr import AbstractAdvTargAddr
from ble.components.pdu.AbstractPDU import AbstractPDU
from ble.fields.HexDataField import HexDataField
from ble.utils.HelperMethods import HelperMethods
from ble.utils.StringCursor import StringCursor

PDU_REGISTRY = PDURegistry()


@PDU_REGISTRY.register(0b0000, "ADV_IND")
class AdvInd(AbstractAdvDataPDU):
    def __init__(self) -> None:
        AbstractAdvDataPDU.__init__(self)
        self.header.rx_add.set_name("RFU")


@PDU_REGISTRY.register(0b0001, "ADV_DIRECT_IND")
class AdvDirectInd(AbstractAdvTargAddr):
    def __init__(self) -> None:
        AbstractAdvTargAddr.__init__(self)


@PDU_REGISTRY.register(0b0010, "ADV_NONCONN_IND")
class AdvNonConnInd(AbstractAdvDataPDU):
    def __init__(self) -> None:
        AbstractAdvDataPDU.__init__(self)
        self.header.rx_add.set_name("RFU")
        self.header.ch_sel.set_name("RFU")


@PDU_REGISTRY.register(0b0110, "ADV_SCAN_IND")
class AdvScanInd(AbstractAdvDataPDU):
    def __init__(self) -> None:
        AbstractAdvDataPDU.__init__(self)
        self.header.rx_add.set_name("RFU")
        self.header.ch_sel.set_name("RFU")

@PDU_REGISTRY.register(0b0100, "SCAN_RSP")
class ScanRsp(AbstractAdvDataPDU):
    def __init__(self) -> None:
        AbstractAdvDataPDU.__init__(self)
        self.header.rx_add.set_name("RFU")
        self.header.ch_sel.set_name("RFU")
        self.adv_data.set_name("Scan Response Data")


@PDU_REGISTRY.register(0b0011, "SCAN_REQ")
class ScanReq(AbstractAdvTargAddr):
    def __init__(self) -> None:
        AbstractAdvTargAddr.__init__(self)
        self.header.ch_sel.set_name("RFU")



@PDU_REGISTRY.register(None,"Raw PDU")
class RawPDU(AbstractPDU):
    def __init__(self) -> None:
        AbstractPDU.__init__(self)
        self.data: HexDataField = HexDataField("Raw Data")

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.data.get_length(bit=bit) + self.header.get_length(bit=bit)

    @override
    def update(self) -> None:
        self.data.update()
        self._update_header()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 2, byte=True)

        cursor = StringCursor(value)

        self.header.from_string(cursor.read_bytes(2))
        self.data.from_string(cursor.read_to_end(byte_aware=True))

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return prefix * "0x" + self.header.to_string() + self.data.to_string()

    @override
    def print(self) -> None:
        self._print(ident=0)

    @override
    def _print(self, ident: int = 0) -> None:
        print("\t" * ident + self.get_name() + ": " + self._get_registry_name())
        self.header._print(ident=ident + 1)
        self.data._print(ident=ident + 1)

    @override
    def get_path_segment(self) -> PathSegment:
        return PathSegment(self.get_name(), keys={"pdu_type": self._get_registry_name()})


@PDU_REGISTRY.register(None, "Null PDU")
class NullPDU(AbstractPDU):
    def __init__(self) -> None:
        AbstractPDU.__init__(self)

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.header.get_length(bit=bit)

    @override
    def update(self) -> None:
        self.header = AdvHeader()
    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        pass

    @override
    def to_string(self, prefix: bool = False) -> str:
        return prefix * "0x" + ""

    @override
    def _print(self, ident: int = 0) -> None:
        pass

