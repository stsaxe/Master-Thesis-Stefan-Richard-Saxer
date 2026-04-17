from abc import ABC

from ble.components.pdu.AbstractPDU import AbstractPDU
from ble.fields.HexDataField import HexDataField

class AbstractAdvPDU(AbstractPDU, ABC):
    def __init__(self) -> None:
        AbstractPDU.__init__(self)
        self.advertising_address: HexDataField = HexDataField("Advertising Address", target_byte_length=6)
