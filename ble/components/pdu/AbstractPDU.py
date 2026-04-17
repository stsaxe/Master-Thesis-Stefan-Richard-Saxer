from __future__ import annotations
from typing import override, TYPE_CHECKING

from ble.errors.OverLengthError import OverLengthError
from ble.interfaces.PrintInterface import PrintInterface

if TYPE_CHECKING:
    pass

from abc import ABC

from ble.components.pdu.AdvHeader import AdvHeader
from ble.utils.HelperMethods import HelperMethods
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.ParserInterface import ParserInterface
from ble.interfaces.RegistryInterface import RegistryInterface
from ble.walking.PathSegment import PathSegment


class AbstractPDU(NodeInterface, ParserInterface, HelperMethods, RegistryInterface, PrintInterface, ABC):
    def __init__(self) -> None:
        NodeInterface.__init__(self, "PDU")
        self.header: AdvHeader = AdvHeader()
        self._update_pdu_type()

    def _update_pdu_type(self) -> None:
        if self._get_registry_key() is not None:
            self.header.pdu_type.set_value(HelperMethods.int_to_bin(self._get_registry_key(), pad='nibble'), bin=True)

    def _update_header(self) -> None:
        self._update_pdu_type()
        self.header.update()

        length = self.get_length(bit=False) - self.header.get_length(bit=False)
        length = HelperMethods.int_to_hex(length, pad='byte')

        if len(length) > self.header.length.get_target_byte_length() * 2:
            raise OverLengthError(f"PDU is too long for Header Length Field with Target Byte Length {self.header.length.get_target_byte_length()}")

        self.header.length.set_value(length)

    @override
    def _get_registry_name(self) -> str:
        return self.PDU_REGISTRY_NAME

    @override
    def _get_registry_key(self) -> int | None:
        return self.PDU_REGISTRY_KEY

    @override
    def get_path_segment(self) -> PathSegment:
        return PathSegment(self.get_name(), keys={"pdu_type": [self._get_registry_name(),
                                                               self.header.pdu_type.get_value(prefix=True, bin=True),
                                                               self.header.pdu_type.get_value(prefix=False, bin=True),
                                                               ]
                                                  })
