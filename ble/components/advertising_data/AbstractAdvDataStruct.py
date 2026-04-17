from __future__ import annotations

from abc import ABC
from typing import override

from ble.errors.OverLengthError import OverLengthError
from ble.fields.HexDataField import HexDataField
from ble.interfaces.PrintInterface import PrintInterface
from ble.utils.HelperMethods import HelperMethods
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.ParserInterface import ParserInterface
from ble.interfaces.RegistryInterface import RegistryInterface
from ble.walking.PathSegment import PathSegment


class AbstractAdvDataStruct(NodeInterface, ParserInterface, HelperMethods, RegistryInterface, PrintInterface, ABC):
    def __init__(self) -> None:
        NodeInterface.__init__(self, "Adv Struct")
        self.length: HexDataField = HexDataField("Length",'00', target_byte_length=1)

        self.type: HexDataField = HexDataField("Type", target_byte_length=1)
        self._set_type()

    def _set_type(self):
        if self._get_registry_key() is not None:
            self.type.set_value(HelperMethods.int_to_hex(self._get_registry_key(), pad="byte"))

    def _update_length(self) -> None:
        length = self.get_length(bit=False) - self.length.get_length(bit=False)
        length = HelperMethods.int_to_hex(length, pad="byte")

        if len(length) > self.length.get_target_byte_length() * 2:
            raise OverLengthError(f"Adv Struct is too long for Length Field with Target Byte Length {self.length.get_target_byte_length()}")

        self.length.set_value(length)

    def _print_first_lines(self, ident: int = 0) -> None:
        print('\t' * ident + self.get_name() + f": {self._get_registry_name()}")
        self.length._print(ident=ident + 1)
        self.type._print(ident=ident + 1)

    @override
    def _get_registry_name(self) -> str:
        return self.ADVERTISING_REGISTRY_NAME

    @override
    def _get_registry_key(self) -> int | None:
        return self.ADVERTISING_REGISTRY_KEY

    @override
    def get_path_segment(self) -> PathSegment:
        return PathSegment(self.get_name(), keys={"adv_type": [self._get_registry_name(),
                                                               HelperMethods.int_to_hex(self._get_registry_key(), pad="byte"),
                                                               self.type.get_value(prefix=True),
                                                               self.type.get_value(prefix=False)
                                                               ]
                                                  })