from __future__ import annotations
from typing import override, TYPE_CHECKING

from ble.walking.PathSegment import PathSegment

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface
from ble.utils.StringCursor import StringCursor
from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.components.advertising_data.AbstractServiceData import AbstractServiceDataStruct
from ble.components.advertising_data.AbstractServiceUUIDList import AbstractServiceUUIDListStruct
from ble.components.advertising_data.AbstractLocalName import AbstractLocalNameStruct
from ble.components.ComponentRegistry import AdvertisingRegistry
from ble.fields.BitDataField import BitDataField
from ble.fields.HexDataField import HexDataField
from ble.utils.HelperMethods import HelperMethods

ADVERTISING_REGISTRY = AdvertisingRegistry()


@ADVERTISING_REGISTRY.register(None, "Raw Advertising Data")
class RawAdvDataStruct(AbstractAdvDataStruct):
    def __init__(self) -> None:
        AbstractAdvDataStruct.__init__(self)
        self.data: HexDataField = HexDataField("Data")

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.data.get_length(bit=bit) + self.length.get_length(bit=bit) + self.type.get_length(bit=bit)

    @override
    def update(self):
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)

        policy = self.get_parse_policy(parse_mode)
        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 2, byte=True)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(self.length.get_target_byte_length()), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(self.type.get_target_byte_length()), parse_mode=parse_mode)
        self.data.from_string(cursor.read_to_end(byte_aware=True), parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return prefix * "0x" + self.length.to_string() + self.type.to_string() + self.data.to_string()

    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        self.data._print(ident + 1)

    @override
    def get_path_segment(self) -> PathSegment:
        return PathSegment(self.get_name(), keys={"adv_type": self._get_registry_name()})


@ADVERTISING_REGISTRY.register(0x01, "Flags")
@ADVERTISING_REGISTRY.requirements(0x01, {"EIR": "C1", "AD": "C1", "SRD": "X", "ACAD": "X", "OOB": "C1"})
class Flags(AbstractAdvDataStruct):
    def __init__(self) -> None:
        AbstractAdvDataStruct.__init__(self)
        self.bit_0: BitDataField = BitDataField("LE Limited Discoverable Mode", target_bit_length=1)
        self.bit_1: BitDataField = BitDataField("LE General Discoverable Mode", target_bit_length=1)
        self.bit_2: BitDataField = BitDataField(r"BR/EDR Not Supported", target_bit_length=1)
        self.bit_3: BitDataField = BitDataField(r"Simultaneous LE and BR/EDR to Same Device Capable (Controller)",
                                                target_bit_length=1)
        self.bit_4: BitDataField = BitDataField(r"Simultaneous LE and BR/EDR to Same Device Capable (Host)",
                                                target_bit_length=1)
        self.bit_5_to_7: BitDataField = BitDataField("RFU", target_bit_length=3)

    @override
    def get_length(self, bit: bool = False) -> int:
        length = self.length.get_length(bit=bit) + self.type.get_length(bit=bit)

        length_bits = (self.bit_0.get_bit_length()
                       + self.bit_1.get_bit_length()
                       + self.bit_2.get_bit_length()
                       + self.bit_3.get_bit_length()
                       + self.bit_4.get_bit_length()
                       + self.bit_5_to_7.get_bit_length())

        if not bit:
            length_bits = length_bits // 8

        return length + length_bits

    @override
    def update(self):
        self._set_type()
        self.bit_5_to_7.set_value("000", bin=True)
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.hex_string_length(value, length=3, byte=True)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(self.length.get_target_byte_length()), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(self.type.get_target_byte_length()), parse_mode=parse_mode)

        flags = HelperMethods.hex_to_bin(cursor.read_bytes(1), pad='byte')

        self.bit_0.set_value(flags[7], bin=True)
        self.bit_1.set_value(flags[6], bin=True)
        self.bit_2.set_value(flags[5], bin=True)
        self.bit_3.set_value(flags[4], bin=True)
        self.bit_4.set_value(flags[3], bin=True)
        self.bit_5_to_7.set_value(flags[:3], bin=True)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        out = self.length.to_string() + self.type.to_string()

        flags = self.bit_5_to_7.get_value(bin=True)
        flags += self.bit_4.get_value(bin=True)
        flags += self.bit_3.get_value(bin=True)
        flags += self.bit_2.get_value(bin=True)
        flags += self.bit_1.get_value(bin=True)
        flags += self.bit_0.get_value(bin=True)

        flags = HelperMethods.bin_to_hex(flags, pad='byte')

        return prefix * '0x' + out + flags

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        self.bit_5_to_7._print(ident=ident + 1)
        self.bit_4._print(ident=ident + 1)
        self.bit_3._print(ident=ident + 1)
        self.bit_2._print(ident=ident + 1)
        self.bit_1._print(ident=ident + 1)
        self.bit_0._print(ident=ident + 1)


@ADVERTISING_REGISTRY.register(0x02, "Incomplete List of 16 bit Service or Service Class UUIDs")
@ADVERTISING_REGISTRY.requirements(0x02, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceUUID16ListIncomplete(AbstractServiceUUIDListStruct):
    def __init__(self) -> None:
        AbstractServiceUUIDListStruct.__init__(self, bit=16)


@ADVERTISING_REGISTRY.register(0x03, "Complete List of 16 bit Service or Service Class UUIDs")
@ADVERTISING_REGISTRY.requirements(0x03, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceUUID16ListComplete(AbstractServiceUUIDListStruct):
    def __init__(self) -> None:
        AbstractServiceUUIDListStruct.__init__(self, bit=16)


@ADVERTISING_REGISTRY.register(0x04, "Incomplete List of 32 bit Service or Service Class UUIDs")
@ADVERTISING_REGISTRY.requirements(0x04, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceUUID32ListIncomplete(AbstractServiceUUIDListStruct):
    def __init__(self) -> None:
        AbstractServiceUUIDListStruct.__init__(self, bit=32)


@ADVERTISING_REGISTRY.register(0x05, "Complete List of 32 bit Service or Service Class UUIDs")
@ADVERTISING_REGISTRY.requirements(0x05, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceUUID32ListComplete(AbstractServiceUUIDListStruct):
    def __init__(self) -> None:
        AbstractServiceUUIDListStruct.__init__(self, bit=32)


@ADVERTISING_REGISTRY.register(0x06, "Incomplete List of 128 bit Service or Service Class UUIDs")
@ADVERTISING_REGISTRY.requirements(0x06, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceUUID128ListIncomplete(AbstractServiceUUIDListStruct):
    def __init__(self) -> None:
        AbstractServiceUUIDListStruct.__init__(self, bit=128)


@ADVERTISING_REGISTRY.register(0x07, "Complete List of 128 bit Service or Service Class UUIDs")
@ADVERTISING_REGISTRY.requirements(0x07, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceUUID128ListComplete(AbstractServiceUUIDListStruct):
    def __init__(self) -> None:
        AbstractServiceUUIDListStruct.__init__(self, bit=128)


@ADVERTISING_REGISTRY.register(0x08, "Shortened Local Name")
@ADVERTISING_REGISTRY.requirements(0x08, {"EIR": "C1", "AD": "C1", "SRD": "C1", "ACAD": "X", "OOB": "C1"})
class ShortenedLocalName(AbstractLocalNameStruct):
    def __init__(self) -> None:
        AbstractLocalNameStruct.__init__(self)


@ADVERTISING_REGISTRY.register(0x09, "Complete Local Name")
@ADVERTISING_REGISTRY.requirements(0x09, {"EIR": "C1", "AD": "C1", "SRD": "C1", "ACAD": "X", "OOB": "C1"})
class CompleteLocalName(AbstractLocalNameStruct):
    def __init__(self) -> None:
        AbstractLocalNameStruct.__init__(self)


@ADVERTISING_REGISTRY.register(0x0a, "Tx Power Level")
@ADVERTISING_REGISTRY.requirements(0x0a, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "X", "OOB": "O"})
class TxPowerLevel(AbstractAdvDataStruct):
    def __init__(self) -> None:
        AbstractAdvDataStruct.__init__(self)
        self.power_level: HexDataField = HexDataField("Power Level", target_byte_length=1)

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.power_level.get_length(bit=bit) + self.length.get_length(bit=bit) + self.type.get_length(bit=bit)

    @override
    def update(self):
        self._set_type()
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.hex_string_length(value, 3, byte=True)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(self.length.get_target_byte_length()), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(self.type.get_target_byte_length()), parse_mode=parse_mode)
        self.power_level.from_string(cursor.read_bytes(self.power_level.get_target_byte_length()),
                                     parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return prefix * '0x' + self.length.to_string() + self.type.to_string() + self.power_level.to_string()

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        self.power_level._print(ident=ident + 1)


@ADVERTISING_REGISTRY.register(0x16, "Service Data 16 bit UUID")
@ADVERTISING_REGISTRY.requirements(0x16, {"EIR": "X", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceData16Bits(AbstractServiceDataStruct):
    def __init__(self):
        AbstractServiceDataStruct.__init__(self, bit=16)


@ADVERTISING_REGISTRY.register(0x20, "Service Data 32 bit UUID")
@ADVERTISING_REGISTRY.requirements(0x20, {"EIR": "X", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceData32Bits(AbstractServiceDataStruct):
    def __init__(self):
        AbstractServiceDataStruct.__init__(self, bit=32)


@ADVERTISING_REGISTRY.register(0x21, "Service Data 128 bit UUID")
@ADVERTISING_REGISTRY.requirements(0x21, {"EIR": "X", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ServiceData128Bits(AbstractServiceDataStruct):
    def __init__(self):
        AbstractServiceDataStruct.__init__(self, bit=128)


@ADVERTISING_REGISTRY.register(0xff, "Manufacturer Specific")
@ADVERTISING_REGISTRY.requirements(0xff, {"EIR": "O", "AD": "O", "SRD": "O", "ACAD": "O", "OOB": "O"})
class ManufacturerSpecific(AbstractAdvDataStruct):
    def __init__(self) -> None:
        AbstractAdvDataStruct.__init__(self)
        self.company_id: HexDataField = HexDataField("Company ID", target_byte_length=2)
        self.data: HexDataField = HexDataField("Data")

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.data.get_length(bit=bit) + self.company_id.get_length(bit=bit) + self.length.get_length(
            bit=bit) + self.type.get_length(bit=bit)

    @override
    def update(self):
        self.company_id.update()
        self.data.update()
        self._set_type()
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.minimum_hex_string_length(value, 4, byte=True)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(self.length.get_target_byte_length()), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(self.type.get_target_byte_length()), parse_mode=parse_mode)
        self.company_id.from_string(
            HelperMethods.hex_le_to_be(cursor.read_bytes(self.company_id.get_target_byte_length())),
            parse_mode=parse_mode)
        self.data.from_string(cursor.read_to_end(byte_aware=True), parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return "0x" * prefix + self.length.to_string() + self.type.to_string() + HelperMethods.hex_be_to_le(
            self.company_id.to_string()) + self.data.to_string()

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        self.company_id._print(ident=ident + 1)
        self.data._print(ident=ident + 1)


@ADVERTISING_REGISTRY.register(0x19, "Appearance")
@ADVERTISING_REGISTRY.requirements(0x19, {"EIR": "X", "AD": "C2", "SRD": "C2", "ACAD": "X", "OOB": "C1"})
class Appearance(AbstractAdvDataStruct):
    def __init__(self) -> None:
        AbstractAdvDataStruct.__init__(self)
        self.appearance: HexDataField = HexDataField("Appearance", target_byte_length=2)

    @override
    def get_length(self, bit: bool = False) -> int:
        return self.appearance.get_length(bit=bit) + self.length.get_length(bit=bit) + self.type.get_length(bit=bit)

    @override
    def update(self):
        self.appearance.update()
        self._set_type()
        self._update_length()

    @override
    def from_string(self, value: str, parse_mode: str | ParsePolicyInterface = "normal") -> None:
        value = HelperMethods.clean_hex_value(value)
        policy = self.get_parse_policy(parse_mode=parse_mode)

        policy.hex_string_is_in_byte_format(value)
        policy.hex_string_length(value, 4, byte=True)

        cursor = StringCursor(value)

        self.length.from_string(cursor.read_bytes(self.length.get_target_byte_length()), parse_mode=parse_mode)
        self.type.from_string(cursor.read_bytes(self.type.get_target_byte_length()), parse_mode=parse_mode)

        self.appearance.from_string(
            HelperMethods.hex_le_to_be(cursor.read_bytes(self.appearance.get_target_byte_length())),
            parse_mode=parse_mode)

        policy.verify(self)

    @override
    def to_string(self, prefix: bool = False) -> str:
        return "0x" * prefix + self.length.to_string() + self.type.to_string() + HelperMethods.hex_be_to_le(
            self.appearance.to_string())

    @override
    def _print(self, ident: int = 0) -> None:
        self._print_first_lines(ident=ident)
        self.appearance._print(ident=ident + 1)
