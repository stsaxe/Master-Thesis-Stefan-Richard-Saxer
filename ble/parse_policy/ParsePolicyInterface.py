from __future__ import annotations

from abc import abstractmethod, ABC

from ble.components.advertising_data.AdvertisingData import Appearance
from ble.utils.StringCursor import StringCursor
from ble.components.packet.AccessAddress import AccessAddress
from ble.components.packet.CRC import CRC
from ble.components.pdu.AbstractPDU import AbstractPDU
from ble.components.advertising_data.AbstractServiceData import AbstractServiceDataStruct
from ble.components.advertising_data.AbstractServiceUUIDList import AbstractServiceUUIDListStruct
from ble.components.pdu.AdvHeader import AdvHeader
from ble.components.pdu.AdvertisingPDUs import AdvInd, AdvDirectInd, AdvNonConnInd, AdvScanInd, ScanRsp, ScanReq, RawPDU
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AbstractLocalName import AbstractLocalNameStruct
from ble.utils.HelperMethods import HelperMethods

from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.components.advertising_data.AdvertisingData import Flags, TxPowerLevel, RawAdvDataStruct, ManufacturerSpecific

from ble.errors.ParseError import ParseError
from ble.parse_policy.MultiDispatch import MultiDispatchSupport, multidispatchmethod
from ble.components.packet.Packet import Packet

class ParsePolicyInterface(MultiDispatchSupport, HelperMethods, ABC):

    @staticmethod
    def hex_string_is_in_byte_format(value: str, empty_allowed: bool = False) -> None:
        value = HelperMethods.clean_hex_value(value, empty_allowed=empty_allowed)

        if len(value) % 2 != 0:
            raise ParseError(f"Length of value must be a Multiple of 2")

    @staticmethod
    def hex_string_length(value: str, length: int, byte: bool = False) -> None:
        assert isinstance(length, int), "length must be an integer"
        assert len(value) >= 0, "length must be positive or zero"
        HelperMethods.check_valid_string(value, empty_allowed=True)

        if len(value) != length and not byte:
            raise ParseError(f"Length of value must be equal to {length}")

        elif byte:
            ParsePolicyInterface.hex_string_is_in_byte_format(value)

            if len(value) // 2 != length:
                raise ParseError(f"Length of value must be equal to {length} Bytes")

    @staticmethod
    def minimum_hex_string_length(value: str, length: int, byte: bool = False) -> None:
        HelperMethods.check_valid_string(value, empty_allowed=True)
        assert isinstance(length, int), "length must be an integer"

        if len(value) < length and not byte:
            raise ParseError(f"Length of value must be at least equal to {length}")
        elif byte:
            ParsePolicyInterface.hex_string_is_in_byte_format(value)

            if len(value) // 2 < length:
                raise ParseError(f"Length of value must be at least equal to {length} Bytes")

    @staticmethod
    def verify_adv_data_type(component: AbstractAdvDataStruct):
        if component.type.get_value() != HelperMethods.int_to_hex(component._get_registry_key(), pad="byte"):
            raise ParseError(f"Advertising Data type does not match type of class")

        ParsePolicyInterface.verify_hex_field_length(component.type, 8, byte=False)

    @staticmethod
    def verify_adv_data_length(component: AbstractAdvDataStruct):
        if component.length.get_value() != HelperMethods.int_to_hex(component.get_length() - 1, pad="byte"):
            raise ParseError(f"Length of Advertising Data Struct does not match length indicated in the hex string")

        ParsePolicyInterface.verify_hex_field_length(component.length, 8, byte=False)

    @staticmethod
    def verify_hex_field_length(field: HexDataField, length: int, byte: bool = False) -> None:
        field_length = field.get_length(bit=not byte)

        if field_length != length:
            message = f"Field {field.get_name()} must have length {length}"

            if byte:
                message += " Bytes"
            else:
                message += " Bits"

            raise ParseError(message)

    @multidispatchmethod
    def verify(self, *args):
        pass

    @abstractmethod
    @verify.register(Flags)
    def verify_flags_adv_struct(self, component: Flags):
        pass

    @abstractmethod
    @verify.register(RawAdvDataStruct)
    def verify_raw_adv_struct(self, component: RawAdvDataStruct):
        pass

    @abstractmethod
    @verify.register(TxPowerLevel)
    def verify_tx_power_level_adv_struct(self, component: TxPowerLevel):
        pass

    @abstractmethod
    @verify.register(ManufacturerSpecific)
    def verify_manufacturer_specific_adv_struct(self, component: ManufacturerSpecific):
        pass

    @abstractmethod
    @verify.register(AbstractLocalNameStruct)
    def verify_local_name_struct(self, component: AbstractLocalNameStruct):
        pass

    @abstractmethod
    @verify.register(AbstractServiceDataStruct)
    def verify_service_data_struct(self, component: AbstractServiceDataStruct):
        pass

    @abstractmethod
    @verify.register(AbstractServiceUUIDListStruct)
    def verify_service_uuid_list(self, component: AbstractServiceUUIDListStruct):
        pass

    @abstractmethod
    @verify.register(Appearance)
    def verify_appearance_adv_struct(self, component: Appearance):
        pass

    @staticmethod
    def verify_header_fields(header: AdvHeader, verify_length: bool = True):
        if header.ch_sel.get_bit_length() != 1:
            raise ParseError(f"PDU Header Field {header.ch_sel.get_name()} does not have bit length 1")

        if header.tx_add.get_bit_length() != 1:
            raise ParseError(f"PDU Header Field {header.tx_add.get_name()} does not have bit length 1")

        if header.rx_add.get_bit_length() != 1:
            raise ParseError(f"PDU Header Field {header.rx_add.get_name()} does not have bit length 1")

        if header.rfu.get_bit_length() != 1:
            raise ParseError(f"PDU Header Field {header.rfu.get_name()} does not have bit length 1")

        if header.pdu_type.get_bit_length() != 4:
            raise ParseError(f"PDU Header Field {header.pdu_type.get_name()} does not have bit length 4")

        if verify_length:
            ParsePolicyInterface.verify_hex_field_length(header.length, 1, byte=True)

    @staticmethod
    def verify_pdu_header(pdu: AbstractPDU):
        header = pdu.header

        if HelperMethods.int_to_bin(pdu._get_registry_key(), pad='nibble') != header.pdu_type.get_value(prefix=False,
                                                                                                        bin=True):
            raise ParseError(f"PDU Header does not indicate correct Type")

        if pdu.get_length() != int(header.length.get_value(), 16) + 2:
            raise ParseError("PDU Header Length does not match Length indicated in PDU")

    @abstractmethod
    @verify.register(AdvInd)
    def verify_adv_ind_pdu(self, component: AdvInd):
        pass

    @abstractmethod
    @verify.register(AdvDirectInd)
    def verify_adv_direct_ind_pdu(self, component: AdvDirectInd):
        pass

    @abstractmethod
    @verify.register(AdvNonConnInd)
    def verify_adv_non_conn_ind_pdu(self, component: AdvNonConnInd):
        pass

    @abstractmethod
    @verify.register(AdvScanInd)
    def verify_adv_scan_ind_pdu(self, component: AdvScanInd):
        pass

    @abstractmethod
    @verify.register(ScanRsp)
    def verify_adv_scan_rsp_pdu(self, component: ScanRsp):
        pass

    @abstractmethod
    @verify.register(ScanReq)
    def verify_adv_scan_req_pdu(self, component: ScanReq):
        pass

    @abstractmethod
    @verify.register(RawPDU)
    def verify_adv_raw_pdu(self, component: RawPDU):
        pass

    @abstractmethod
    @verify.register(AdvHeader)
    def verify_adv_header(self, component: AdvHeader):
        pass

    @verify.register(AccessAddress)
    def verify_access_address(self, component: AccessAddress):
        ParsePolicyInterface.verify_hex_field_length(component, length=4, byte=True)

    @verify.register(CRC)
    def verify_crc(self, component: CRC):
        ParsePolicyInterface.verify_hex_field_length(component, length=3, byte=True)

    @abstractmethod
    def extract_adv_struct(self, cursor: StringCursor, adv_type: str, length: int) -> AbstractAdvDataStruct:
        pass

    @abstractmethod
    def extract_pdu(self, pdu_string: str) -> AbstractPDU:
        pass

    @staticmethod
    def _extract_pdu_type(pdu_string: str) -> int:
        value = HelperMethods.clean_hex_value(pdu_string)
        assert len(value) >= 2, "pdu string must have at least 1 Byte"
        return int(value[1], 16)

    @verify.register(Packet)
    def verify_packet(self, component: Packet):
        pass