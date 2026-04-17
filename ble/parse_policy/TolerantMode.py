from typing import override

from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.components.pdu.AbstractPDU import AbstractPDU
from ble.components.pdu.AbstractAdvDataPDU import AbstractAdvDataPDU
from ble.components.pdu.AdvHeader import AdvHeader
from ble.errors.ParseError import ParseError
from ble.parse_policy.BaseParsePolicy import BaseParsePolicy
from ble.components.advertising_data.AdvertisingData import Flags, ManufacturerSpecific, TxPowerLevel, RawAdvDataStruct, \
    ADVERTISING_REGISTRY, Appearance
from ble.parse_policy.ParseRegistry import PARSE_POLICY_REGISTRY
from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface
from ble.components.advertising_data.AbstractServiceData import AbstractServiceDataStruct
from ble.components.advertising_data.AbstractServiceUUIDList import AbstractServiceUUIDListStruct

from ble import RawPDU, ScanReq, ScanRsp, AdvScanInd, AdvNonConnInd, AdvDirectInd, AdvInd, NullPDU, PDU_REGISTRY

from ble.components.advertising_data.AbstractLocalName import AbstractLocalNameStruct
from ble.utils.StringCursor import StringCursor


@PARSE_POLICY_REGISTRY.register("tolerant")
class TolerantMode(BaseParsePolicy):

    @override
    @staticmethod
    def minimum_hex_string_length(value: str, length: int, byte: bool = False) -> None:
        BaseParsePolicy.check_valid_string(value, empty_allowed=True)
        assert isinstance(length, int), "length must be an integer"

        if len(value) < length and not byte:
            raise ParseError(f"Length of value must be at least equal to {length}")

        elif byte and len(value) // 2 < length:
            raise ParseError(f"Length of value must be at least equal to {length} Bytes")

    @staticmethod
    @override
    def hex_string_length(value: str, length: int, byte: bool = False) -> None:
        TolerantMode.minimum_hex_string_length(value, length, byte)

    @staticmethod
    @override
    def hex_string_is_in_byte_format(value: str, empty_allowed: bool = False) -> None:
        pass

    @ParsePolicyInterface.verify.register(Flags)
    @override
    def verify_flags_adv_struct(self, component: Flags):
        fields = [component.bit_0, component.bit_1, component.bit_2, component.bit_3, component.bit_4]

        for field in fields:
            if field.get_bit_length() != 1:
                raise ParseError(f"Field {field.get_name()} must have length 1 Bit")


    @ParsePolicyInterface.verify.register(TxPowerLevel)
    @override
    def verify_tx_power_level_adv_struct(self, component: TxPowerLevel):
        ParsePolicyInterface.verify_hex_field_length(component.power_level, 8, byte=False)

    @ParsePolicyInterface.verify.register(RawAdvDataStruct)
    @override
    def verify_raw_adv_struct(self, component: RawAdvDataStruct):
        ParsePolicyInterface.verify_hex_field_length(component.type, 8, byte=False)


    @ParsePolicyInterface.verify.register(ManufacturerSpecific)
    @override
    def verify_manufacturer_specific_adv_struct(self, component: ManufacturerSpecific):
        pass

    @override
    @ParsePolicyInterface.verify.register(AbstractLocalNameStruct)
    def verify_local_name_struct(self, component: AbstractLocalNameStruct):
        pass

    @override
    @ParsePolicyInterface.verify.register(AbstractServiceDataStruct)
    def verify_service_data_struct(self, component: AbstractServiceDataStruct):
        ParsePolicyInterface.verify_hex_field_length(component.uuid, component.bit, byte=False)


    @override
    @ParsePolicyInterface.verify.register(AbstractServiceUUIDListStruct)
    def verify_service_uuid_list(self, component: AbstractServiceUUIDListStruct):
        for uuid in component.uuids:
            ParsePolicyInterface.verify_hex_field_length(uuid, component.bit, byte=False)


    @ParsePolicyInterface.verify.register(Appearance)
    def verify_appearance_adv_struct(self, component: Appearance):
        ParsePolicyInterface.verify_hex_field_length(component.appearance, 2, byte=True)

    @staticmethod
    def verify_advertising_data_pdu(component: AbstractAdvDataPDU):
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

    @ParsePolicyInterface.verify.register(AdvInd)
    def verify_adv_ind_pdu(self, component: AdvInd):
        self.verify_advertising_data_pdu(component)


    @ParsePolicyInterface.verify.register(AdvDirectInd)
    def verify_adv_direct_ind_pdu(self, component: AdvDirectInd):
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

    @ParsePolicyInterface.verify.register(AdvNonConnInd)
    def verify_adv_non_conn_ind_pdu(self, component: AdvNonConnInd):
        self.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(AdvScanInd)
    def verify_adv_scan_ind_pdu(self, component: AdvScanInd):
        self.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(ScanRsp)
    def verify_adv_scan_rsp_pdu(self, component: ScanRsp):
        self.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(ScanReq)
    def verify_adv_scan_req_pdu(self, component: ScanReq):
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)


    @ParsePolicyInterface.verify.register(RawPDU)
    def verify_adv_raw_pdu(self, component: RawPDU):
        pass


    @ParsePolicyInterface.verify.register(AdvHeader)
    def verify_adv_header(self, component: AdvHeader):
        ParsePolicyInterface.verify_header_fields(component, verify_length=False)

    @override
    def extract_adv_struct(self, cursor: StringCursor, adv_type: str, length: int) -> AbstractAdvDataStruct:
        try:
            adv_struct = ADVERTISING_REGISTRY[int(adv_type, 16)]()
        except KeyError:
            adv_struct = RawAdvDataStruct()

        try:
            adv_string = cursor.read_bytes(length + 1)
        except:
            adv_string = cursor.read_to_end()

        try:
            adv_struct.from_string(adv_string, parse_mode=self)
        except:
            adv_struct = RawAdvDataStruct()
            adv_struct.from_string(adv_string, parse_mode=self)

        return adv_struct

    @override
    def extract_pdu(self, pdu_string: str) -> AbstractPDU:
        if len(pdu_string) >= 2:
            pdu_type_int = self._extract_pdu_type(pdu_string)
            try:
                pdu = PDU_REGISTRY[pdu_type_int]()
            except KeyError:
                pdu = RawPDU()

            pdu.from_string(pdu_string, parse_mode=self)

            return pdu

        else:
            return NullPDU()
