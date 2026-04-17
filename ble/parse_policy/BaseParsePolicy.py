from abc import ABC
from typing import override

from ble.components.advertising_data.AbstractAdvDataStruct import AbstractAdvDataStruct
from ble.components.pdu.AbstractPDU import AbstractPDU
from ble.components.pdu.AdvertisingPDUs import RawPDU, NullPDU, PDU_REGISTRY
from ble.components.advertising_data.AbstractServiceData import AbstractServiceDataStruct
from ble.errors.ParseError import ParseError
from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface
from ble.components.advertising_data.AdvertisingData import TxPowerLevel, RawAdvDataStruct, ManufacturerSpecific, \
    ADVERTISING_REGISTRY, Appearance
from ble.components.advertising_data.AbstractLocalName import AbstractLocalNameStruct
from ble.components.advertising_data.AbstractServiceUUIDList import AbstractServiceUUIDListStruct
from ble.utils.StringCursor import StringCursor


class BaseParsePolicy(ParsePolicyInterface, ABC):
    @override
    @ParsePolicyInterface.verify.register(TxPowerLevel)
    def verify_tx_power_level_adv_struct(self, component: TxPowerLevel):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        ParsePolicyInterface.verify_hex_field_length(component.power_level, 8, byte=False)

    @override
    @ParsePolicyInterface.verify.register(RawAdvDataStruct)
    def verify_raw_adv_struct(self, component: RawAdvDataStruct):
        ParsePolicyInterface.verify_adv_data_length(component)

        ParsePolicyInterface.verify_hex_field_length(component.type, 8, byte=False)


    @override
    @ParsePolicyInterface.verify.register(ManufacturerSpecific)
    def verify_manufacturer_specific_adv_struct(self, component: ManufacturerSpecific):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        data_length = int(component.data.get_length(bit=True) // 8) * 8
        #ParsePolicyInterface.verify_hex_field_length(component.data, data_length, byte=False)


    @override
    @ParsePolicyInterface.verify.register(AbstractLocalNameStruct)
    def verify_local_name_struct(self, component: AbstractLocalNameStruct):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        #name_length = int(component.device_name.get_length(bit=True) // 8) * 8
        #ParsePolicyInterface.verify_hex_field_length(component.device_name, name_length, byte=False)


    @override
    @ParsePolicyInterface.verify.register(AbstractServiceDataStruct)
    def verify_service_data_struct(self, component: AbstractServiceDataStruct):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        ParsePolicyInterface.verify_hex_field_length(component.uuid, component.bit, byte=False)

        #data_length = int(component.data.get_length(bit=True) // 8) * 8
        #ParsePolicyInterface.verify_hex_field_length(component.data, data_length, byte=False)

    @override
    @ParsePolicyInterface.verify.register(AbstractServiceUUIDListStruct)
    def verify_service_uuid_list(self, component: AbstractServiceUUIDListStruct):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        for uuid in component.uuids:
            ParsePolicyInterface.verify_hex_field_length(uuid, component.bit, byte=False)

    @ParsePolicyInterface.verify.register(Appearance)
    def verify_appearance_adv_struct(self, component: Appearance):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        ParsePolicyInterface.verify_hex_field_length(component.appearance, 2, byte=True)


    @ParsePolicyInterface.verify.register(RawPDU)
    def verify_adv_raw_pdu(self, component: RawPDU):
        if component.get_length() != int(component.header.length.get_value(), 16) + 2:
            raise ParseError("PDU Header Length does not match Length indicated in PDU")

    @override
    def extract_adv_struct(self, cursor: StringCursor, adv_type: str, length: int) -> AbstractAdvDataStruct:
        try:
            adv_struct = ADVERTISING_REGISTRY[int(adv_type, 16)]()
        except KeyError:
            adv_struct = RawAdvDataStruct()

        try:
            adv_string = cursor.read_bytes(length + 1)
        except:
            raise ParseError(
                "Ran out of Bytes to read, provided hex-string is too short for indicated adv struct length")

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
            pdu = pdu

        elif len(pdu_string) == 0:
            pdu = NullPDU()

        else:
            raise ParseError("Unable to parse PDU")

        return pdu


