import math
from typing import override, List

from ble.components.MultiNodeContainer import MultiNodeContainer
from ble.components.packet.Packet import Packet
from ble.components.pdu.AbstractAdvDataPDU import AbstractAdvDataPDU
from ble.components.pdu.AdvHeader import AdvHeader
from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface
from ble.components.advertising_data.AdvertisingData import Flags, TxPowerLevel, RawAdvDataStruct
from ble.errors.ParseError import ParseError
from ble.parse_policy.BaseParsePolicy import BaseParsePolicy
from ble.parse_policy.ParseRegistry import PARSE_POLICY_REGISTRY
from ble import RawPDU, ScanReq, ScanRsp, AdvScanInd, AdvNonConnInd, AdvDirectInd, AdvInd, PDU_REGISTRY, \
    ADVERTISING_REGISTRY, AbstractAdvDataStruct


@PARSE_POLICY_REGISTRY.register("strict")
class StrictMode(BaseParsePolicy):
    @ParsePolicyInterface.verify.register(Flags)
    @override
    def verify_flags_adv_struct(self, component: Flags):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        if component.bit_5_to_7.get_value(bin=True) != '000':
            raise ParseError(f"Bits 5 to 7 are reserved and must be 0")

        fields = [component.bit_0, component.bit_1, component.bit_2, component.bit_3, component.bit_4]

        for field in fields:
            if field.get_bit_length() != 1:
                raise ParseError(f"Field {field.get_name()} must have length 1 Bit")


    @staticmethod
    def verify_adv_struct_occurrences(adv_data: MultiNodeContainer | List[AbstractAdvDataStruct], context: str):
        occurrences_for_context = ADVERTISING_REGISTRY.get_occurrences_per_context(context)

        found_occurrences = dict()
        for adv_type in occurrences_for_context.keys():
            found_occurrences[adv_type] = 0


        for adv_data_struct in adv_data:
            adv_type = adv_data_struct._get_registry_key()

            if adv_type is not None:
                found_occurrences[adv_type] += 1

        for adv_type, count in found_occurrences.items():
            if count > occurrences_for_context[adv_type]:
                raise ParseError(f"Advertising type {hex(adv_type)} has {count} occurrences, which is more than allowed in advertising context {context}")

    @staticmethod
    def verify_advertising_data_pdu(component: AbstractAdvDataPDU):
        ParsePolicyInterface.verify_pdu_header(component)
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

        if component.get_length() - component.header.get_length() > 37:
            raise ParseError("PDU is too long, maximum allowed length for Payload is 37 Bytes")


    @ParsePolicyInterface.verify.register(AdvInd)
    def verify_adv_ind_pdu(self, component: AdvInd):
        StrictMode.verify_advertising_data_pdu(component)
        StrictMode.verify_adv_struct_occurrences(component.adv_data, context='AD')

        if component.header.rx_add.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

    @ParsePolicyInterface.verify.register(AdvDirectInd)
    def verify_adv_direct_ind_pdu(self, component: AdvDirectInd):
        ParsePolicyInterface.verify_pdu_header(component)
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

    @ParsePolicyInterface.verify.register(AdvNonConnInd)
    def verify_adv_non_conn_ind_pdu(self, component: AdvNonConnInd):
        StrictMode.verify_advertising_data_pdu(component)
        StrictMode.verify_adv_struct_occurrences(component.adv_data, context='AD')

        if component.header.rx_add.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

        if component.header.ch_sel.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

    @ParsePolicyInterface.verify.register(AdvScanInd)
    def verify_adv_scan_ind_pdu(self, component: AdvScanInd):
        StrictMode.verify_advertising_data_pdu(component)
        StrictMode.verify_adv_struct_occurrences(component.adv_data, context='AD')

        if component.header.rx_add.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

        if component.header.ch_sel.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

    @ParsePolicyInterface.verify.register(ScanRsp)
    def verify_adv_scan_rsp_pdu(self, component: ScanRsp):
        StrictMode.verify_advertising_data_pdu(component)
        StrictMode.verify_adv_struct_occurrences(component.adv_data, context='SRD')

        if component.header.rx_add.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

        if component.header.ch_sel.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

    @ParsePolicyInterface.verify.register(ScanReq)
    def verify_adv_scan_req_pdu(self, component: ScanReq):
        ParsePolicyInterface.verify_pdu_header(component)
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

        if component.header.ch_sel.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")


    @ParsePolicyInterface.verify.register(AdvHeader)
    def verify_adv_header(self, component: AdvHeader):
        if component.rfu.get_value(bin=True) != '0':
            raise ParseError(f"Reserved Bits must be 0")

        ParsePolicyInterface.verify_header_fields(component)


    @ParsePolicyInterface.verify.register(Packet)
    def verify_packet(self, component: Packet):
        current_crc = component.crc.get_value()
        target_crc = component.crc.from_pdu(component.pdu.to_string())

        if current_crc != target_crc:
            raise ParseError(f"CRC Mismatch")


