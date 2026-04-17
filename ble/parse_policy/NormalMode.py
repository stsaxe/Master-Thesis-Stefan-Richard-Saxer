from typing import override

from ble import RawPDU, ScanReq, ScanRsp, AdvScanInd, AdvNonConnInd, AdvDirectInd, AdvInd
from ble.components.advertising_data.AdvertisingData import Flags
from ble.components.pdu.AbstractAdvDataPDU import AbstractAdvDataPDU
from ble.components.pdu.AdvHeader import AdvHeader
from ble.errors.ParseError import ParseError
from ble.parse_policy.BaseParsePolicy import BaseParsePolicy
from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface
from ble.parse_policy.ParseRegistry import PARSE_POLICY_REGISTRY


@PARSE_POLICY_REGISTRY.register("normal")
class NormalMode(BaseParsePolicy):
    @ParsePolicyInterface.verify.register(Flags)
    @override
    def verify_flags_adv_struct(self, component: Flags):
        ParsePolicyInterface.verify_adv_data_type(component)
        ParsePolicyInterface.verify_adv_data_length(component)

        fields = [component.bit_0, component.bit_1, component.bit_2, component.bit_3, component.bit_4]

        for field in fields:
            if field.get_bit_length() != 1:
                raise ParseError(f"Field {field.get_name()} must have length 1 Bit")

        if component.bit_5_to_7.get_bit_length() != 3:
            raise ParseError(f"Field {component.bit_5_to_7.get_name()} must have length 3 Bit")

    @staticmethod
    def verify_advertising_data_pdu(component: AbstractAdvDataPDU):
        ParsePolicyInterface.verify_pdu_header(component)
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

    @ParsePolicyInterface.verify.register(AdvInd)
    def verify_adv_ind_pdu(self, component: AdvInd):
        NormalMode.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(AdvDirectInd)
    def verify_adv_direct_ind_pdu(self, component: AdvDirectInd):
        ParsePolicyInterface.verify_pdu_header(component)
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

    @ParsePolicyInterface.verify.register(AdvNonConnInd)
    def verify_adv_non_conn_ind_pdu(self, component: AdvNonConnInd):
        NormalMode.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(AdvScanInd)
    def verify_adv_scan_ind_pdu(self, component: AdvScanInd):
        NormalMode.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(ScanRsp)
    def verify_adv_scan_rsp_pdu(self, component: ScanRsp):
        NormalMode.verify_advertising_data_pdu(component)

    @ParsePolicyInterface.verify.register(ScanReq)
    def verify_adv_scan_req_pdu(self, component: ScanReq):
        ParsePolicyInterface.verify_pdu_header(component)
        ParsePolicyInterface.verify_hex_field_length(component.advertising_address, 6, byte=True)

    @ParsePolicyInterface.verify.register(AdvHeader)
    def verify_adv_header(self, component: AdvHeader):
        ParsePolicyInterface.verify_header_fields(component)






