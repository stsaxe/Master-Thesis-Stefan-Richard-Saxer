import unittest

from ble import RawAdvDataStruct, TxPowerLevel, Flags, ScanRsp, HelperMethods, ScanReq, ManufacturerSpecific
from ble.components.pdu.AdvHeader import AdvHeader
from ble.components.pdu.AdvertisingPDUs import AdvInd, AdvNonConnInd, AdvScanInd
from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.MultiNodeContainer import MultiNodeContainer


class TestAdvertisingDataPDUs(unittest.TestCase):
    adv_data_pdus = [AdvInd(), AdvNonConnInd(), AdvScanInd(), ScanReq()]

    def test_structure_adv_ind(self):
        pdu = AdvInd()
        pdu.print()

        self.assertIsInstance(pdu.header, AdvHeader)

        header = pdu.header

        self.assertEqual(header.pdu_type.get_value(bin=True), "0000")
        self.assertEqual(header.rx_add.get_value(bin=True), "")
        self.assertEqual(header.tx_add.get_value(bin=True), "")
        self.assertEqual(header.ch_sel.get_value(bin=True), "")
        self.assertEqual(header.rfu.get_value(bin=True), "")
        self.assertEqual(header.length.get_value(), "")

        self.assertEqual(pdu.header.get_length(bit=True), 4)

        self.assertIsInstance(pdu.advertising_address, HexDataField)
        self.assertEqual(pdu.advertising_address.get_value(), "")

        self.assertIsInstance(pdu.adv_data, MultiNodeContainer)
        self.assertEqual(pdu.adv_data.get_name(), "Advertising Data")
        self.assertEqual(len(pdu.adv_data), 0)

        self.assertEqual(pdu.header.rx_add.get_name(), "RFU")

    def test_structure_adv_non_conn_ind(self):
        pdu = AdvNonConnInd()
        self.assertEqual(pdu.header.rx_add.get_name(), "RFU")
        self.assertEqual(pdu.header.ch_sel.get_name(), "RFU")

        self.assertEqual(pdu._get_registry_name(), "ADV_NONCONN_IND")
        self.assertEqual(pdu._get_registry_key(), 0b0010)

        segment = pdu.get_path_segment()
        self.assertEqual(segment.keys, {"pdu_type": ["adv_nonconn_ind", "0b0010", "0010"]})

    def test_structure_adv_scan_ind(self):
        pdu = AdvScanInd()
        self.assertEqual(pdu.header.rx_add.get_name(), "RFU")
        self.assertEqual(pdu.header.ch_sel.get_name(), "RFU")

        self.assertEqual(pdu._get_registry_name(), "ADV_SCAN_IND")
        self.assertEqual(pdu._get_registry_key(), 0b0110)

        segment = pdu.get_path_segment()
        self.assertEqual(segment.keys, {"pdu_type": ["adv_scan_ind", "0b0110", "0110"]})

    def test_structure_scan_rsp(self):
        pdu = ScanRsp()
        self.assertEqual(pdu.header.rx_add.get_name(), "RFU")
        self.assertEqual(pdu.header.ch_sel.get_name(), "RFU")

        self.assertEqual(pdu._get_registry_name(), "SCAN_RSP")
        self.assertEqual(pdu._get_registry_key(), 0b0100)
        self.assertEqual(pdu.adv_data.get_name(), "Scan Response Data")

        segment = pdu.get_path_segment()
        self.assertEqual(segment.keys, {"pdu_type": ["scan_rsp", "0b0100", "0100"]})

    def test_getters(self):
        pdu = AdvInd()

        self.assertEqual(pdu._get_registry_name(), "ADV_IND")
        self.assertEqual(pdu._get_registry_key(), 0b0000)

        segment = pdu.get_path_segment()

        self.assertEqual(segment.name, "pdu")
        self.assertEqual(segment.keys, {"pdu_type": ["adv_ind", "0b0000", "0000"]})

    def test_from_and_to_string(self):
        string = ("600C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        self.assertEqual(pdu.get_length(), len(string) // 2)
        self.assertEqual(pdu.get_length(bit=True), len(string) * 2 * 2)

        self.assertEqual(pdu.header.to_string(), "600C")
        self.assertEqual(pdu.advertising_address.to_string(), "78563412CDAB")

        for idx, adv in enumerate(pdu.adv_data):
            if idx == 1:
                self.assertIsInstance(adv, TxPowerLevel)
                self.assertEqual(adv.to_string(), "020ACB")

            elif idx == 0:
                self.assertIsInstance(adv, Flags)
                self.assertEqual(adv.to_string(), "02011A")

    def test_from_and_to_string_raw_pdu(self):
        string = ("600C" + "ABCD12345678" + '02011A' + '02BBCB')

        pdu = AdvInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)

        self.assertEqual(pdu.get_length(), len(string) // 2)
        self.assertEqual(pdu.get_length(bit=True), len(string) * 2 * 2)

        for idx, adv in enumerate(pdu.adv_data):
            if idx == 1:
                self.assertIsInstance(adv, RawAdvDataStruct)

    def test_update(self):
        string = ("600C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)

        pdu.header.pdu_type.set_value("1111", bin=True)
        pdu.header.length.set_value("CD")

        for idx, adv in enumerate(pdu.adv_data):
            adv.length.set_value("AB")
            adv.type.set_value("FF")

        pdu.header.rfu.set_value("1", bin=True)

        pdu.update()
        self.assertEqual(pdu.to_string(), string)

    def test_parse_mode_strict_occurrences(self):

        header = HelperMethods.bin_to_hex("00000000", pad="byte")

        advertising_address = "ABCD12345678"
        flags = "020103"
        tx_power_level = "020A23"
        manufacturer_specific = "06FFABCD123456"
        raw_adv = "04ABCD1234"

        pay_load = advertising_address + flags + manufacturer_specific + flags + tx_power_level
        length = HelperMethods.int_to_hex(len(pay_load) // 2, pad="byte")
        pdu_string = header + length + pay_load

        # flags appear to often
        with self.assertRaises(ParseError):
            pdu = AdvInd()
            pdu.from_string(pdu_string, parse_mode="strict")

        # tx power level not allowed
        header = HelperMethods.bin_to_hex("00000100", pad="byte")
        pay_load = advertising_address + flags + manufacturer_specific + tx_power_level
        length = HelperMethods.int_to_hex(len(pay_load) // 2, pad="byte")
        pdu_string = header + length + pay_load

        with self.assertRaises(ParseError):
            pdu = ScanRsp()
            pdu.from_string(pdu_string, parse_mode="strict")

    def test_parse_mode_strict_over_length(self):
        header = HelperMethods.bin_to_hex("00000000", pad="byte")

        a_32_byte_adv_struct = "1FFF"
        for i in range(30):
            a_32_byte_adv_struct += "AB"

        advertising_address = "ABCD12345678"

        assert len(a_32_byte_adv_struct) // 2 == 32

        pay_load = advertising_address + a_32_byte_adv_struct
        length = HelperMethods.int_to_hex(len(pay_load) // 2, pad="byte")

        # the struct needs to be one byte longer than the limit (0x26 is 38 bytes)
        assert length == "26"

        pdu_string = header + length + pay_load

        with self.assertRaises(ParseError):
            pdu = AdvInd()
            pdu.from_string(pdu_string, parse_mode="strict")

    def test_parse_mode_normal_adv_ind(self):
        string = ("600C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)


        # verify that header bits are ignored
        string = ("A00C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        with self.assertRaises(ParseError):
            string = ("60AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # wrong type
        with self.assertRaises(ParseError):
            string = ("6A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("600C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string)

        with self.assertRaises(ParseError):
            string = ("600C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string)

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("600C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string)
        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("600D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string)

    def test_parse_mode_normal_adv_non_conn_ind(self):
        string = ("620C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvNonConnInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # verify that header bits are ignored
        string = ("A20C" + "ABCD12345678" + '02011A' + '020ACB')
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        with self.assertRaises(ParseError):
            string = ("62AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # wrong type
        with self.assertRaises(ParseError):
            string = ("6A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("620C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string)

        with self.assertRaises(ParseError):
            string = ("620C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string)

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("620C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string)
        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("620D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string)

    def test_parse_mode_normal_adv_scan_ind(self):
        string = ("660C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvScanInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # verify that header bits are ignored
        string = ("A60C" + "ABCD12345678" + '02011A' + '020ACB')
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        with self.assertRaises(ParseError):
            string = ("66AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # wrong type
        with self.assertRaises(ParseError):
            string = ("6A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("660C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string)

        with self.assertRaises(ParseError):
            string = ("660C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string)

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("660C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string)
        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("660D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string)

    def test_parse_mode_normal_scan_rsp(self):
        string = ("640C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = ScanRsp()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # verify that header bits are ignored
        string = ("A40C" + "ABCD12345678" + '02011A' + '020ACB')
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        with self.assertRaises(ParseError):
            string = ("64AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # wrong type
        with self.assertRaises(ParseError):
            string = ("6A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string)

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("640C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string)

        with self.assertRaises(ParseError):
            string = ("640C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string)

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("660C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string)

        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("660D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string)

    def test_parse_mode_strict_adv_ind(self):
        string = ("600C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvInd()

        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # verify that header flags are checked
        string = ("ABCD12345678" + '02011A' + '020ACB')
        header = AdvHeader()
        header.from_string("000C")
        header.rx_add.set_value("1", bin=True)
        header.tx_add.set_value("1", bin=True)
        header.ch_sel.set_value("1", bin=True)
        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")

        # wrong length
        with self.assertRaises(ParseError):
            string = ("60AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = ("6A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("600C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = ("600C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string, parse_mode="strict")

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("600C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("600D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string, parse_mode="strict")

    def test_parse_mode_strict_adv_non_conn_ind(self):
        string = ("020C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvNonConnInd()
        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # verify that header flags are checked
        string = ("ABCD12345678" + '02011A' + '020ACB')
        header = AdvHeader()
        header.from_string("020C")
        header.rx_add.set_value("1", bin=True)
        header.ch_sel.set_value("0", bin=True)

        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")

        header.rx_add.set_value("0", bin=True)
        header.ch_sel.set_value("1", bin=True)

        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")

        # wrong length
        with self.assertRaises(ParseError):
            string = ("02AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = ("0A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("020C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = ("020C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string, parse_mode="strict")

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("020C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")
        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("020D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string, parse_mode="strict")


    def test_parse_mode_strict_adv_scan_ind(self):
        string = ("060C" + "ABCD12345678" + '02011A' + '020ACB')

        pdu = AdvScanInd()
        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # verify that header flags are checked
        string = ("ABCD12345678" + '02011A' + '020ACB')
        header = AdvHeader()
        header.from_string("060C")
        header.rx_add.set_value("1", bin=True)
        header.ch_sel.set_value("0", bin=True)

        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")

        header.rx_add.set_value("0", bin=True)
        header.ch_sel.set_value("1", bin=True)

        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")

        # wrong length
        with self.assertRaises(ParseError):
            string = ("06AC" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = ("0A0C" + "ABCD12345678" + '02011A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("060C" + "ABCD12345678" + '02011A' + '020ACBAA')
            pdu.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = ("060C" + "ABCD12345678" + '02011A' + '020ACBA')
            pdu.from_string(string, parse_mode="strict")

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("060C" + "ABCD12345678" + '02031A' + '020ACB')
            pdu.from_string(string, parse_mode="strict")
        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("060D" + "ABCD12345678" + '02011A' + '07FFABCD')
            pdu.from_string(string, parse_mode="strict")


    def test_parse_mode_strict_scan_rsp(self):
        string = ("040D" + "ABCD12345678" + '03FFABCD' + '020ACB')

        pdu = ScanRsp()
        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)


        # verify that header flags are checked
        string = ("ABCD12345678" + '03FFABCD' + '020ACB')
        header = AdvHeader()
        header.from_string("040D")
        header.ch_sel.set_value("1", bin=True)

        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")


        string = ("ABCD12345678" + '03FFABCD' + '020ACB')
        header = AdvHeader()
        header.from_string("040D")
        header.rx_add.set_value("1", bin=True)
        with self.assertRaises(ParseError):
            pdu.from_string(header.to_string() + string, parse_mode="strict")

        # wrong length
        with self.assertRaises(ParseError):
            string = ("04AD" + "ABCD12345678" + '03FFABCD' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = ("0A0D" + "ABCD12345678" + '03FFABCD' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # exceeding bit
        with self.assertRaises(ParseError):
            string = ("040D" + "ABCD12345678" + '03FFABCD' + '020ACBAA')
            pdu.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = ("040D" + "ABCD12345678" + '03FFABCD' + '020ACBA')
            pdu.from_string(string, parse_mode="strict")

        # parse error in adv struct, "incorrect type"
        with self.assertRaises(ParseError):
            string = ("060D" + "ABCD12345678" + '03FFABCD' + '020ACB')
            pdu.from_string(string, parse_mode="strict")

        # run out of bits to read, final struct indicates longer length than available
        with self.assertRaises(ParseError):
            string = ("060E" + "ABCD12345678" + '03FFABCD' + '07FFABCD')
            pdu.from_string(string, parse_mode="strict")


    def test_parse_mode_tolerant_adv_ind(self):
        valid_string = "600C" + "ABCD12345678" + '02011A' + '020ACB'

        # test base case
        pdu = AdvInd()
        pdu.from_string(valid_string, parse_mode='strict')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)


        # test with raw pdu
        valid_string = "600C" + "ABCD12345678" + '02BB1A' + '020ACB'
        pdu.from_string(valid_string, parse_mode='tolerant')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)

        # test with raw pdu with length of pdu too short
        valid_string = "600C" + "ABCD12345678" + '020111' + '05AAAA'
        pdu.from_string(valid_string, parse_mode='tolerant')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)

        # test with valid pdu with invalid length of pdu (too short)
        valid_string = "600C" + "ABCD12345678" + '09FF1A05BBCB'
        pdu.from_string(valid_string, parse_mode='tolerant')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)

        for adv_struct in pdu.adv_data:
            self.assertIsInstance(adv_struct, ManufacturerSpecific)

        # test with valid pdu with invalid length (too short)
        valid_string = "600B" + "ABCD12345678" + '020111' + '05FF'
        pdu.from_string(valid_string, parse_mode='tolerant')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)

        for idx, adv_struct in enumerate(pdu.adv_data):
            if idx == 1:
                self.assertIsInstance(adv_struct, RawAdvDataStruct)

        # test with valid pdu with invalid pdu at the end, needs to be converted to raw pdu
        valid_string = "600C" + "ABCD12345678" + '020111' + '01FF'
        pdu.from_string(valid_string, parse_mode='tolerant')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)

        for idx, adv_struct in enumerate(pdu.adv_data):
            if idx == 1:
                self.assertIsInstance(adv_struct, RawAdvDataStruct)

    def test_parse_mode_tolerant_update_adv_ind(self):
        pdu = AdvInd()

        valid_string = "F20D" + "ABCD12345678" + '020111' + '03FF'
        pdu.from_string(valid_string, parse_mode='tolerant')
        self.assertEqual(pdu.to_string(), valid_string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)

        pdu.update()
        target_string = "600B" + "ABCD12345678" + '020111' + '01FF'
        self.assertEqual(pdu.to_string(prefix=True), "0x" + target_string)



