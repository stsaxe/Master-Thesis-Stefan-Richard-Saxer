import unittest

from ble.components.pdu.AdvHeader import AdvHeader
from ble.components.pdu.AdvertisingPDUs import AdvDirectInd, ScanReq
from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField


class TestTargetAddrPDUs(unittest.TestCase):
    def test_structure_adv_ind(self):
        pdu = AdvDirectInd()

        self.assertIsInstance(pdu.header, AdvHeader)

        header = pdu.header

        self.assertEqual(header.pdu_type.get_value(bin=True), "0001")
        self.assertEqual(header.rx_add.get_value(bin=True), "")
        self.assertEqual(header.tx_add.get_value(bin=True), "")
        self.assertEqual(header.ch_sel.get_value(bin=True), "")
        self.assertEqual(header.rfu.get_value(bin=True), "")
        self.assertEqual(header.length.get_value(), "")

        self.assertEqual(pdu.header.get_length(bit=True), 4)

        self.assertIsInstance(pdu.advertising_address, HexDataField)
        self.assertEqual(pdu.advertising_address.get_value(), "")

        self.assertIsInstance(pdu.target_address, HexDataField)
        self.assertEqual(pdu.target_address.get_value(), "")


    def test_structure_adv_non_conn_ind(self):
        pdu = ScanReq()

        self.assertEqual(pdu.header.ch_sel.get_name(), "RFU")

        self.assertEqual(pdu._get_registry_name(), "SCAN_REQ")
        self.assertEqual(pdu._get_registry_key(), 0b0011)

        segment = pdu.get_path_segment()
        self.assertEqual(segment.keys, {"pdu_type": ["scan_req", "0b0011", "0011"]})



    def test_getters(self):
        pdu = AdvDirectInd()

        self.assertEqual(pdu._get_registry_name(), "ADV_DIRECT_IND")
        self.assertEqual(pdu._get_registry_key(), 0b0001)

        segment = pdu.get_path_segment()

        self.assertEqual(segment.name, "pdu")
        self.assertEqual(segment.keys, {"pdu_type": ["adv_direct_ind", "0b0001", "0001"]})

    def test_from_and_to_string(self):
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = AdvDirectInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        self.assertEqual(pdu.get_length(), len(string) // 2)
        self.assertEqual(pdu.get_length(bit=True), len(string) * 2 * 2)

        self.assertEqual(pdu.header.to_string(), "610C")
        self.assertEqual(pdu.advertising_address.to_string(), "78563412CDAB")
        self.assertEqual(pdu.target_address.to_string(), "11EEDDCCBBAA")



    def test_update(self):
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = AdvDirectInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)

        pdu.header.pdu_type.set_value("1010", bin=True)
        pdu.header.length.set_value("CD")
        pdu.header.rfu.set_value("1", bin=True)

        pdu.update()
        self.assertEqual(pdu.to_string(), string)


    def test_parse_mode_normal(self):
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = AdvDirectInd()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        string = ("61AA" + "ABCD12345678" + 'AABBCCDDEE11')
        with self.assertRaises(ParseError):
            pdu.from_string(string)

        # wrong type
        string = ("600C" + "ABCD12345678" + 'AABBCCDDEE11')
        with self.assertRaises(ParseError):
            pdu.from_string(string)

        # exceeding pdu bytes:
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11' + "AA")
        with self.assertRaises(ParseError):
            pdu.from_string(string)

        # to few pdu bytes:
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE')
        with self.assertRaises(ParseError):
            pdu.from_string(string)

        # invalid hex characters:
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEEXX')
        with self.assertRaises(ValueError):
            pdu.from_string(string)

        # non reserved header Bit
        string = ("E10C" + "ABCD12345678" + 'AABBCCDDEE11')
        pdu.from_string(string)


        string = ("E30C" + "ABCD12345678" + 'AABBCCDDEE11')
        full_valid_string = ("C30C" + "ABCD12345678" + 'AABBCCDDEE11')
        pdu = ScanReq()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)

        pdu.update()
        self.assertEqual(pdu.to_string(), full_valid_string)



    def test_parse_mode_strict(self):
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = AdvDirectInd()
        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        string = ("61AA" + "ABCD12345678" + 'AABBCCDDEE11')
        with self.assertRaises(ParseError):
            pdu.from_string(string, parse_mode="strict")

        # wrong type
        string = ("600C" + "ABCD12345678" + 'AABBCCDDEE11')
        with self.assertRaises(ParseError):
            pdu.from_string(string, parse_mode="strict")

        # exceeding pdu bytes:
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11' + "AA")
        with self.assertRaises(ParseError):
            pdu.from_string(string, parse_mode="strict")

        # to few pdu bytes:
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE')
        with self.assertRaises(ParseError):
            pdu.from_string(string, parse_mode="strict")

        # invalid hex characters:
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEEXX')
        with self.assertRaises(ValueError):
            pdu.from_string(string, parse_mode="strict")

        # non reserved header Bit
        string = ("E10C" + "ABCD12345678" + 'AABBCCDDEE11')
        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(), string)

        pdu = ScanReq()

        with self.assertRaises(ParseError):
            string = ("E30C" + "ABCD12345678" + 'AABBCCDDEE11')
            pdu.from_string(string, parse_mode="strict")


    def test_parse_mode_tolerant(self):
        valid_string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11')


        pdu = AdvDirectInd()
        string = ("610C" + "ABCD12345678" + 'AABBCCDDEE11')
        pdu.from_string(string, parse_mode="tolerant")
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(), string)
        pdu.update()
        self.assertEqual(pdu.to_string(), valid_string)


        #incorect type, length, reserved bit set and exceeding bytes
        pdu = ScanReq()

        valid_string = ("C30C" + "ABCD12345678" + 'AABBCCDDEE11')
        string = ("E9FF" + "ABCD12345678" + 'AABBCCDDEE11CC')

        pdu.from_string(string, parse_mode="tolerant")
        self.assertEqual(pdu.to_string(), string[:-2])

        pdu.update()
        self.assertEqual(pdu.to_string(), valid_string)



