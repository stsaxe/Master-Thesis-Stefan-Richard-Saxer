import unittest
from unittest import TestCase

from ble.components.pdu.AdvHeader import AdvHeader
from ble.components.pdu.AdvertisingPDUs import RawPDU
from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField

class TestRawPDU(unittest.TestCase):
    def test_structure(self):
        pdu = RawPDU()

        self.assertIsInstance(pdu.header, AdvHeader)

        header = pdu.header

        self.assertEqual(header.pdu_type.get_value(bin=True), "")
        self.assertEqual(header.rx_add.get_value(bin=True), "")
        self.assertEqual(header.tx_add.get_value(bin=True), "")
        self.assertEqual(header.ch_sel.get_value(bin=True), "")
        self.assertEqual(header.rfu.get_value(bin=True), "")
        self.assertEqual(header.length.get_value(), "")

        self.assertEqual(pdu.header.get_length(bit=True), 0)

        self.assertIsInstance(pdu.data, HexDataField)
        self.assertEqual(pdu.data.get_value(), "")
        self.assertEqual(pdu.data.get_name(), "Raw Data")

    def test_getters(self):
        pdu = RawPDU()

        self.assertEqual(pdu._get_registry_name(), "Raw PDU")
        self.assertEqual(pdu._get_registry_key(), None)

        segment = pdu.get_path_segment()

        self.assertEqual(segment.name, "pdu")
        self.assertEqual(segment.keys, {"pdu_type": ["raw_pdu"]})

    def test_update(self):
        string = ("6FAB" + "ABCD12345678" + 'AABBCCDDEE11')
        valid_string = ("6F0C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = RawPDU()
        pdu.from_string(string, parse_mode="tolerant")
        self.assertEqual(pdu.to_string(), string)
        pdu.update()
        self.assertEqual(pdu.to_string(prefix=True), "0x" + valid_string)


    def test_to_and_from_string(self):
        string = ("6F0C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = RawPDU()
        pdu.from_string(string)
        self.assertEqual(pdu.to_string(), string)
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)
        self.assertEqual(pdu.to_string(prefix=False), string)

        self.assertEqual(pdu.header.to_string(prefix=True), "0x" + "6F0C")
        self.assertEqual(pdu.data.to_string(prefix=True), "0x" +  "ABCD12345678" + 'AABBCCDDEE11')

        self.assertEqual(pdu.get_length(), len(string) // 2)
        self.assertEqual(pdu.get_length(bit=True), len(string) * 2 * 2)

    def test_parse_mode_normal(self):
        string = ("6F0C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = RawPDU()
        pdu.from_string(string, parse_mode="normal")
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        with self.assertRaises(ParseError):
            string = ("6FCC" + "ABCD12345678" + 'AABBCCDDEE11')
            pdu.from_string(string, parse_mode="normal")

        # too short
        with self.assertRaises(ParseError):
            string = ("6F")
            pdu.from_string(string, parse_mode="normal")

        # exceeding bits
        with self.assertRaises(ParseError):
            string += "A"
            pdu.from_string(string, parse_mode="normal")

    def test_parse_mode_tolerant(self):
        string = ("6F0C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = RawPDU()
        pdu.from_string(string, parse_mode="normal")
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length and exceeding bit
        string = ("6FCC" + "ABCD12345678" + 'AABBCCDDEE11B')
        pdu.from_string(string, parse_mode="tolerant")
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string[:-1])

    def test_parse_mode_strict(self):
        string = ("6F0C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = RawPDU()
        pdu.from_string(string, parse_mode="strict")
        self.assertEqual(pdu.to_string(prefix=True), "0x" + string)

        # wrong length
        with self.assertRaises(ParseError):
            string = ("6FCC" + "ABCD12345678" + 'AABBCCDDEE11')
            pdu.from_string(string, parse_mode="strict")

        # too short
        with self.assertRaises(ParseError):
            string = ("6F")
            pdu.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string += "A"
            pdu.from_string(string, parse_mode="strict")


    def test_print(self):
        string = ("6F0C" + "ABCD12345678" + 'AABBCCDDEE11')

        pdu = RawPDU()
        pdu.from_string(string, parse_mode="normal")

        pdu.print()





