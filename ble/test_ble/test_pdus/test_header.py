import io
import struct
import sys
import unittest

from ble.fields.HexDataField import HexDataField
from ble.fields.BitDataField import BitDataField
from ble.components.pdu.AdvHeader import AdvHeader
from ble.errors.ParseError import ParseError


class TestAdvHeader(unittest.TestCase):
    def test_structure(self):
        header = AdvHeader()

        self.assertEqual(header.pdu_type.get_value(), "")
        self.assertEqual(header.pdu_type.get_name(), "PDU Type")
        self.assertIsInstance(header.pdu_type, BitDataField)


        self.assertEqual(header.ch_sel.get_value(), "")
        self.assertEqual(header.ch_sel.get_name(), "ChSel")
        self.assertIsInstance(header.ch_sel, BitDataField)

        self.assertEqual(header.rfu.get_value(), "")
        self.assertEqual(header.rfu.get_name(), "RFU")
        self.assertIsInstance(header.rfu, BitDataField)


        self.assertEqual(header.tx_add.get_value(), "")
        self.assertEqual(header.tx_add.get_name(), "TxAdd")
        self.assertIsInstance(header.tx_add, BitDataField)

        self.assertEqual(header.rx_add.get_value(), "")
        self.assertEqual(header.rx_add.get_name(), "RxAdd")
        self.assertIsInstance(header.rx_add, BitDataField)

        self.assertEqual(header.length.get_value(), "")
        self.assertEqual(header.length.get_name(), "Length")
        self.assertIsInstance(header.length, HexDataField)

        segment = header.get_path_segment()
        self.assertEqual(segment.name, "packet_header")
        self.assertEqual(segment.keys, dict())

    def test_from_and_to_string(self):
        string = "CA" + "1B"
        header = AdvHeader()
        header.from_string(string)

        self.assertEqual(header.pdu_type.get_value(bin=True), "1010")
        self.assertEqual(header.rx_add.get_value(bin=True), "1")
        self.assertEqual(header.tx_add.get_value(bin=True), "1")
        self.assertEqual(header.ch_sel.get_value(bin=True), "0")
        self.assertEqual(header.rfu.get_value(bin=True), "0")
        self.assertEqual(header.length.get_value(), "1B")

        self.assertEqual(header.to_string(), string)

        string = "C2" + "1B"
        header.from_string(string)
        self.assertEqual(header.pdu_type.get_value(bin=True), "0010")
        self.assertEqual(header.to_string(), string)

    def test_header_all_flags_zero(self):
        string = "00" + "00"
        header = AdvHeader()
        header.from_string(string)

        self.assertEqual(header.pdu_type.get_value(bin=True), "0000")
        self.assertEqual(header.rx_add.get_value(bin=True), "0")
        self.assertEqual(header.tx_add.get_value(bin=True), "0")
        self.assertEqual(header.ch_sel.get_value(bin=True), "0")
        self.assertEqual(header.rfu.get_value(bin=True), "0")
        self.assertEqual(header.length.get_value(), "00")



    def test_update(self):
        string = "CA" + "1B"
        header = AdvHeader()
        header.from_string(string)
        header.rfu.set_value("1")

        header.rx_add.set_name("RFU")
        header.ch_sel.set_name("RFU")

        header.update()

        self.assertEqual(header.pdu_type.get_value(bin=True), "1010")
        self.assertEqual(header.rx_add.get_value(bin=True), "0")
        self.assertEqual(header.tx_add.get_value(bin=True), "1")
        self.assertEqual(header.ch_sel.get_value(bin=True), "0")
        self.assertEqual(header.rfu.get_value(bin=True), "0")
        self.assertEqual(header.length.get_value(), "1B")

    def test_get_length(self):
        string = "CA" + "1B"
        header = AdvHeader()
        header.from_string(string)

        self.assertEqual(header.get_length(), 2)
        self.assertEqual(header.get_length(bit=True), 16)

        header.length.set_value("CD")
        header.pdu_type.set_value("1010", bin=True)

        self.assertEqual(header.get_length(), 2)
        self.assertEqual(header.get_length(bit=True), 16)

    def test_parse_mode_normal(self):

        # length too long
        with self.assertRaises(ParseError):
            string = "CA" + "1BC"

            header = AdvHeader()
            header.from_string(string, parse_mode="normal")

        # length too short
        with self.assertRaises(ParseError):
            string = "CA"

            header = AdvHeader()
            header.from_string(string, parse_mode="normal")

        # length too long, exceeding Bytes
        with self.assertRaises(ParseError):
            string = "CA" + "1BAC"

            header = AdvHeader()
            header.from_string(string, parse_mode="normal")

    def test_parse_mode_strict(self):
        # length too long
        with self.assertRaises(ParseError):
            string = "CA" + "1BC"

            header = AdvHeader()
            header.from_string(string, parse_mode="strict")

        # length too short
        with self.assertRaises(ParseError):
            string = "CA"

            header = AdvHeader()
            header.from_string(string, parse_mode="strict")

        # length too long, exceeding Bytes
        with self.assertRaises(ParseError):
            string = "CA" + "1BAC"

            header = AdvHeader()
            header.from_string(string, parse_mode="strict")

        # reserved bit set
        with self.assertRaises(ParseError):
            string = "F1234"

            header = AdvHeader()
            header.from_string(string, parse_mode="strict")

    def test_parse_mode_tolerant(self):
        string = "CA" + "1BC"

        header = AdvHeader()
        header.from_string(string, parse_mode="tolerant")
        self.assertEqual(header.to_string(), "CA1B")

        string = "CA" + "1BAC"
        header.from_string(string, parse_mode="tolerant")
        self.assertEqual(header.to_string(), "CA1B")

        with self.assertRaises(ParseError):
            string = "CA"
            header.from_string(string, parse_mode="tolerant")

    def test_print(self):
        string = "CA" + "1B"
        header = AdvHeader()
        header.from_string(string)

        header.print()

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        header._print(ident = 2)

        self.assertEqual(2 * "\t" + "Packet Header\n" +
                         3 * "\t" + "PDU Type: 0b1010\n" +
                         3 * "\t" + "RFU: 0b0\n" +
                         3 * "\t" + "ChSel: 0b0\n" +
                         3 * "\t" + "TxAdd: 0b1\n" +
                         3 * "\t" + "RxAdd: 0b1\n" +
                         3 * "\t" + "Length: 0x1B\n",
                         capturedOutput.getvalue())







