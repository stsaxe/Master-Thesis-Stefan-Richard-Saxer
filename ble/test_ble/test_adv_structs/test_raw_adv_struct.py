import io
import sys
import unittest

from ble import NormalMode
from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import RawAdvDataStruct

class TestRawAdvStruct(unittest.TestCase):
    def test_structure(self) -> None:
        struct = RawAdvDataStruct()
        self.assertIsInstance(struct.length, HexDataField)
        self.assertIsInstance(struct.type, HexDataField)
        self.assertIsInstance(struct.data, HexDataField)
        self.assertEqual(struct.data.get_name(), "Data")
        self.assertEqual(struct.data.get_value(), "")

        self.assertEqual(struct.type.get_value(), "")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 1)

        self.assertEqual(struct._get_registry_name(), "Raw Advertising Data")
        self.assertIsNone(struct._get_registry_key())

        self.assertEqual(struct.get_name(), "Adv Struct")

        struct._set_type()
        self.assertEqual(struct.type.get_value(), "")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": ["raw_advertising_data"]})

    def test_from_and_to_string(self):
        struct = RawAdvDataStruct()

        string = '0x03AA12AB'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x03AA12AB")
        self.assertEqual(struct.to_string(prefix=False), "03AA12AB")
        self.assertEqual(struct.type.get_value(), "AA")
        self.assertEqual(struct.length.get_value(), "03")
        self.assertEqual(struct.data.get_value(), "12AB")

        self.assertEqual(struct.get_length(), 4)
        self.assertEqual(struct.get_length(bit=True), 32)




    def test_pars_mode_tolerant(self):
        struct = RawAdvDataStruct()

        string = '0x03AA12ABC'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x03AA12AB")

        string = '0x09AA12AB'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x09AA12AB")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x03AA12AB")

        with self.assertRaises(ValueError):
            struct = RawAdvDataStruct()
            string = '0x03AA12AX'
            struct.from_string(string, parse_mode="tolerant")


    def test_parse_mode_strict_and_normal(self):
        struct = RawAdvDataStruct()

        with self.assertRaises(ValueError):
            string = '0x03AA12AX'
            struct.from_string(string, parse_mode="normal")

        with self.assertRaises(ParseError):
            string = '0x03AA12ABC'
            struct.from_string(string, parse_mode="normal")


        with self.assertRaises(ParseError):
            string = '0x03AA12ABC'
            struct.from_string(string, parse_mode=NormalMode())


        with self.assertRaises(ParseError):
            string = '0x09AA12AB'
            struct.from_string(string, parse_mode="normal")

        with self.assertRaises(ValueError):
            string = '0x03AA12AX'
            struct.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = '0x09AA12AB'
            struct.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = '0x03AA12ABC'
            struct.from_string(string, parse_mode="strict")

    def test_update(self):
        struct = RawAdvDataStruct()

        string = '0x03AA12AB'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x03AA12AB")
        struct.length.set_value("09")
        self.assertEqual(struct.to_string(prefix=True), "0x09AA12AB")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x03AA12AB")


    def test_print(self):
        struct = RawAdvDataStruct()

        string = '0x03AA12AB'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)

        self.assertEqual(2 * "\t" + "Adv Struct: Raw Advertising Data\n" +
                         3 * "\t" + "Length: 0x03\n" +
                         3 * "\t" + "Type: 0xAA\n" +
                         3 * "\t" + "Data: 0x12AB\n",
                         capturedOutput.getvalue())



