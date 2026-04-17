import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import ADVERTISING_REGISTRY, Appearance


class TestManufacturerSpecificAdvStruct(unittest.TestCase):
    def test_structure(self) -> None:
        struct = Appearance()
        self.assertIsInstance(struct.appearance, HexDataField)
        self.assertEqual(struct.appearance.get_name(), "Appearance")
        self.assertEqual(struct.appearance.get_value(), "")

        self.assertEqual(struct.type.get_value(), "19")
        self.assertEqual(struct.length.get_value(), "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Appearance")
        self.assertEqual(struct._get_registry_key(), 0x19)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": ['appearance', '19', '0x19']})

    def test_from_and_to_string(self):
        struct = Appearance()

        string = '0x0319ABCD'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x0319ABCD")
        self.assertEqual(struct.to_string(prefix=False), "0319ABCD")
        self.assertEqual(struct.type.get_value(), "19")
        self.assertEqual(struct.length.get_value(), "03")
        self.assertEqual(struct.appearance.get_value(), "CDAB")

        self.assertEqual(struct.get_length(), 4)
        self.assertEqual(struct.get_length(bit=True), 4 * 8)

    def test_pars_mode_tolerant(self):
        struct = Appearance()

        # parse valid string
        string = '0x0319ABCD'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0319ABCD")

        # update incorrect length and type and exceeding length bits
        string = '0x01EEABCD1'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x01EEABCD")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0319ABCD")

        # invalid hex characters
        with self.assertRaises(ValueError):
            struct = Appearance()
            string = '0x0319AXCD'
            struct.from_string(string, parse_mode="tolerant")

    def test_parse_mode_strict_and_normal(self):
        struct = Appearance()

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x0119ABCD'
            struct.from_string(string, parse_mode="normal")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x03FFABCD'
            struct.from_string(string, parse_mode="normal")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x0319ABCDE'
            struct.from_string(string, parse_mode="normal")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x0319XXCD'
            struct.from_string(string, parse_mode="normal")

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x0519ABCD'
            struct.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x0311ABCD'
            struct.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x0319ABCDE'
            struct.from_string(string, parse_mode="strict")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x0319XXCD'
            struct.from_string(string, parse_mode="strict")

    def test_update(self):
        struct = Appearance()

        string = '0x0319ABCD'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x0319ABCD")
        struct.length.set_value("09")
        struct.type.set_value("AA")
        self.assertEqual(struct.to_string(prefix=True), "0x09AAABCD")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0319ABCD")

    def test_print(self):
        struct = Appearance()

        string = '0x0319ABCD'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident=2)

        self.assertEqual(2 * "\t" + "Adv Struct: Appearance\n" +
                         3 * "\t" + "Length: 0x03\n" +
                         3 * "\t" + "Type: 0x19\n" +
                         3 * "\t" + "Appearance: 0xCDAB\n",
                         capturedOutput.getvalue())

    def test_occurrences(self):
        target = {'EIR': 'X', 'AD': 'C2', 'SRD': 'C2', 'ACAD': 'X', 'OOB': 'C1'}
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x19), target)
