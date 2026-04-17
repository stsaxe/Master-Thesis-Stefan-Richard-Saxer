import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import ADVERTISING_REGISTRY, ManufacturerSpecific, CompleteLocalName, ShortenedLocalName


class TestLocalNameAdvStruct(unittest.TestCase):
    def test_structure_shortened(self) -> None:
        struct = ShortenedLocalName()
        self.assertIsInstance(struct.device_name, HexDataField)
        self.assertEqual(struct.device_name.get_name(), "Device Name")
        self.assertEqual(struct.device_name.get_value(), "")


        self.assertEqual(struct.type.get_value(), "08")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Shortened Local Name")
        self.assertEqual(struct._get_registry_key(), 0x08)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'shortened_local_name','08', '0x08']})

    def test_structure_complete(self) -> None:
        struct = CompleteLocalName()
        self.assertIsInstance(struct.device_name, HexDataField)
        self.assertEqual(struct.device_name.get_name(), "Device Name")
        self.assertEqual(struct.device_name.get_value(), "")

        self.assertEqual(struct.type.get_value(), "09")
        self.assertEqual(struct.length.get_value(), "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Complete Local Name")
        self.assertEqual(struct._get_registry_key(), 0x09)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": ['complete_local_name', '09', '0x09']})

    def test_from_and_to_string(self):
        struct = ShortenedLocalName()

        string = '0x0408AB1234'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x0408AB1234")
        self.assertEqual(struct.to_string(prefix=False), "0408AB1234")
        self.assertEqual(struct.type.get_value(), "08")
        self.assertEqual(struct.length.get_value(), "04")
        self.assertEqual(struct.device_name.get_value(), "AB1234")

        self.assertEqual(struct.get_length(), 5)
        self.assertEqual(struct.get_length(bit=True), 5*8)

    def test_pars_mode_tolerant(self):
        struct = CompleteLocalName()

        # parse valid string
        string = '0x0409AB1234'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0409AB1234")

        # update incorrect length and type and exceeding length bits
        string = '0x09AAAB1234F'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x09AAAB1234")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0409AB1234")

        # invalid hex characters
        with self.assertRaises(ValueError):
            struct = CompleteLocalName()
            string = '0x0409AB123X'
            struct.from_string(string, parse_mode="tolerant")

    def test_parse_mode_strict_and_normal(self):
        struct = CompleteLocalName()

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x0909AB1234'
            struct.from_string(string, parse_mode="normal")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x0412AB1234'
            struct.from_string(string, parse_mode="normal")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x0409AB1234C'
            struct.from_string(string, parse_mode="normal")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x0409AB123X'
            struct.from_string(string, parse_mode="normal")

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x0209AB1234'
            struct.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x04012AB1234'
            struct.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x0409AB1234C'
            struct.from_string(string, parse_mode="strict")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x0409AB123X'
            struct.from_string(string, parse_mode="strict")

    def test_update(self):
        struct = ShortenedLocalName()

        string = '0x0408AB1234'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x0408AB1234")
        struct.length.set_value("09")
        struct.type.set_value("AA")
        self.assertEqual(struct.to_string(prefix=True), "0x09AAAB1234")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0408AB1234")


    def test_print(self):
        struct = ShortenedLocalName()

        string = '06084170706c65'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)

        self.assertEqual(2 * "\t" + "Adv Struct: Shortened Local Name\n" +
                         3 * "\t" + "Length: 0x06\n" +
                         3 * "\t" + "Type: 0x08\n" +
                         3 * "\t" + "Device Name: Apple\n",
                         capturedOutput.getvalue())

    def test_occurrences(self):

        target = {'EIR': 'C1', 'AD': 'C1', 'SRD': 'C1', 'ACAD': 'X', 'OOB': 'C1'}

        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x08), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x09), target)


