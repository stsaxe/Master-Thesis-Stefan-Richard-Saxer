import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import ADVERTISING_REGISTRY, ManufacturerSpecific


class TestManufacturerSpecificAdvStruct(unittest.TestCase):
    def test_structure(self) -> None:
        struct = ManufacturerSpecific()
        self.assertIsInstance(struct.company_id, HexDataField)
        self.assertEqual(struct.company_id.get_name(), "Company ID")
        self.assertEqual(struct.company_id.get_value(), "")

        self.assertIsInstance(struct.data, HexDataField)
        self.assertEqual(struct.data.get_name(), "Data")
        self.assertEqual(struct.data.get_value(), "")

        self.assertEqual(struct.type.get_value(), "FF")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Manufacturer Specific")
        self.assertEqual(struct._get_registry_key(), 0xFF)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'manufacturer_specific','ff', '0xff']})



    def test_from_and_to_string(self):
        struct = ManufacturerSpecific()

        string = '0x06FFAB12CD1234'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x06FFAB12CD1234")
        self.assertEqual(struct.to_string(prefix=False), "06FFAB12CD1234")
        self.assertEqual(struct.type.get_value(), "FF")
        self.assertEqual(struct.length.get_value(), "06")
        self.assertEqual(struct.data.get_value(), "CD1234")
        self.assertEqual(struct.company_id.get_value(), "12AB")

        self.assertEqual(struct.get_length(), 7)
        self.assertEqual(struct.get_length(bit=True), 7*8)

        struct = ManufacturerSpecific()

        string = '0x03FFAB12'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x03FFAB12")
        self.assertEqual(struct.to_string(prefix=False), "03FFAB12")
        self.assertEqual(struct.type.get_value(), "FF")
        self.assertEqual(struct.length.get_value(), "03")
        self.assertEqual(struct.data.get_value(), "")
        self.assertEqual(struct.company_id.get_value(), "12AB")

        self.assertEqual(struct.get_length(), 4)
        self.assertEqual(struct.get_length(bit=True), 4 * 8)

    def test_pars_mode_tolerant(self):
        struct = ManufacturerSpecific()

        # parse valid string
        string = '0x06FFAB12CD1234'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x06FFAB12CD1234")

        # update incorrect length and type and exceeding length bits
        string = '0x01AAAB12CD1234B'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x01AAAB12CD1234")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x06FFAB12CD1234")

        # invalid hex characters
        with self.assertRaises(ValueError):
            struct = ManufacturerSpecific()
            string = '0x06FFAB12CD12XX'
            struct.from_string(string, parse_mode="tolerant")



    def test_parse_mode_strict_and_normal(self):
        struct = ManufacturerSpecific()

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x03FFAB12CD1234'
            struct.from_string(string, parse_mode="normal")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x06FAAB12CD1234'
            struct.from_string(string, parse_mode="normal")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x06FFAB12CD1234C'
            struct.from_string(string, parse_mode="normal")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x06FFAB12CD1234X'
            struct.from_string(string, parse_mode="normal")

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x03FFAB12CD1234'
            struct.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x06FFAB12CD1234C'
            struct.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x06FAAB12CD1234'
            struct.from_string(string, parse_mode="strict")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x06FFAB12CD1234X'
            struct.from_string(string, parse_mode="strict")






    def test_update(self):
        struct = ManufacturerSpecific()

        string = '0x06FFAB12CD1234'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x06FFAB12CD1234")
        struct.length.set_value("09")
        struct.type.set_value("AA")
        self.assertEqual(struct.to_string(prefix=True), "0x09AAAB12CD1234")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x06FFAB12CD1234")


    def test_print(self):
        struct = ManufacturerSpecific()

        string = '0x06FFAB12CD1234'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)

        self.assertEqual(2 * "\t" + "Adv Struct: Manufacturer Specific\n" +
                         3 * "\t" + "Length: 0x06\n" +
                         3 * "\t" + "Type: 0xFF\n" +
                         3 * "\t" + "Company ID: 0x12AB\n" +
                         3 * "\t" + "Data: 0xCD1234\n",
                         capturedOutput.getvalue())

    def test_occurrences(self):
        struct = ManufacturerSpecific()
        target = {'EIR': 'O', 'AD': 'O', 'SRD': 'O', 'ACAD': 'O', 'OOB': 'O'}

        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0xFF), target)


