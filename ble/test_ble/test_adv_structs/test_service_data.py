import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import ADVERTISING_REGISTRY, ServiceData16Bits, ServiceData32Bits, ServiceData128Bits


class TestServiceDataAdvStruct(unittest.TestCase):
    def test_structure_16_bit(self) -> None:
        struct = ServiceData16Bits()
        self.assertIsInstance(struct.uuid, HexDataField)
        self.assertEqual(struct.uuid.get_name(), "16 bit UUID")
        self.assertEqual(struct.uuid.get_value(), "")

        self.assertIsInstance(struct.data, HexDataField)
        self.assertEqual(struct.data.get_name(), "Data")
        self.assertEqual(struct.data.get_value(), "")

        self.assertEqual(struct.type.get_value(), "16")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Service Data 16 bit UUID")
        self.assertEqual(struct._get_registry_key(), 0x16)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'service_data_16_bit_uuid','16', '0x16']})


    def test_structure_32_bit(self) -> None:
        struct = ServiceData32Bits()
        self.assertIsInstance(struct.uuid, HexDataField)
        self.assertEqual(struct.uuid.get_name(), "32 bit UUID")
        self.assertEqual(struct.uuid.get_value(), "")

        self.assertIsInstance(struct.data, HexDataField)
        self.assertEqual(struct.data.get_name(), "Data")
        self.assertEqual(struct.data.get_value(), "")

        self.assertEqual(struct.type.get_value(), "20")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Service Data 32 bit UUID")
        self.assertEqual(struct._get_registry_key(), 0x20)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'service_data_32_bit_uuid','20', '0x20']})

    def test_structure_128_bit(self) -> None:
        struct = ServiceData128Bits()
        self.assertIsInstance(struct.uuid, HexDataField)
        self.assertEqual(struct.uuid.get_name(), "128 bit UUID")
        self.assertEqual(struct.uuid.get_value(), "")

        self.assertIsInstance(struct.data, HexDataField)
        self.assertEqual(struct.data.get_name(), "Data")
        self.assertEqual(struct.data.get_value(), "")

        self.assertEqual(struct.type.get_value(), "21")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Service Data 128 bit UUID")
        self.assertEqual(struct._get_registry_key(), 0x21)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'service_data_128_bit_uuid','21', '0x21']})

    def test_from_and_to_string_128bit(self):
        struct = ServiceData128Bits()

        string = '0x1421CB34AB12CB34AB12CB34AB12CB34AB12CD1234'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x1421CB34AB12CB34AB12CB34AB12CB34AB12CD1234")
        self.assertEqual(struct.to_string(prefix=False), "1421CB34AB12CB34AB12CB34AB12CB34AB12CD1234")
        self.assertEqual(struct.type.get_value(), "21")
        self.assertEqual(struct.length.get_value(), "14")
        self.assertEqual(struct.uuid.get_value(), "12AB34CB12AB34CB12AB34CB12AB34CB")
        self.assertEqual(struct.data.get_value(), "CD1234")


        self.assertEqual(struct.get_length(), 21)
        self.assertEqual(struct.get_length(bit=True), 21*8)


    def test_from_and_to_string_32bit(self):
        struct = ServiceData32Bits()

        string = '0x0820CB34AB12CD1234'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x0820CB34AB12CD1234")
        self.assertEqual(struct.to_string(prefix=False), "0820CB34AB12CD1234")
        self.assertEqual(struct.type.get_value(), "20")
        self.assertEqual(struct.length.get_value(), "08")
        self.assertEqual(struct.uuid.get_value(), "12AB34CB")
        self.assertEqual(struct.data.get_value(), "CD1234")


        self.assertEqual(struct.get_length(), 9)
        self.assertEqual(struct.get_length(bit=True), 9*8)

    def test_from_and_to_string_16_bit(self):
        struct = ServiceData16Bits()

        string = '0x0616CB34CD1234'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x0616CB34CD1234")
        self.assertEqual(struct.to_string(prefix=False), "0616CB34CD1234")
        self.assertEqual(struct.type.get_value(), "16")
        self.assertEqual(struct.length.get_value(), "06")
        self.assertEqual(struct.uuid.get_value(), "34CB")
        self.assertEqual(struct.data.get_value(), "CD1234")


        self.assertEqual(struct.get_length(), 7)
        self.assertEqual(struct.get_length(bit=True), 7*8)

    def test_pars_mode_tolerant(self):
        struct = ServiceData16Bits()

        string = '0x0616CB34CD1234'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0616CB34CD1234")

        string = '0x0911CB34CD1234C'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0911CB34CD1234")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0616CB34CD1234")

        with self.assertRaises(ValueError):
            struct = ServiceData16Bits()
            string = '0x0616CB34CD123X'
            struct.from_string(string, parse_mode="tolerant")

    def test_parse_mode_strict_and_normal(self):
        struct = ServiceData16Bits()

        with self.assertRaises(ParseError):
            string = '0x0516CB34CD1234'
            struct.from_string(string, parse_mode="normal")

        with self.assertRaises(ParseError):
            string = '0x0613CB34CD1234'
            struct.from_string(string, parse_mode="normal")

        with self.assertRaises(ParseError):
            string = '0x0616CB34CD1234C'
            struct.from_string(string, parse_mode="normal")

        with self.assertRaises(ValueError):
            string = '0x0616CB34CD1234X'
            struct.from_string(string, parse_mode="normal")


        with self.assertRaises(ParseError):
            string = '0x0516CB34CD1234'
            struct.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = '0x0613CB34CD1234'
            struct.from_string(string, parse_mode="strict")

        with self.assertRaises(ParseError):
            string = '0x0616CB34CD1234C'
            struct.from_string(string, parse_mode="strict")

        with self.assertRaises(ValueError):
            string = '0x0616CB34CD1234X'
            struct.from_string(string, parse_mode="strict")

    def test_update(self):
        struct = ServiceData32Bits()

        string = '0x0820CB34AB12CD1234'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x0820CB34AB12CD1234")
        struct.length.set_value("09")
        struct.type.set_value("AA")
        self.assertEqual(struct.to_string(prefix=True), "0x09AACB34AB12CD1234")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0820CB34AB12CD1234")


    def test_print(self):
        struct = ServiceData16Bits()

        string = '0x0616CB34CD1234'

        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)

        self.assertEqual(2 * "\t" + "Adv Struct: Service Data 16 bit UUID\n" +
                         3 * "\t" + "Length: 0x06\n" +
                         3 * "\t" + "Type: 0x16\n" +
                         3 * "\t" + "16 bit UUID: 0x34CB\n" +
                         3 * "\t" + "Data: 0xCD1234\n",
                         capturedOutput.getvalue())

    def test_occurrences(self):
        target = {'EIR': 'X', 'AD': 'O', 'SRD': 'O', 'ACAD': 'O', 'OOB': 'O'}

        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x16), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x20), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x21), target)


