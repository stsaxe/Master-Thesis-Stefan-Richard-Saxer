import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import ADVERTISING_REGISTRY, ServiceUUID16ListIncomplete, ServiceUUID16ListComplete, ServiceUUID32ListIncomplete, ServiceUUID32ListComplete, ServiceUUID128ListIncomplete, ServiceUUID128ListComplete
from ble.components.MultiNodeContainer import MultiNodeContainer


class TestServiceUUIDListAdvStruct(unittest.TestCase):
    def test_structure_16_bit_incomplete(self) -> None:
        struct = ServiceUUID16ListIncomplete()
        self.assertIsInstance(struct.uuids, MultiNodeContainer)

        self.assertEqual(struct.uuids.get_name(), "16 bit UUIDs")

        self.assertEqual(struct.uuids.components, [])


        self.assertEqual(struct.type.get_value(), "02")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Incomplete List of 16 bit Service or Service Class UUIDs")
        self.assertEqual(struct._get_registry_key(), 0x02)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'incomplete_list_of_16_bit_service_or_service_class_uuids','02', '0x02']})

    def test_structure_16_bit_complete(self) -> None:
        struct = ServiceUUID16ListComplete()
        self.assertIsInstance(struct.uuids, MultiNodeContainer)

        self.assertEqual(struct.uuids.components, [])


        self.assertEqual(struct.type.get_value(), "03")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Complete List of 16 bit Service or Service Class UUIDs")
        self.assertEqual(struct._get_registry_key(), 0x03)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'complete_list_of_16_bit_service_or_service_class_uuids','03', '0x03']})


    def test_structure_32_bit_incomplete(self) -> None:
        struct = ServiceUUID32ListIncomplete()
        self.assertIsInstance(struct.uuids, MultiNodeContainer)

        self.assertEqual(struct.uuids.components, [])


        self.assertEqual(struct.type.get_value(), "04")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Incomplete List of 32 bit Service or Service Class UUIDs")
        self.assertEqual(struct._get_registry_key(), 0x04)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'incomplete_list_of_32_bit_service_or_service_class_uuids','04', '0x04']})

    def test_structure_32_bit_complete(self) -> None:
        struct = ServiceUUID32ListComplete()
        self.assertIsInstance(struct.uuids, MultiNodeContainer)

        self.assertEqual(struct.uuids.components, [])


        self.assertEqual(struct.type.get_value(), "05")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Complete List of 32 bit Service or Service Class UUIDs")
        self.assertEqual(struct._get_registry_key(), 0x05)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'complete_list_of_32_bit_service_or_service_class_uuids','05', '0x05']})


    def test_structure_128_bit_incomplete(self) -> None:
        struct = ServiceUUID128ListIncomplete()
        self.assertIsInstance(struct.uuids, MultiNodeContainer)

        self.assertEqual(struct.uuids.components, [])


        self.assertEqual(struct.type.get_value(), "06")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Incomplete List of 128 bit Service or Service Class UUIDs")
        self.assertEqual(struct._get_registry_key(), 0x06)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'incomplete_list_of_128_bit_service_or_service_class_uuids','06', '0x06']})

    def test_structure_128_bit_complete(self) -> None:
        struct = ServiceUUID128ListComplete()
        self.assertIsInstance(struct.uuids, MultiNodeContainer)

        self.assertEqual(struct.uuids.components, [])


        self.assertEqual(struct.type.get_value(), "07")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Complete List of 128 bit Service or Service Class UUIDs")
        self.assertEqual(struct._get_registry_key(), 0x07)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'complete_list_of_128_bit_service_or_service_class_uuids','07', '0x07']})





    def test_from_and_to_string(self):
        struct = ServiceUUID16ListComplete()

        string = '0x0703AB12CD34AB12'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")
        self.assertEqual(struct.to_string(prefix=False), "0703AB12CD34AB12")
        self.assertEqual(struct.type.get_value(), "03")
        self.assertEqual(struct.length.get_value(), "07")

        targets = ["12AB", "34CD", "12AB"]

        for uuid, target in zip(struct.uuids, targets):
            self.assertEqual(uuid.get_value(), target)
            self.assertIsInstance(uuid, HexDataField)
            self.assertEqual(uuid.get_name(), "16 bit UUID")

        self.assertEqual(struct.get_length(), 8)
        self.assertEqual(struct.get_length(bit=True), 8*8)

        # run this twice to verify that the uuid container is cleared before every parse from string
        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")
        self.assertEqual(struct.to_string(prefix=False), "0703AB12CD34AB12")
        self.assertEqual(struct.type.get_value(), "03")
        self.assertEqual(struct.length.get_value(), "07")

        targets = ["12AB", "34CD", "12AB"]

        for uuid, target in zip(struct.uuids, targets):
            self.assertEqual(uuid.get_value(), target)
            self.assertIsInstance(uuid, HexDataField)
            self.assertEqual(uuid.get_name(), "16 bit UUID")

    def test_pars_mode_tolerant(self):
        struct = ServiceUUID16ListComplete()

        # parse valid string
        string = '0x0703AB12CD34AB12'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")

        # update incorrect length and type and exceeding length bits
        string = '0x0201AB12CD34AB12D'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0201AB12CD34AB12")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")

        # exceeding byte
        string = '0x0703AB12CD34AB12CC'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")


        # invalid hex characters
        with self.assertRaises(ValueError):
            struct = ServiceUUID16ListComplete()
            string = '0x0703AB12CD34AB1X'
            struct.from_string(string, parse_mode="tolerant")

    def test_parse_mode_strict_and_normal(self):
        struct = ServiceUUID16ListComplete()

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x0203AB12CD34AB12'
            struct.from_string(string, parse_mode="normal")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x0701AB12CD34AB12'
            struct.from_string(string, parse_mode="normal")

        # exceeding byte
        with self.assertRaises(ParseError):
            string = '0x0803AB12CD34AB12FF'
            struct.from_string(string, parse_mode="normal")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x0703AB12CD34AB1X'
            struct.from_string(string, parse_mode="normal")


        # wrong length
        with self.assertRaises(ParseError):
            string = '0xF703AB12CD34AB12'
            struct.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x07FFAB12CD34AB12'
            struct.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x0703AB12CD34AB12C'
            struct.from_string(string, parse_mode="strict")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x0703AB12CD34AXXX'
            struct.from_string(string, parse_mode="strict")

    def test_update(self):
        struct = ServiceUUID16ListComplete()

        string = '0x0703AB12CD34AB12'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")
        struct.length.set_value("09")
        struct.type.set_value("AA")
        self.assertEqual(struct.to_string(prefix=True), "0x09AAAB12CD34AB12")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x0703AB12CD34AB12")


    def test_print(self):
        struct = ServiceUUID16ListComplete()

        string = '0x0703AB12CD34AB12'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)

        self.assertEqual(2 * "\t" + "Adv Struct: Complete List of 16 bit Service or Service Class UUIDs\n" +
                         3 * "\t" + "Length: 0x07\n" +
                         3 * "\t" + "Type: 0x03\n" +
                         3 * "\t" + "16 bit UUIDs\n" +
                         4 * "\t" + "16 bit UUID: 0x12AB\n"+
                         4 * "\t" + "16 bit UUID: 0x34CD\n"+
                         4 * "\t" + "16 bit UUID: 0x12AB\n",
                         capturedOutput.getvalue())

    def test_occurrences(self):
        target = {'EIR': 'O', 'AD': 'O', 'SRD': 'O', 'ACAD': 'O', 'OOB': 'O'}
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x02), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x03), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x04), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x5), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x06), target)
        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x07), target)


