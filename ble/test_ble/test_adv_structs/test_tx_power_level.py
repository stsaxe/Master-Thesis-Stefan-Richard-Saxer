import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.advertising_data.AdvertisingData import TxPowerLevel, ADVERTISING_REGISTRY


class TestTxPowerLevelAdvStruct(unittest.TestCase):
    def test_structure(self) -> None:
        struct = TxPowerLevel()
        self.assertIsInstance(struct.power_level, HexDataField)
        self.assertEqual(struct.power_level.get_name(), "Power Level")
        self.assertEqual(struct.power_level.get_value(), "")

        self.assertEqual(struct.type.get_value(), "0A")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Tx Power Level")
        self.assertEqual(struct._get_registry_key(), 0x0a)

        self.assertEqual(struct.get_name(), "Adv Struct")

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'tx_power_level','0a', '0x0a']})



    def test_from_and_to_string(self):
        struct = TxPowerLevel()

        string = '0x020A1B'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x020A1B")
        self.assertEqual(struct.to_string(prefix=False), "020A1B")
        self.assertEqual(struct.type.get_value(), "0A")
        self.assertEqual(struct.length.get_value(), "02")
        self.assertEqual(struct.power_level.get_value(), "1B")

        self.assertEqual(struct.get_length(), 3)
        self.assertEqual(struct.get_length(bit=True), 24)


    def test_pars_mode_tolerant(self):
        struct = TxPowerLevel()

        # parse valid string
        string = '0x020A1B'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x020A1B")

        # update incorrect length and type and exceeding length bits
        string = '0x010C1B'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x010C1B")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x020A1B")

        # invalid hex characters
        with self.assertRaises(ValueError):
            struct = TxPowerLevel()
            string = '0x020A1X'
            struct.from_string(string, parse_mode="tolerant")



    def test_parse_mode_strict_and_normal(self):
        struct = TxPowerLevel()

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x0120A1B'
            struct.from_string(string, parse_mode="normal")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x020F1B'
            struct.from_string(string, parse_mode="normal")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x020A1BC'
            struct.from_string(string, parse_mode="normal")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x020A1X'
            struct.from_string(string, parse_mode="normal")

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x090A1B'
            struct.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x022A1B'
            struct.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x020A1BC'
            struct.from_string(string, parse_mode="strict")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x020A1X'
            struct.from_string(string, parse_mode="strict")


    def test_update(self):
        struct = TxPowerLevel()

        string = '0x020A1B'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x020A1B")
        struct.length.set_value("09")
        struct.type.set_value("AA")
        self.assertEqual(struct.to_string(prefix=True), "0x09AA1B")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x020A1B")


    def test_print(self):
        struct = TxPowerLevel()

        string = '0x020A1B'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)

        self.assertEqual(2 * "\t" + "Adv Struct: Tx Power Level\n" +
                         3 * "\t" + "Length: 0x02\n" +
                         3 * "\t" + "Type: 0x0A\n" +
                         3 * "\t" + "Power Level: 0x1B\n",
                         capturedOutput.getvalue())

    def test_occurances(self):
        target = {'EIR': 'O', 'AD': 'O', 'SRD': 'O', 'ACAD': 'X', 'OOB': 'O'}

        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x0a), target)


