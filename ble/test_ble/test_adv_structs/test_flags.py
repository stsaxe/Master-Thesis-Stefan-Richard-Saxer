import io
import sys
import unittest

from ble.errors.ParseError import ParseError
from ble.fields.BitDataField import BitDataField
from ble.components.advertising_data.AdvertisingData import Flags, ADVERTISING_REGISTRY


class TestFlagsAdvStruct(unittest.TestCase):
    def test_structure(self):
        struct = Flags()

        self.assertIsInstance(struct.bit_0, BitDataField)
        self.assertIsInstance(struct.bit_1, BitDataField)
        self.assertIsInstance(struct.bit_2, BitDataField)
        self.assertIsInstance(struct.bit_3, BitDataField)
        self.assertIsInstance(struct.bit_4, BitDataField)
        self.assertIsInstance(struct.bit_5_to_7, BitDataField)
        self.assertEqual(struct.get_name(), "Adv Struct")

        self.assertEqual(struct.bit_0.get_value(), "")

        self.assertEqual(struct.bit_0.get_value(bin=True), "")
        self.assertEqual(struct.bit_1.get_value(bin=True), "")
        self.assertEqual(struct.bit_2.get_value(bin=True), "")
        self.assertEqual(struct.bit_3.get_value(bin=True), "")
        self.assertEqual(struct.bit_4.get_value(bin=True), "")
        self.assertEqual(struct.bit_5_to_7.get_value(bin=True), "")


        self.assertEqual(struct.bit_4.get_name(), "Simultaneous LE and BR/EDR to Same Device Capable (Host)")
        self.assertEqual(struct.bit_3.get_name(), "Simultaneous LE and BR/EDR to Same Device Capable (Controller)")
        self.assertEqual(struct.bit_2.get_name(), "BR/EDR Not Supported")
        self.assertEqual(struct.bit_1.get_name(), "LE General Discoverable Mode")
        self.assertEqual(struct.bit_0.get_name(), "LE Limited Discoverable Mode")
        self.assertEqual(struct.bit_5_to_7.get_name(), "RFU")


        self.assertEqual(struct.type.get_value(), "01")
        self.assertEqual(struct.length.get_value(),  "00")
        self.assertEqual(struct.get_length(), 2)

        self.assertEqual(struct._get_registry_name(), "Flags")
        self.assertEqual(struct._get_registry_key(), 0x01)

        segment = struct.get_path_segment()

        self.assertEqual(segment.name, "adv_struct")
        self.assertEqual(segment.keys, {"adv_type": [ 'flags','01', '0x01']})



    def test_from_and_to_string(self):
        struct = Flags()

        string = '0x02011a'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x02011A")
        self.assertEqual(struct.to_string(prefix=False), "02011A")
        self.assertEqual(struct.type.get_value(), "01")
        self.assertEqual(struct.length.get_value(), "02")

        struct.print()

        self.assertEqual(struct.bit_0.get_value(bin=True), "0")
        self.assertEqual(struct.bit_1.get_value(bin=True), "1")
        self.assertEqual(struct.bit_2.get_value(bin=True), "0")
        self.assertEqual(struct.bit_3.get_value(bin=True), "1")
        self.assertEqual(struct.bit_4.get_value(bin=True), "1")
        self.assertEqual(struct.bit_5_to_7.get_value(bin=True), "000")

    def test_from_and_to_string_flags_all_zero(self):
        struct = Flags()

        string = '0x020100'

        struct.from_string(string)
        self.assertEqual(struct.to_string(prefix=True), "0x020100")
        self.assertEqual(struct.to_string(prefix=False), "020100")
        self.assertEqual(struct.type.get_value(), "01")
        self.assertEqual(struct.length.get_value(), "02")

        struct.print()

        self.assertEqual(struct.bit_0.get_value(bin=True), "0")
        self.assertEqual(struct.bit_1.get_value(bin=True), "0")
        self.assertEqual(struct.bit_2.get_value(bin=True), "0")
        self.assertEqual(struct.bit_3.get_value(bin=True), "0")
        self.assertEqual(struct.bit_4.get_value(bin=True), "0")
        self.assertEqual(struct.bit_5_to_7.get_value(bin=True), "000")


        self.assertEqual(struct.get_length(), 3)
        self.assertEqual(struct.get_length(bit=True), 24)

    def test_pars_mode_tolerant(self):
        struct = Flags()

        # parse valid string
        string = '0x0201AA'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0201AA")

        # update incorrect length and type and exceeding length bits and invalid RFU values
        string = '0x0702AACB'
        struct.from_string(string, parse_mode="tolerant")
        self.assertEqual(struct.to_string(prefix=True), "0x0702AA")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x02010A")

        # invalid hex characters
        with self.assertRaises(ValueError):
            struct = Flags()
            string = '0x02011X'
            struct.from_string(string, parse_mode="tolerant")

    def test_parse_mode_strict_and_normal(self):
        struct = Flags()

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x01011A'
            struct.from_string(string, parse_mode="normal")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x02311A'
            struct.from_string(string, parse_mode="normal")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x02011AC'
            struct.from_string(string, parse_mode="normal")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x02011X'
            struct.from_string(string, parse_mode="normal")

        # invalid RFU values
        string = '0x0201AA'
        struct.from_string(string, parse_mode="normal")
        self.assertEqual(struct.to_string(prefix=True), "0x0201AA")

        # wrong length
        with self.assertRaises(ParseError):
            string = '0x06011A'
            struct.from_string(string, parse_mode="strict")

        # wrong type
        with self.assertRaises(ParseError):
            string = '0x02411A'
            struct.from_string(string, parse_mode="strict")

        # exceeding bits
        with self.assertRaises(ParseError):
            string = '0x02011AC'
            struct.from_string(string, parse_mode="strict")

        # invalid hex characters
        with self.assertRaises(ValueError):
            string = '0x02011X'
            struct.from_string(string, parse_mode="strict")

        # invalid RFU values
        with self.assertRaises(ParseError):
            string = '0x0201AA'
            struct.from_string(string, parse_mode="strict")

    def test_update(self):
        struct = Flags()

        string = '0x02011A'
        struct.from_string(string)

        self.assertEqual(struct.to_string(prefix=True), "0x02011A")


        struct.length.set_value("09")
        struct.type.set_value("AA")
        struct.bit_5_to_7.set_value("110", bin=True)
        self.assertEqual(struct.to_string(prefix=True), "0x09AADA")
        struct.update()
        self.assertEqual(struct.to_string(prefix=True), "0x02011A")


    def test_print(self):
        struct = Flags()

        string = '0x02011A'
        struct.from_string(string)

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        struct._print(ident = 2)



        self.assertEqual(2 * "\t" + "Adv Struct: Flags\n" +
                         3 * "\t" + "Length: 0x02\n" +
                         3 * "\t" + "Type: 0x01\n" +
                         3 * "\t" + "RFU: 0b000\n" +
                         3 * "\t" + "Simultaneous LE and BR/EDR to Same Device Capable (Host): 0b1\n" +
                         3 * "\t" + "Simultaneous LE and BR/EDR to Same Device Capable (Controller): 0b1\n" +
                         3 * "\t" + "BR/EDR Not Supported: 0b0\n" +
                         3 * "\t" + "LE General Discoverable Mode: 0b1\n" +
                         3 * "\t" + "LE Limited Discoverable Mode: 0b0\n",
                         capturedOutput.getvalue())

    def test_occurances(self):
        target = {'EIR': 'C1', 'AD': 'C1', 'SRD': 'X', 'ACAD': 'X', 'OOB': 'C1'}

        self.assertEqual(ADVERTISING_REGISTRY.get_occurrences(0x01), target)


