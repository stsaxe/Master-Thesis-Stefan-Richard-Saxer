import io
import sys
import unittest

from ble import HexDataField
from ble.errors.ParseError import ParseError


class TestHexField(unittest.TestCase):
    def test_setters_and_getters(self):
        field = HexDataField("Test", "0xABCD", bin=False)

        self.assertEqual(field.get_bit_length(), 4 * 4)
        self.assertEqual(field.get_length(), 2)
        self.assertEqual(field.get_length(bit=True), 2 * 8)
        self.assertEqual(field.get_name(), "Test")
        self.assertEqual(field.get_value(), "ABCD")
        self.assertEqual(field.get_value(prefix=True), "0xABCD")

        field.set_value("1234")
        field.set_name("ABCD")

        self.assertEqual(field.get_name(), "ABCD")
        self.assertEqual(field.get_value(prefix=True), "0x1234")

    def test_target_length(self):
        field = HexDataField("Test", "0x1234", bin=False, target_bit_length=16)
        self.assertEqual(field.get_bit_length(), 16)
        self.assertEqual(field.get_value(), "1234")

        field = HexDataField("Test", "0x1234", bin=False, target_byte_length=2)
        self.assertEqual(field.get_target_byte_length(), 2)
        self.assertEqual(field.get_value(), "1234")

        field = HexDataField("Test", "0x1234", bin=False, target_byte_length=2, target_bit_length=16)
        self.assertEqual(field.get_bit_length(), 16)
        self.assertEqual(field.get_target_byte_length(), 2)
        self.assertEqual(field.get_value(), "1234")

        with self.assertRaises(AssertionError):
            field = HexDataField("Test", "0x1234", bin=False, target_byte_length=4, target_bit_length=16)

        with self.assertRaises(AssertionError):
            field = HexDataField("Test", "0x12", bin=False, target_byte_length=4)

    def test_binary(self):
        field = HexDataField("Test", "0b1001", bin=True)

        self.assertEqual(field.get_bit_length(), 8)
        self.assertEqual(field.get_value(bin=True), "00001001")
        self.assertEqual(field.get_value(bin=True, prefix=True), "0b00001001")
        self.assertEqual(field.get_value(bin=False), "09")
        self.assertEqual(field.get_length(), 1)

        field.set_value("10011010", bin=True)
        self.assertEqual(field.get_value(prefix=True), "0x9A")

    def test_empty_value(self):
        field = HexDataField("Test", bin=False)

        self.assertEqual(field.get_bit_length(), 0)
        self.assertEqual(field.get_length(), 0)
        self.assertEqual(field.get_value(), "")

        field = HexDataField("Test", bin=True)

        self.assertEqual(field.get_bit_length(), 0)
        self.assertEqual(field.get_length(), 0)
        self.assertEqual(field.get_value(), "")

    def test_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            field = HexDataField("", "0xABCD", bin=False)

        with self.assertRaises(ValueError):
            field = HexDataField("Test", "0xabcZ", bin=False)

        with self.assertRaises(AssertionError):
            field = HexDataField("Test", "12345", bin=False)

        with self.assertRaises(ValueError):
            field = HexDataField("Test", "1234", bin=True)

        with self.assertRaises(AssertionError):
            field = HexDataField("Test", 1234, bin=True)

    def test_to_and_from_string(self):
        field = HexDataField("Test", bin=False)

        field.from_string("ABCD")
        self.assertEqual(field.get_value(prefix=True), "0xABCD")
        field.from_string("0xABCD")
        self.assertEqual(field.get_value(prefix=True), "0xABCD")

        self.assertEqual(field.to_string(), "ABCD")
        self.assertEqual(field.to_string(prefix=True), "0xABCD")

    def test_hard_inputs(self):
        field = HexDataField("Test", bin=False)

        field.from_string("0xa:B:_12C-D")
        self.assertEqual(field.get_value(prefix=False), "AB12CD")

    def test_print(self):
        field = HexDataField("Test", value="CBDA")

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        field._print(ident=1)

        self.assertEqual('\t' + 'Test: ' + '0xCBDA\n', capturedOutput.getvalue())

    def test_empty_field(self):
        field = HexDataField("Test", bin=False)
        self.assertEqual(field.get_bit_length(), 0)
        self.assertEqual(field.get_length(), 0)
        self.assertEqual(field.get_value(), "")
        self.assertEqual(field.get_value(bin=True), "")
        self.assertEqual(field.get_value(prefix=True), "0x")
        self.assertEqual(field.get_value(prefix=True, bin=True), "0b")

        field = HexDataField("Test", "0x12", bin=False)
        self.assertEqual(field.get_value(prefix=True, bin=False), "0x12")
        field.set_value("")
        self.assertEqual(field.get_value(prefix=False), "")

    def test_parse_policy(self):
        field = HexDataField("Test", bin=False)

        field.from_string("0x12AB", parse_mode="normal")
        self.assertEqual(field.get_value(prefix=True), "0x12AB")

        field.from_string("0x", parse_mode="normal")
        self.assertEqual(field.get_value(prefix=True), "0x")

        field.from_string("0x12AB", parse_mode="strict")
        self.assertEqual(field.get_value(prefix=True), "0x12AB")

        field.from_string("0x", parse_mode="strict")
        self.assertEqual(field.get_value(prefix=True), "0x")

        with self.assertRaises(ParseError):
            field.from_string("0x12ABC", parse_mode="normal")

        with self.assertRaises(ValueError):
            field.from_string("0x12AX", parse_mode="normal")

        field.from_string("0x12AB", parse_mode="normal")
        self.assertEqual(field.get_value(prefix=True), "0x12AB")

        field.from_string("0x", parse_mode="normal")
        self.assertEqual(field.get_value(prefix=True), "0x")

        with self.assertRaises(ParseError):
            field.from_string("0x12ABC", parse_mode="strict")

        with self.assertRaises(ValueError):
            field.from_string("0x12AX", parse_mode="strict")

        # test best effort
        field.from_string("0x12ABC", parse_mode="tolerant")
        self.assertEqual(field.get_value(prefix=True), "0x12AB")
