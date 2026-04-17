import io
import sys
import unittest

from ble import BitDataField


class TestBitDataField(unittest.TestCase):
    def test_getters_and_setters(self):
        field = BitDataField("Test", "0xABC", bin=False)

        self.assertEqual(field.get_bit_length(), 3*4)
        self.assertEqual(field.get_name(), "Test")
        self.assertEqual(field.get_value(), "ABC")
        self.assertEqual(field.get_value(prefix=True), "0xABC")
        self.assertEqual(field.get_value(bin=True, prefix=True), "0b101010111100")
        self.assertEqual(field.get_value(bin=True), "101010111100")

        field.set_value("101001010", bin=True)
        field.set_name("ABCD")

        self.assertEqual(field.get_name(), "ABCD")
        self.assertEqual(field.get_value(prefix=True), "0x14A")

        field.set_name("ABCD")

        # test leading zeroes
        field.set_value("000", bin=True)
        self.assertEqual(field.get_value(bin=True), "000")

        field.set_value("000", bin=True)
        self.assertEqual(field.get_value(bin=False), "0")

        field.set_value("12AB", bin=False)
        self.assertEqual(field.get_name(), "ABCD")
        self.assertEqual(field.get_value(prefix=True), "0x12AB")

        field.set_value("7", bin=False)
        self.assertEqual(field.get_value(prefix=True, bin=True), "0b111")

    def test_target_bit_length(self):
        field = BitDataField("Test", bin=False, target_bit_length=3)

        field.set_value("110", bin=True)
        self.assertEqual(field.get_target_bit_length(), 3)
        self.assertEqual(field.get_value(bin=True), "110")

        with self.assertRaises(AssertionError):
            field = BitDataField("Test", bin=False, target_bit_length=3)

            field.set_value("0110", bin=True)

        with self.assertRaises(AssertionError):
            field = BitDataField("Test", bin=False, target_bit_length=3)

            field.set_value("1110", bin=True)

        field = BitDataField("Test", bin=False)
        self.assertIsNone(field.get_target_bit_length())

    def test_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            field = BitDataField("", "0xABC", bin=False)

        with self.assertRaises(ValueError):
            field = BitDataField("Test", "0xabcZ", bin=False)

        with self.assertRaises(ValueError):
            field = BitDataField("Test", "12345", bin=True)

        with self.assertRaises(AssertionError):
            field = BitDataField("Test", 1234, bin=True)



    def test_print(self):
        field = BitDataField("Test",value="0x7")

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        field._print(ident = 1)

        self.assertEqual('\t' + 'Test: ' + '0b111\n', capturedOutput.getvalue())



    def test_empty_field(self):
        field = BitDataField("Test", bin=False)
        self.assertEqual(field.get_bit_length(), 0)
        self.assertEqual(field.get_name(), "Test")
        self.assertEqual(field.get_value(prefix=True), "0x")
        self.assertEqual(field.get_value(prefix=False), "")

        field = BitDataField("Test", bin=True)
        self.assertEqual(field.get_bit_length(), 0)
        self.assertEqual(field.get_value(prefix=True, bin=True), "0b")
        self.assertEqual(field.get_value(prefix=False), "")

        field = BitDataField("Test","1000", bin=True)
        self.assertEqual(field.get_value(prefix=True, bin=True), "0b1000")
        field.set_value("")
        self.assertEqual(field.get_value(prefix=False), "")

    def test_set_bit_value_and_get_hex(self):
        field = BitDataField("Test", "000", bin=True)
        self.assertEqual(field.get_value(bin=False), "0")

        field = BitDataField("Test", "010", bin=True)
        self.assertEqual(field.get_value(bin=False), "2")




