import unittest

from ble.utils.HelperMethods import HelperMethods


class TestHelperMethods(unittest.TestCase):
    def test_valid_string(self):
        value = HelperMethods.check_valid_string("12AB")
        self.assertEqual(value, "12AB")

        value = HelperMethods.check_valid_string("", empty_allowed=True)
        self.assertEqual(value, "")

        with self.assertRaises(AssertionError):
            value = HelperMethods.check_valid_string("")

        with self.assertRaises(AssertionError):
            value = HelperMethods.check_valid_string(123)


    def test_bin_to_hex(self):
        value = HelperMethods.bin_to_hex("1111")
        self.assertEqual(value, "F")
        value = HelperMethods.bin_to_hex("11110000")
        self.assertEqual(value, "F0")

        value = HelperMethods.bin_to_hex("0000001111", pad="nibble")
        self.assertEqual(value, "00F")

        value = HelperMethods.bin_to_hex("1111", pad="byte")
        self.assertEqual(value, "0F")

        with self.assertRaises(AssertionError):
            value = HelperMethods.bin_to_hex("0F", pad="xxx")

    def test_int_to_hex(self):
        value = HelperMethods.int_to_hex(161)
        self.assertEqual(value, "A1")

        value = HelperMethods.int_to_hex(5, pad="byte")
        self.assertEqual(value, "05")

        value = HelperMethods.int_to_hex(5, pad="nibble")
        self.assertEqual(value, "5")

        with self.assertRaises(AssertionError):
            value = HelperMethods.int_to_hex(-4)

        with self.assertRaises(AssertionError):
            value = HelperMethods.int_to_hex("3")

    def test_int_to_bin(self):
        value = HelperMethods.int_to_bin(5)
        self.assertEqual(value, "101")

        value = HelperMethods.int_to_bin(5, pad="nibble")
        self.assertEqual(value, "0101")

        value = HelperMethods.int_to_bin(5, pad="byte")
        self.assertEqual(value, "00000101")

        with self.assertRaises(AssertionError):
            value = HelperMethods.int_to_bin(-4)

        with self.assertRaises(AssertionError):
            value = HelperMethods.int_to_bin("test")


        with self.assertRaises(AssertionError):
            value = HelperMethods.int_to_bin(4, pad="xxx")

    def test_hex_to_bin(self):
        value = HelperMethods.hex_to_bin("0")
        self.assertEqual(value,  "0")

        value = HelperMethods.hex_to_bin("5")
        self.assertEqual(value,  "101")

        value = HelperMethods.hex_to_bin("5", pad="nibble")
        self.assertEqual(value,  "0101")

        value = HelperMethods.hex_to_bin("5", pad="byte")
        self.assertEqual(value,  "00000101")

        value = HelperMethods.hex_to_bin("05")
        self.assertEqual(value, "101")

        value = HelperMethods.hex_to_bin("05", pad="nibble")
        self.assertEqual(value,  "0101")

        value = HelperMethods.hex_to_bin("05", pad="byte")
        self.assertEqual(value,  "00000101")

        with self.assertRaises(AssertionError):
            value = HelperMethods.hex_to_bin("05", pad="xxx")

    def test_hex_le_to_be(self):
        value = HelperMethods.hex_le_to_be("AB1234")
        self.assertEqual(value, "3412AB")

        value = HelperMethods.hex_le_to_be("")
        self.assertEqual(value, "")

        with self.assertRaises(AssertionError):
            value = HelperMethods.hex_le_to_be("AB123")


    def test_hex_be_to_le(self):
        value = HelperMethods.hex_be_to_le("AB1234")
        self.assertEqual(value, "3412AB")

        value = HelperMethods.hex_le_to_be("")
        self.assertEqual(value, "")

        with self.assertRaises(AssertionError):
            value = HelperMethods.hex_le_to_be("AB123")

        value = HelperMethods.hex_be_to_le(HelperMethods.hex_le_to_be("AB1234"))
        self.assertEqual(value, "AB1234")


    def test_clean_hex_Value(self):
        value = HelperMethods.clean_hex_value("   0XA:b 12__- 34  ")
        self.assertEqual(value, "AB1234")

        value = HelperMethods.clean_hex_value("00AB")
        self.assertEqual(value, "00AB")

        value = HelperMethods.clean_hex_value("0xAb 12__- 34", prefix=True)
        self.assertEqual(value, "0xAB1234")

        value = HelperMethods.clean_hex_value("", empty_allowed=True)
        self.assertEqual(value, "")

        with self.assertRaises(ValueError):
            value = HelperMethods.clean_hex_value("ABZXY123")

        with self.assertRaises(AssertionError):
            value = HelperMethods.clean_hex_value("", empty_allowed=False)

    def test_clean_bin_Value(self):
        value = HelperMethods.clean_bin_value("   0B0:1 11__- 10  ")
        self.assertEqual(value, "011110")

        value = HelperMethods.clean_bin_value("   0B0:1 11__- 10  ", prefix=True)
        self.assertEqual(value, "0b011110")

        value = HelperMethods.clean_bin_value("", empty_allowed=True)
        self.assertEqual(value, "")

        with self.assertRaises(ValueError):
            value = HelperMethods.clean_bin_value("010AB1")

        with self.assertRaises(AssertionError):
            value = HelperMethods.clean_bin_value("", empty_allowed=False)





