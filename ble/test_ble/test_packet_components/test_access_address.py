import unittest

from ble.errors.ParseError import ParseError
from ble.fields.HexDataField import HexDataField
from ble.components.packet.AccessAddress import AccessAddress


class TestAccessAddress(unittest.TestCase):
    def test_structure(self):
        addr = AccessAddress()
        self.assertIsInstance(addr, HexDataField)
        self.assertEqual(addr.to_string(prefix=True), "0x")
        self.assertEqual(addr.to_string(prefix=False), "")
        self.assertEqual(addr.get_name(), "Access Address")

    def test_from_string(self):
        addr = AccessAddress()

        addr.from_string("AB123456")
        self.assertEqual(addr.get_value(), "563412AB")
        self.assertEqual(addr.to_string(), "AB123456")

    def test_policy_strict(self):
        addr = AccessAddress()
        addr.from_string("AB123456", parse_mode="strict")
        self.assertEqual(addr.get_value(), "563412AB")

        with self.assertRaises(ParseError):
            addr.from_string("AB12345678", parse_mode="strict")

    def test_policy_tolerant(self):
        addr = AccessAddress()
        addr.from_string("AB12345678", parse_mode="tolerant")
        self.assertEqual(addr.get_value(), "563412AB")






