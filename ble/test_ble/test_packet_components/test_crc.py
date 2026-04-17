import io
import sys
import unittest

from ble.components.packet.CRC import CRC
from ble.fields.HexDataField import HexDataField


class TestCrc(unittest.TestCase):
    def test_structure(self):
        crc = CRC()

        self.assertIsInstance(crc, HexDataField)
        self.assertEqual(crc.to_string(prefix=True), "0x")
        self.assertEqual(crc.to_string(prefix=False), "")
        self.assertEqual(crc.get_name(), "CRC")

    def test_print(self):
        crc = CRC()
        crc.set_value("123456")

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        crc._print(ident=1)
        self.assertEqual('\t' + 'CRC: ' + '0x482C6A\n', capturedOutput.getvalue())

    def test_print_empty(self):
        crc = CRC()

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        crc._print(ident=1)
        self.assertEqual('\t' + 'CRC: ' + '0x\n', capturedOutput.getvalue())

    def test_from_pdu(self):
        crc = CRC()
        crc_value = crc.from_pdu("")
        self.assertEqual('AAAAAA', crc_value)

        crc_value = crc.from_pdu("4024320cfb574d5a02011a1aff4c000c0e009c6b8f40440f1583ec895148b410050318c0b525")


        # the reference value here was computed with scapy
        self.assertEqual('B8F7D4', crc_value)

        with self.assertRaises(ValueError):
            crc.from_pdu("1")

