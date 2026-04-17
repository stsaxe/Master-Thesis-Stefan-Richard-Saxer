import io
import sys
from unittest import TestCase

from ble.components.pdu.AdvertisingPDUs import NullPDU
from ble.components.pdu.AdvHeader import AdvHeader


class TestNullPdu(TestCase):
    def test_structure(self):
        pdu = NullPDU()
        self.assertEqual(pdu.get_length(), 0)
        self.assertEqual(pdu.header.pdu_type.get_value(bin=True), "")
        self.assertEqual(pdu.header.length.get_value(bin=True), "")

        self.assertIsInstance(pdu.header, AdvHeader)

    def test_from_and_to_string(self):
        pdu = NullPDU()
        pdu.from_string("ABCD")
        self.assertEqual(pdu.to_string(), "")
        self.assertEqual(pdu.to_string(prefix=True), "0x")

    def test_update(self):
        pdu = NullPDU()
        self.assertEqual(pdu.header.get_length(), 0)

        pdu.header.pdu_type.set_value("1010", bin=True)
        pdu.header.length.set_value("AB")

        self.assertEqual(pdu.header.get_length(bit=True), 12)
        self.assertEqual(pdu.get_length(bit=True), 12)

        pdu.update()

        self.assertEqual(pdu.header.get_length(), 0)

    def test_print(self):
        pdu = NullPDU()

        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput

        pdu._print(ident=2)

        self.assertEqual("", capturedOutput.getvalue())
