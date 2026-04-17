import datetime
import unittest

from scapy.contrib.automotive.autosar.pdu import PDU
from scapy.utils import atol

from ble import AdvInd, AbstractAdvDataPDU
from ble.components.packet.AccessAddress import AccessAddress
from ble.components.packet.CRC import CRC
from ble.components.pdu.AdvertisingPDUs import NullPDU, RawPDU
from ble.components.packet.Packet import Packet
from ble.errors.ParseError import ParseError


class TestPacket(unittest.TestCase):
    def test_structure(self):
        pkt = Packet()

        self.assertEqual(pkt.get_name(), "Packet")
        self.assertIsInstance(pkt.pdu, NullPDU)
        self.assertIsInstance(pkt.access_address, AccessAddress)
        self.assertIsInstance(pkt.crc, CRC)

        self.assertEqual(pkt.get_length(), 0)
        self.assertEqual(pkt.to_string(prefix=True), "0x")
        self.assertEqual(pkt.to_string(), "")

    def test_from_and_to_string(self):
        pdu_string = "600C" + "ABCD12345678" + '02011A' + '020ACB'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc)
        self.assertEqual(pkt.to_string(), access_address + pdu_string + crc)
        self.assertEqual(pkt.to_string(prefix=True), "0x" + access_address + pdu_string + crc)

        self.assertEqual(pkt.get_length(), 12 + 2 + 4 + 3)
        self.assertEqual(pkt.get_length(bit=True), (12 + 2 + 4 + 3) * 8)

        self.assertIsInstance(pkt.pdu, AdvInd)

    def test_date_rssi_and_channel(self):
        pkt = Packet()
        pkt.set_time(1774652400)

        self.assertEqual(pkt.time, datetime.datetime(2026, 3, 28))

        rssi = "12"
        pkt.set_rssi(rssi)
        self.assertEqual(pkt.rssi.get_value(), rssi)

        pkt.set_rssi(12)
        self.assertEqual(pkt.rssi.get_value(), "0C")

        channel = "13"

        pkt.set_channel(channel)
        self.assertEqual(pkt.channel.get_value(), channel)

        pkt.set_channel(13)
        self.assertEqual(pkt.channel.get_value(), "0D")

        with self.assertRaises(TypeError):
            pkt.set_rssi(12.4)


    def test_parse_mode_normal(self):
        pdu_string = "AA24320cfb574d5a02011a1aff4c000c0e009c6b8f40440f1583ec895148b410050318c0b525"
        access_address = "12345678"
        crc = "B8F7D4"

        pkt = Packet()

        pkt.from_string(access_address + pdu_string + crc)
        self.assertIsInstance(pkt.pdu, RawPDU)

        pkt.from_string(access_address + crc)
        self.assertIsInstance(pkt.pdu, NullPDU)

        with self.assertRaises(ParseError):
            pkt.from_string(access_address + "A" + crc)


    def test_extract_strict_policy(self):
        pdu_string = "4024320cfb574d5a02011a1aff4c000c0e009c6b8f40440f1583ec895148b410050318c0b525"
        access_address = "12345678"
        crc = "B8F7D4"

        # raw pdu
        pkt = Packet()

        pkt.from_string(access_address + pdu_string + crc)
        self.assertIsInstance(pkt.pdu, AdvInd)

        pkt.from_string(access_address + pdu_string + crc, parse_mode="strict")
        self.assertIsInstance(pkt.pdu, AdvInd)

        # empty pdu
        pdu_string = "4024320cfb574d5a02011a1aff4c000c0e009c6b8f40440f1583ec895148b410050318c0b525"
        access_address = "12345678"
        crc = "AAAAAA"

        pkt.from_string(access_address + pdu_string + crc)

        pkt.from_string(access_address + crc, parse_mode="strict")
        self.assertIsInstance(pkt.pdu, NullPDU)

        # single half hex field
        with self.assertRaises(ParseError):
            pkt.from_string(access_address + "A" + crc, parse_mode="strict")

    def test_tolerant_policy(self):
        pdu_string = "6F0C" + "ABCD12345678" + '02011A' + '020ACB'
        access_address = "12345678"
        crc = "AB12CD"

        # raw pdu
        pkt = Packet()

        pkt.from_string(access_address + pdu_string + crc, parse_mode="tolerant")
        self.assertIsInstance(pkt.pdu, RawPDU)

        # empty pdu
        pkt.from_string(access_address + crc, parse_mode="tolerant")
        self.assertIsInstance(pkt.pdu, NullPDU)

        # single half hex field
        pkt.from_string(access_address + "A" + crc, parse_mode="tolerant")
        self.assertEqual(pkt.to_string(), access_address + crc)

    def test_update(self):
        valid_string = "600C" + "ABCD12345678" + '02011A' + '020ACB'
        pdu_string = "60FF" + "ABCD12345678" + '02011A' + '020ACB'

        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode="tolerant")
        pkt.update()

        out = pkt.to_string()
        self.assertEqual(out[:-6], "D6BE898E" + valid_string)

        # just check that the crc changed
        self.assertNotEqual(crc, out[-6:])

    def test_update_crc(self):
        pdu_string = "4024320cfb574d5a02011a1aff4c000c0e009c6b8f40440f1583ec895148b410050318c0b525"
        access_address = "12345678"
        crc = "ABCDEF"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc)
        pkt.update()
        self.assertEqual(pkt.crc.to_string(), "B8F7D4")


    def test_get_time(self):
        pkt = Packet()
        pkt.time = datetime.datetime.fromtimestamp(1775170291.123)
        self.assertAlmostEqual(pkt.get_time(), 1775170291.123)

