import unittest

from ble import ManufacturerSpecific, TxPowerLevel, HexDataField
from ble.components.packet.Packet import Packet
from ble.walking.RetrievalRuntime import RetrievalRuntime
from ble.walking.Path import Path


class TestRetrievalRuntime(unittest.TestCase):
    def test_structure(self):
        path_string = "**"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        self.assertEqual(runtime.values, {target_path: []})
        self.assertEqual(runtime.target_path, [target_path])

    def test_simple_walk_full_path(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct[adv_type=0xFF].length"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        self.assertEqual(runtime.get_values()[target_path][0].get_value(), "06")
        self.assertEqual(len(runtime.get_values()[target_path]), 1)

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct[adv_type=0xFF].data"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        self.assertEqual(runtime.get_values()[target_path][0].get_value(), '123456')

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct.data"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        self.assertEqual(runtime.get_values()[target_path][0].get_value(), '123456')

        runtime.reset()

        self.assertEqual(runtime.get_values(), {target_path: []})

    def test_simple_walk_multiple_values(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct.length"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        values = [f.get_value() for f in runtime.get_values()[target_path]]

        self.assertEqual(['02', '02', '06'], values)

    def test_simple_walk_multiple_values_star_path(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "**.adv_struct.length"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        values = [f.get_value() for f in runtime.get_values()[target_path]]

        self.assertEqual(['02', '02', '06'], values)

    def test_simple_walk_multiple_values_full_star_path_wit(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "**.length"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        values = [f.get_value() for f in runtime.get_values()[target_path]]

        self.assertEqual(['13', '02', '02', '06'], values)

    def test_invalid_path(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct[adv_type=0xAB].data"
        target_path = Path()
        target_path.from_string(path_string)

        runtime = RetrievalRuntime(target_path)

        _ = pkt.walk(runtime)

        self.assertEqual(len(runtime.get_values()[target_path]), 0)

    def test_retrieve_multiple_values(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct[adv_type=0xff, adv_type=0x0a]"
        target_path_1 = Path()
        target_path_1.from_string(path_string)

        path_string = "packet.pdu.packet_header.length"
        target_path_2 = Path()
        target_path_2.from_string(path_string)

        runtime = RetrievalRuntime([target_path_1, target_path_2])

        _ = pkt.walk(runtime)

        self.assertIsInstance(runtime.get_values()[target_path_1][1], ManufacturerSpecific)
        self.assertIsInstance(runtime.get_values()[target_path_1][0], TxPowerLevel)
        self.assertIsInstance(runtime.get_values()[target_path_2][0], HexDataField)

    def test_retrieve_multiple_identical_values(self):
        pdu_string = "6013" + "ABCD12345678" + '020ACB' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        path_string = "packet.pdu[pdu_type=ADV_IND].advertising_data.adv_struct[adv_type=0xff, adv_type=0x0a]"
        target_path_1 = Path()
        target_path_1.from_string(path_string)

        path_string = "packet.pdu.packet_header.length"
        target_path_2 = Path()
        target_path_2.from_string(path_string)

        runtime = RetrievalRuntime([target_path_1, target_path_2])

        _ = pkt.walk(runtime)

        self.assertIsInstance(runtime.get_values()[target_path_1][2], ManufacturerSpecific)
        self.assertIsInstance(runtime.get_values()[target_path_1][1], TxPowerLevel)
        self.assertIsInstance(runtime.get_values()[target_path_1][0], TxPowerLevel)
        self.assertNotEqual(id(runtime.get_values()[target_path_1][0]), id(runtime.get_values()[target_path_1][1]))
        self.assertIsInstance(runtime.get_values()[target_path_2][0], HexDataField)

