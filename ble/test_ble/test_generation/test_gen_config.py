import unittest
from pathlib import Path as LIBRARYPATH

from tqdm import tqdm

from ble import AdvInd, ServiceUUID16ListIncomplete, ManufacturerSpecific
from ble.components.packet.Packet import Packet
from ble.generation.GenConfig import GenConfig


class TestGenConfig(unittest.TestCase):
    def test_gen_config(self):
        gen_config = GenConfig()
        gen_config.from_yaml(str(LIBRARYPATH(__file__).parent / "test_generation_config.yaml"))


        pkt = Packet()
        pkt.generate(gen_config)
        pkt.update()


        self.assertIsInstance(pkt.pdu, AdvInd)
        self.assertIsInstance(pkt.pdu.adv_data[0], ServiceUUID16ListIncomplete)
        self.assertIsInstance(pkt.pdu.adv_data[1], ManufacturerSpecific)
        self.assertEqual(len(pkt.pdu.adv_data[0].uuids), 1)
        self.assertEqual(pkt.pdu.adv_data[1].company_id.get_value(), "AB12")

        data_1 = pkt.pdu.adv_data[1].data.get_value()


        pkt.generate(gen_config)
        pkt.update()

        self.assertIsInstance(pkt.pdu, AdvInd)
        self.assertIsInstance(pkt.pdu.adv_data[0], ServiceUUID16ListIncomplete)
        self.assertIsInstance(pkt.pdu.adv_data[1], ManufacturerSpecific)
        self.assertEqual(pkt.pdu.adv_data[1].company_id.get_value(), "AB12")
        self.assertEqual(len(pkt.pdu.adv_data[0].uuids), 2)


        data_2 = pkt.pdu.adv_data[1].data.get_value()

        self.assertNotEqual(data_1, data_2)


        gen_config.rotate_epoch('packet')
        pkt.generate(gen_config)
        pkt.update()

        self.assertEqual(pkt.pdu.adv_data[1].company_id.get_value(), "1234")


        pkt.print()






