import unittest
import yaml
from pathlib import Path as LIBRARYPATH

from tqdm import tqdm

from ble.components.packet.Packet import Packet
from ble.masking.MaskConfig import MaskConfig


class TestMaskingConfig(unittest.TestCase):
    def test_masking_load(self):
        fixture_path = LIBRARYPATH(__file__).parent / "test_masking_config.yaml"

        with fixture_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        pdu_string = "6014" + "ABCD12345678" + '02011A' + '020A22' + '07FF4C0016345678'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')
        pkt.update()

        mask_config = MaskConfig()
        mask_config.from_dict(config)
        pkt.mask(mask_config)

        self.assertEqual(pkt.pdu.advertising_address.get_value(), "CD31EECF8CAB")

        adv_struct = pkt.pdu.adv_data[2]
        self.assertEqual(adv_struct.data.get_value(), "16CD31EE")

        adv_struct = pkt.pdu.adv_data[1]
        self.assertEqual(adv_struct.power_level.get_value(), "CD")

        pkt.print()

