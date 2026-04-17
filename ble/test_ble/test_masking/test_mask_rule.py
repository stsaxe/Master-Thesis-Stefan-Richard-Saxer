import unittest

from ble.fields.HexDataField import HexDataField
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.components.packet.Packet import Packet
from ble.masking.MaskConditions import AppleContinuityCondition
from ble.masking.MaskActions import MaskBleAddress
from ble.masking.MaskRule import MaskRule
from ble.walking.Path import Path
from ble.yaml.YamlRegistry import NullCondition, NullAction
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class TestMaskingRule(unittest.TestCase):
    def test_structure(self):
        rule = MaskRule()
        self.assertEqual(rule.get_priority(), 0)
        self.assertIsInstance(rule.condition, NullCondition)

        self.assertIsInstance(rule.get_path(), Path)
        self.assertEqual(rule.name, "")
        self.assertIsInstance(rule.action, NullAction)


    def test_from_dict_base_case(self):
        yaml = {"priority": 5, "name": "hello", "path": "**.advertising_address", "action": {"ref": "mask_ble_address", "args": {"length": 10, "rotation_type": "call", "token": "TEST"}},
                "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        rule = MaskRule()
        rule.from_dict(yaml)

        self.assertEqual(rule.name, "hello")
        self.assertEqual(rule.priority, 5)
        self.assertIsInstance(rule.action, MaskBleAddress)
        self.assertEqual(rule.action.token, "TEST")
        self.assertEqual(rule.action.pseudomizer.rotation_type, EpochRotation.CALL)
        self.assertEqual(rule.action.pseudomizer.epoch, 0)
        self.assertEqual(rule.action.length, 10)

        self.assertEqual(rule.get_path().segments[0].name, "**")
        self.assertEqual(rule.get_path().segments[1].name, "advertising_address")

        self.assertIsInstance(rule.condition, AppleContinuityCondition)
        self.assertEqual(rule.condition.continuity_type, ["12"])


    def test_execute(self):
        yaml = {"hello": "TEST", "path": "**.advertising_address", "action": {"ref": "mask_ble_address", "args": {"length": 10, "rotation_type": "packet", "token": "TOKEN"}},
                "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        rule = MaskRule()
        rule.from_dict(yaml)


        config = PseudomizerConfig()
        config.seed  = "TEST"
        config.epoch = 0

        rule.configure_pseudomizer(config)

        address = HexDataField("address")
        address.from_string("ABCD123456")

        rule.execute(address, pkt)
        self.assertEqual(address.get_value(), "CD31EECF8C")

        # test that it does not rotate the epoch
        address = HexDataField("address")
        address.from_string("ABCD123456")

        rule.execute(address, pkt)
        self.assertEqual(address.get_value(), "CD31EECF8C")


    def test_rotate_on_packet(self):
        yaml = {"hello": "TEST", "path": "**.advertising_address", "action": {"ref": "mask_ble_address", "args": {"length": 10, "rotation_type": "packet", "token": "TOKEN"}},
                "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        rule = MaskRule()
        rule.from_dict(yaml)


        config = PseudomizerConfig()
        config.seed  = "TEST"
        config.epoch = 0

        rule.configure_pseudomizer(config)
        rule.rotate_epoch("packet")

        address = HexDataField("address")
        address.from_string("ABCD123456")

        rule.execute(address, pkt)
        self.assertEqual(address.get_value(), "570BFE4753")

    def test_rotate_on_call(self):
        yaml = {"hello": "TEST", "path": "**.advertising_address", "action": {"ref": "mask_ble_address",
                                                                              "args": {"length": 10,
                                                                                       "rotation_type": "call",
                                                                                       "token": "TOKEN"}},
                "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        rule = MaskRule()
        rule.from_dict(yaml)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        rule.configure_pseudomizer(config)

        address = HexDataField("address")
        address.from_string("ABCD123456")

        rule.execute(address, pkt)
        self.assertEqual(address.get_value(), "CD31EECF8C")

        address = HexDataField("address")
        address.from_string("ABCD123456")

        rule.execute(address, pkt)
        self.assertEqual(address.get_value(), "570BFE4753")

    def test_condition_negative(self):
        yaml = {"hello": "TEST", "path": "**.advertising_address", "action": {"ref": "mask_ble_address",
                                                                              "args": {"length": 10,
                                                                                       "rotation_type": "call",
                                                                                       "token": "TOKEN"}},
                "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00AA3456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        rule = MaskRule()
        rule.from_dict(yaml)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        rule.configure_pseudomizer(config)

        address = HexDataField("address")
        address.from_string("ABCD123456")

        rule.execute(address, pkt)
        self.assertEqual(address.get_value(), "ABCD123456")
















