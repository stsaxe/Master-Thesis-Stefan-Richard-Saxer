import unittest

from ble.masking.MaskScope import MaskScope
from ble.fields.HexDataField import HexDataField
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.components.packet.Packet import Packet
from ble.masking.MaskConditions import AppleContinuityCondition
from ble.masking.MaskActions import MaskBleAddress
from ble.walking.Path import Path
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class TestMaskScopes(unittest.TestCase):
    def test_read(self):
        rule_1 = {"priority": 3, "name": "rule_1", "path": "**.advertising_address",
                  "action": {"ref": "mask_ble_address",
                             "args": {"length": 10, "rotation_type": "call", "token": "TEST"}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 10}}}

        rule_2 = {"priority": 5, "name": "rule_2", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TEST"}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        scope_yaml = {"name": "my_scope",
                      "priority": 3,
                      "path": "packet.pdu[pdu_type=0000]",
                      "rules": [rule_2, rule_1]
                      }

        scope = MaskScope()
        scope.from_dict(scope_yaml)

        self.assertEqual(scope.get_name(), "my_scope")
        self.assertEqual(scope.get_priority(), 3)
        self.assertEqual(scope.get_path().segments[0].name, "packet")
        self.assertEqual(scope.get_path().segments[1].name, "pdu")

        rule = scope.rules[0]

        self.assertEqual(rule.name, "rule_1")
        self.assertEqual(rule.priority, 3)
        self.assertIsInstance(rule.action, MaskBleAddress)
        self.assertEqual(rule.action.token, "TEST")
        self.assertEqual(rule.action.pseudomizer.rotation_type, EpochRotation.CALL)
        self.assertEqual(rule.action.pseudomizer.epoch, 0)
        self.assertEqual(rule.action.length, 10)

        self.assertEqual(rule.get_path().segments[0].name, "**")
        self.assertEqual(rule.get_path().segments[1].name, "advertising_address")

        self.assertIsInstance(rule.condition, AppleContinuityCondition)
        self.assertEqual(rule.condition.continuity_type, ["10"])

        rule = scope.rules[1]
        self.assertEqual(rule.name, "rule_2")

    def test_execute(self):
        rule_1 = {"priority": 3, "name": "rule_1", "path": "**.advertising_address",
                  "action": {"ref": "mask_ble_address",
                             "args": {"length": 10, "rotation_type": "packet", "token": "TOKEN"}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        rule_2 = {"priority": 5, "name": "rule_2", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TOKEN"}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        scope_yaml = {"name": "my_scope",
                      "priority": 3,
                      "path": "packet.pdu",
                      "rules": [rule_2, rule_1]
                      }

        scope = MaskScope()
        scope.from_dict(scope_yaml)

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        scope.configure_pseudomizer(config)

        current_path_string = "packet.pdu[pdu_type=0000].advertising_data.adv_struct[adv_type=0xff].data"
        current_path_string = "packet.pdu[pdu_type=0000]"

        path = Path()
        path.from_string(current_path_string)

        scope_path = Path()
        scope_path.from_string("packet.pdu[pdu_type=0000]")

        scope.try_activation_for(scope_path)
        self.assertEqual(scope.is_active(), True)

        field = HexDataField("advertising_address", value="0123456789")
        scope.apply_rules_to_field(field, path, pkt)
        self.assertEqual(field.get_value(), "CD31EECF8C")

        field = HexDataField("advertising_data", value="0123456789")
        scope.apply_rules_to_field(field, path, pkt)
        self.assertEqual(field.get_value(), "0123456789")

        field = HexDataField("advertising_data", value="0123456789")
        scope.apply_rules_to_field(field, path, pkt)
        self.assertEqual(field.get_value(), "0123456789")

        scope.try_deactivation_for(scope_path)

        self.assertEqual(scope.is_active(), False)

    def test_multi_execute(self):
        rule_1 = {"priority": 3, "name": "rule_1", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TOKEN", "end": 4}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}
        rule_2 = {"priority": 5, "name": "rule_2", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TOKEN", "start": 8}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        scope_yaml = {"name": "my_scope",
                      "priority": 1,
                      "path": "packet.pdu",
                      "rules": [rule_2, rule_1]
                      }

        scope = MaskScope()
        scope.from_dict(scope_yaml)

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        scope.configure_pseudomizer(config)

        scope_path = Path()
        scope_path.from_string("packet.pdu[pdu_type=0000]")

        scope.try_activation_for(scope_path)

        current_path_string = "packet.pdu[pdu_type=0000].advertising_data.adv_struct[adv_type=0xff]"
        current_path = Path()
        current_path.from_string(current_path_string)

        field = HexDataField("data", value="0123456789ABCD")
        scope.apply_rules_to_field(field, current_path, pkt)

        self.assertEqual(field.get_value(), "CD314567CD31EE")

    def test_multi_execute_order(self):
        rule_1 = {"priority": 3, "name": "rule_1", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TOKEN", "end": 4}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}
        rule_2 = {"priority": 2, "name": "rule_2", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TEST", "end": 4}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        scope_yaml = {"name": "my_scope",
                      "priority": 1,
                      "path": "packet.pdu",
                      "rules": [rule_2, rule_1]
                      }

        scope = MaskScope()
        scope.from_dict(scope_yaml)

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        scope.configure_pseudomizer(config)

        scope_path = Path()
        scope_path.from_string("packet.pdu[pdu_type=0000]")

        scope.try_activation_for(scope_path)

        current_path_string = "packet.pdu[pdu_type=0000].advertising_data.adv_struct[adv_type=0xff]"
        current_path = Path()
        current_path.from_string(current_path_string)

        field = HexDataField("data", value="0123456789ABCD")
        scope.apply_rules_to_field(field, current_path, pkt)

        self.assertEqual(field.get_value(), "CD31456789ABCD")

    def test_multi_execute_condition(self):
        rule_1 = {"priority": 3, "name": "rule_1", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TOKEN", "end": 4}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}
        rule_2 = {"priority": 2, "name": "rule_2", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TEST", "start": 8}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 10}}}

        scope_yaml = {"name": "my_scope",
                      "priority": 1,
                      "path": "packet.pdu",
                      "rules": [rule_2, rule_1]
                      }

        scope = MaskScope()
        scope.from_dict(scope_yaml)

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        scope.configure_pseudomizer(config)

        scope_path = Path()
        scope_path.from_string("packet.pdu[pdu_type=0000]")

        scope.try_activation_for(scope_path)

        current_path_string = "packet.pdu[pdu_type=0000].advertising_data.adv_struct[adv_type=0xff]"
        current_path = Path()
        current_path.from_string(current_path_string)

        field = HexDataField("data", value="0123456789ABCD")
        scope.apply_rules_to_field(field, current_path, pkt)

        self.assertEqual(field.get_value(), "CD31456789ABCD")

    def test_execution_when_inactive(self):
        rule_1 = {"priority": 3, "name": "rule_1", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TOKEN", "end": 4}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}
        rule_2 = {"priority": 5, "name": "rule_2", "path": "**.adv_struct[adv_type=0xff].data",
                  "action": {"ref": "mask_hex_data", "args": {"rotation_type": "call", "token": "TEST"}},
                  "condition": {"ref": "is_apple_continuity", "args": {"continuity_type": 12}}}

        scope_yaml = {"name": "my_scope",
                      "priority": 1,
                      "path": "packet.pdu",
                      "rules": [rule_2, rule_1]
                      }

        scope = MaskScope()
        scope.from_dict(scope_yaml)

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        scope.configure_pseudomizer(config)

        scope_path = Path()
        scope_path.from_string("packet.crc")

        scope.try_activation_for(scope_path)

        current_path_string = "packet.pdu[pdu_type=0000].advertising_data.adv_struct[adv_type=0xff]"
        current_path = Path()
        current_path.from_string(current_path_string)

        field = HexDataField("data", value="0123456789ABCD")
        scope.apply_rules_to_field(field, current_path, pkt)

        self.assertFalse(scope.is_active())
        self.assertEqual(field.get_value(), "0123456789ABCD")
