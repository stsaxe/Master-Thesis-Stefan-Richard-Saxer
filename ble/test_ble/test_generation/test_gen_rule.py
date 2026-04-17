import unittest

from ble.fields.HexDataField import HexDataField
from ble.generation.GenerationStrategies import DultProtocolStrategy
from ble.generation.GenerationRule import GenerationRule
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class TestGenRule(unittest.TestCase):
    def test_gen_rule(self):
        rule = GenerationRule()

        rule_dict = {"path": "**.packet",
                     "name": "TEST",
                     "action": {"ref": "dult_protocol",
                                "args": {"token": "TOKEN", "network_id": ['01'], "nearby": False,
                                         "payload_byte_length": 4}

                                }

                     }

        rule.from_dict(rule_dict)

        field = HexDataField("Test")

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        rule.configure_pseudomizer(config)

        rule.execute(field, None)

        self.assertEqual(rule.path.segments[1].name, "packet")
        self.assertEqual(rule.name, "TEST")
        self.assertIsInstance(rule.action, DultProtocolStrategy)
        self.assertEqual(field.get_value(), '010031EECF8C')

        rule.rotate_epoch('call')

        rule.execute(field, None)
        self.assertEqual(field.get_value(), '01000BFE4753')

