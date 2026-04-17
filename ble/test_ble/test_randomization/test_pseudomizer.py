import unittest

from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.HexPseudomizer import HexPseudomizer
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class TestHexPseudomizer(unittest.TestCase):
    def test_initialization(self):
        pseudo = HexPseudomizer("packet")

        self.assertEqual(pseudo.seed, " ")
        self.assertEqual(pseudo.epoch, 0)
        self.assertEqual(pseudo.rotation_type, EpochRotation.PACKET)

        pseudo = HexPseudomizer(EpochRotation.STREAM)
        self.assertEqual(pseudo.rotation_type, EpochRotation.STREAM)

        with self.assertRaises(TypeError):
            pseudo = HexPseudomizer(None)

    def test_pseudomization(self):
        pseudo = HexPseudomizer(EpochRotation.STREAM)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        pseudo.configure_pseudomizer(config)

        value = pseudo.pseudomize("TOKEN", 10)
        self.assertEqual(len(value), 10)

        target = 'CD31EECF8C'

        self.assertEqual(target, value)
        self.assertEqual(len(value), 10)

        value = pseudo.pseudomize("TOKEN", 0)
        self.assertEqual(value, "")

        value = pseudo.pseudomize("TOKEN", 10)
        self.assertEqual(target, value)


    def test_impact_of_seed(self):
        pseudo = HexPseudomizer(EpochRotation.STREAM)

        config = PseudomizerConfig()
        config.seed = "ABC"
        config.epoch = 0

        pseudo.configure_pseudomizer(config)

        value = pseudo.pseudomize("TOKEN", 10)
        self.assertNotEqual(value, "CD31EECF8C")


    def test_impact_of_epoch(self):
        pseudo = HexPseudomizer(EpochRotation.STREAM)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 1

        pseudo.configure_pseudomizer(config)

        value = pseudo.pseudomize("TOKEN", 10)

        self.assertEqual(value, "570BFE4753")

    def test_impact_of_token(self):
        pseudo = HexPseudomizer(EpochRotation.STREAM)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        pseudo.configure_pseudomizer(config)

        value = pseudo.pseudomize("ABC", 10)
        self.assertNotEqual(value, "CD31EECF8C")

    def test_rotate(self):
        pseudo = HexPseudomizer(EpochRotation.STREAM)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        pseudo.configure_pseudomizer(config)


        target = 'CD31EECF8C'
        value = pseudo.pseudomize("TOKEN", 10)
        self.assertEqual(target, value)

        pseudo.rotate_epoch(EpochRotation.CALL)
        value = pseudo.pseudomize("TOKEN", 10)
        self.assertEqual(target, value)

        pseudo.rotate_epoch(EpochRotation.STREAM)
        value = pseudo.pseudomize("TOKEN", 10)
        self.assertNotEqual(target, value)

    def test_rotate_never(self):
        pseudo = HexPseudomizer(EpochRotation.NEVER)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 0

        pseudo.configure_pseudomizer(config)

        value = pseudo.pseudomize("TOKEN", 10)
        self.assertEqual(len(value), 10)

        target = 'CD31EECF8C'

        self.assertEqual(target, value)

        pseudo.rotate_epoch("never")

        value = pseudo.pseudomize("TOKEN", 10)
        self.assertEqual(len(value), 10)

        target = 'CD31EECF8C'

        self.assertEqual(target, value)













