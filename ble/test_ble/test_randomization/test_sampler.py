import unittest
from collections import Counter

from torch.utils.data import WeightedRandomSampler

from ble import WeightedItem
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.SamplingPseudomizer import PseudoRandomSampler, WeightedRange
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class TestSampler(unittest.TestCase):
    def test_initialization(self):
        pseudo = PseudoRandomSampler("packet")

        self.assertEqual(pseudo.seed, " ")
        self.assertEqual(pseudo.epoch, 0)
        self.assertEqual(pseudo.rotation_type, EpochRotation.PACKET)

        pseudo = PseudoRandomSampler(EpochRotation.STREAM)
        self.assertEqual(pseudo.rotation_type, EpochRotation.STREAM)

        with self.assertRaises(TypeError):
            pseudo = PseudoRandomSampler(None)

    def test_configure(self):
        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 1

        pseudo = PseudoRandomSampler("packet")
        pseudo.configure_pseudomizer(config)

        self.assertEqual(pseudo.seed, "TEST")
        self.assertEqual(pseudo.epoch, 1)

    def test_determinism(self):
        pseudo = PseudoRandomSampler("packet")
        pseudo.epoch = 0
        pseudo.seed = 'TEST'

        w1 = WeightedRange(0, 1000, 100)
        w2 = WeightedRange(1001, 2000, 100)
        w3 = WeightedRange(2001, 3000, 100)

        sample = pseudo.sample_int("TOKEN", [w1, w2, w3])
        self.assertEqual(sample, 2893)

        sample = pseudo.sample_int("TOKEN", [w1, w2, w3])
        self.assertEqual(sample, 2893)

        sample = pseudo.sample_int("MYTOKEN", [w1, w2, w3])
        self.assertEqual(sample, 2189)

    def test_rotate_epoch(self):
        pseudo = PseudoRandomSampler("packet")
        pseudo.epoch = 0
        pseudo.seed = 'TEST'

        w1 = WeightedRange(0, 1000, 100)
        w2 = WeightedRange(1001, 2000, 100)
        w3 = WeightedRange(2001, 3000, 100)

        sample = pseudo.sample_int("TOKEN", [w1, w2, w3])
        self.assertEqual(sample, 2893)

        pseudo.rotate_epoch("call")

        sample = pseudo.sample_int("TOKEN", [w1, w2, w3])
        self.assertEqual(sample, 2893)

        pseudo.rotate_epoch("packet")

        sample = pseudo.sample_int("TOKEN", [w1, w2, w3])
        self.assertEqual(sample, 1485)

    def test_sample_int(self):
        pseudo = PseudoRandomSampler("packet")
        pseudo.epoch = 0
        pseudo.seed = 'TEST'

        dist = [
            WeightedRange(0, 1, 4),
            WeightedRange(2, 2, 3),
            WeightedRange(3, 3, 2),
            WeightedRange(4, 4, 1),
        ]

        n = 10_000
        counts = Counter(
            pseudo.sample_int(f"TOKEN_{i}", dist)
            for i in range(n)
        )

        freq0 = counts[0] / n
        freq1 = counts[1] / n
        freq2 = counts[2] / n
        freq3 = counts[3] / n
        freq4 = counts[4] / n

        self.assertAlmostEqual(freq0, 0.2, delta=0.01)
        self.assertAlmostEqual(freq1, 0.2, delta=0.01)
        self.assertAlmostEqual(freq2, 0.3, delta=0.01)
        self.assertAlmostEqual(freq3, 0.2, delta=0.01)
        self.assertAlmostEqual(freq4, 0.1, delta=0.01)

    def test_sample_item(self):
        pseudo = PseudoRandomSampler("packet")

        item_0 = WeightedItem("Item0", 1)
        item_1 = WeightedItem("Item1", 2)
        item_2 = WeightedItem("Item2", 3)
        item_3 = WeightedItem("Item3", 4)

        items = [item_0, item_1, item_2, item_3]

        n = 10_000
        counts = Counter(
            pseudo.sample_item(f"TOKEN_{i}", items)
            for i in range(n)
        )

        freq0 = counts['Item0'] / n
        freq1 = counts['Item1'] / n
        freq2 = counts['Item2'] / n
        freq3 = counts['Item3'] / n

        self.assertAlmostEqual(freq0, 0.1, delta=0.01)
        self.assertAlmostEqual(freq1, 0.2, delta=0.01)
        self.assertAlmostEqual(freq2, 0.3, delta=0.01)
        self.assertAlmostEqual(freq3, 0.4, delta=0.01)

    def test_sample_k_items_without_replacement(self):
        pseudo = PseudoRandomSampler("packet")

        item_0 = WeightedItem("Item0", 1)
        item_1 = WeightedItem("Item1", 2)
        item_2 = WeightedItem("Item2", 3)
        item_3 = WeightedItem("Item3", 4)

        items = [item_0, item_1, item_2, item_3]

        n = 10_000
        counts = Counter(
            tuple(pseudo.sample_k_items_without_replacement(f"TOKEN_{i}", k=2, items=items))
            for i in range(n)
        )

        for sample in counts:
            self.assertEqual(len(sample), 2)
            self.assertEqual(len(set(sample)), 2)

        exp_freq_item3_item2 = 4 / (4 + 3 + 2 + 1) * 3 / (3 + 2 + 1)

        freq_item3_item2 = counts[('Item3', 'Item2')] / n
        self.assertAlmostEqual(freq_item3_item2, exp_freq_item3_item2, delta=0.01)

        exp_freq_item2_item3 = 3 / (4 + 3 + 2 + 1) * 4 / (4 + 2 + 1)
        freq_item2_item3 = counts[('Item2', 'Item3')] / n
        self.assertAlmostEqual(freq_item2_item3, exp_freq_item2_item3, delta=0.01)

    def test_sample_k_items_with_replacement(self):
        pseudo = PseudoRandomSampler("packet")

        item_0 = WeightedItem("Item0", 1)
        item_1 = WeightedItem("Item1", 2)
        item_2 = WeightedItem("Item2", 3)
        item_3 = WeightedItem("Item3", 4)

        items = [item_0, item_1, item_2, item_3]

        n = 10_000
        counts = Counter(
            tuple(pseudo.sample_k_items_with_replacement(f"TOKEN_{i}", k=2, items=items))
            for i in range(n)
        )

        for sample in counts:
            self.assertEqual(len(sample), 2)

        exp_freq_item3_item2 = 4 / (4 + 3 + 2 + 1) * 3 / (4 + 3 + 2 + 1)

        freq_item3_item2 = counts[('Item3', 'Item2')] / n
        self.assertAlmostEqual(freq_item3_item2, exp_freq_item3_item2, delta=0.01)

        freq_item2_item3 = counts[('Item2', 'Item3')] / n
        self.assertAlmostEqual(freq_item2_item3, exp_freq_item3_item2, delta=0.01)

        exp_freq_item3_item3 = (4 / (4 + 3 + 2 + 1)) ** 2
        freq_item3_item3 = counts[('Item3', 'Item3')] / n
        self.assertAlmostEqual(exp_freq_item3_item3, freq_item3_item3, delta=0.01)



