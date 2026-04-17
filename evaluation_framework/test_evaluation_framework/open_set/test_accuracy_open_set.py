import unittest

import torch

from evaluation_framework.metrics.open_set.accuracy_open_set import AccuracyOpenSet


class TestAccuracyAtFPR(unittest.TestCase):
    def test_accuracy_micro_garbage(self):
        logits = [[5, 2, 3, 4],
                  [1, 2, 5, 4],
                  [1, 2, 3, 4],

                  [1, 5, 3, 4],
                  [1, 5, 3, 4],
                  [1, 2, 3, 4],

                  [1, 2, 5, 4],
                  [1, 5, 2, 3],

                  [1, 2, 3, 4]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3]

        # 0: 1 / 3
        # 1: 2 / 3
        # 2: 1 / 2
        # 3: 1 / 1

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='micro', unknown_label=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(5 / 9, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (micro)', name)

    def test_accuracy_macro_garbage(self):
        logits = [[5, 2, 3, 4],
                  [1, 2, 5, 4],
                  [1, 2, 3, 4],

                  [1, 5, 3, 4],
                  [1, 5, 3, 4],
                  [1, 2, 3, 4],

                  [1, 2, 5, 4],
                  [1, 5, 2, 3],

                  [1, 2, 3, 4]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3]

        # 0: 1 / 3
        # 1: 2 / 3
        # 2: 1 / 2
        # 3: 1 / 1

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='macro', unknown_label=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((1 / 3 + 2 / 3 + 1 / 2 + 1) / 4, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (macro)', name)

    def test_accuracy_weighted_garbage(self):
        logits = [[5, 2, 3, 4],
                  [1, 2, 5, 4],
                  [1, 2, 3, 4],

                  [1, 5, 3, 4],
                  [1, 5, 3, 4],
                  [1, 2, 3, 4],

                  [1, 2, 5, 4],
                  [1, 5, 2, 3],

                  [1, 2, 3, 4]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3]

        # 0: 1 / 3
        # 1: 2 / 3
        # 2: 1 / 2
        # 3: 1 / 1

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='weighted', unknown_label=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual((3 * 1 / 3 + 3 * 2 / 3 + 2 * 1 / 2 + 1) / 9, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (weighted)', name)

    def test_accuracy_balanced_garbage(self):
        logits = [[5, 2, 3, 4],
                  [1, 2, 5, 4],
                  [1, 2, 3, 4],

                  [1, 5, 3, 4],
                  [1, 5, 3, 4],
                  [1, 2, 3, 4],

                  [1, 2, 5, 4],
                  [1, 5, 2, 3],

                  [1, 2, 3, 4]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3]

        # 0: 1 / 3
        # 1: 2 / 3
        # 2: 1 / 2
        # 3: 1 / 1

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='balanced', unknown_label=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(0.5 * (1 + 1 + 1) / 6 + 0.5 * 2 / 3, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (balanced)', name)

    def test_accuracy_balanced_garbage_v2(self):
        logits = [[5, 2, 3, 4],
                  [1, 2, 5, 4],
                  [1, 2, 3, 4],

                  [1, 5, 3, 4],
                  [1, 5, 3, 4],
                  [1, 2, 3, 4],

                  [1, 2, 5, 4],
                  [1, 5, 2, 3],
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2]

        # 0: 1 / 3
        # 1: 2 / 3
        # 2: 1 / 2

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='balanced', unknown_label=2)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(0.5 * (1 + 2) / 6 + 0.5 * 1 / 2, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (balanced)', name)

    def test_accuracy_binary_garbage(self):
        logits = [[5, 2, 3, 4],
                  [1, 2, 5, 4],
                  [1, 2, 3, 4],

                  [1, 5, 3, 4],
                  [1, 5, 3, 4],
                  [1, 2, 3, 4],

                  [1, 2, 5, 4],
                  [1, 5, 2, 3],

                  [1, 2, 3, 4]
                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 3]

        # 0: 1 / 3
        # 1: 2 / 3
        # 2: 1 / 2
        # 3: 1 / 1

        # known: 5 / 6
        # unknown: 2 / 3

        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='binary', unknown_label=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, float)
        self.assertAlmostEqual(7 / 9, accuracy, places=5)

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (binary)', name)

    def test_accuracy_micro_thresholding(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [40, 60],
                  [90, 10],

                  [70, 30],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='micro', unknown_label=2, precision=0)

        solution = torch.tensor([3 / 8, 3 / 8])

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertTrue(torch.allclose(solution, accuracy, atol=1e-4))

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (micro)', name)

        metric = AccuracyOpenSet(average='micro', unknown_label=2, precision=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertEqual(accuracy[7].float(), 0.375)
        self.assertEqual(accuracy[8].float(), 0.5)
        self.assertEqual(accuracy[9].float(), 0.375)

    def test_accuracy_macro_thresholding(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [40, 60],
                  [90, 10],

                  [70, 30],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='macro', unknown_label=2, precision=0)

        solution = torch.tensor([(2 / 3 + 1 / 2 + 0 / 3) / 3, (0 + 0 + 3 / 3) / 3])

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertTrue(torch.allclose(solution, accuracy, atol=1e-4))

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (macro)', name)

        metric = AccuracyOpenSet(average='macro', unknown_label=2, precision=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(accuracy[6].float()), (2 / 3 + 1 / 2 + 0 / 3) / 3, places=3)
        self.assertAlmostEqual(float(accuracy[7].float()), (2 / 3 + 0 / 2 + 1 / 3) / 3, places=3)
        self.assertAlmostEqual(float(accuracy[8].float()), (2 / 3 + 0 / 2 + 2 / 3) / 3, places=3)
        self.assertAlmostEqual(float(accuracy[9].float()), (1 / 3 + 0 / 2 + 2 / 3) / 3, places=3)
        self.assertAlmostEqual(float(accuracy[9].float()), (0 / 3 + 0 / 2 + 3 / 3) / 3, places=3)

    def test_accuracy_weighted_thresholding(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [40, 60],
                  [90, 10],

                  [70, 30],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='weighted', unknown_label=2, precision=0)

        solution = torch.tensor([(3 * 2 / 3 + 2 * 1 / 2 + 3 * 0 / 3) / 8, (0 + 0 + 3 * 3 / 3) / 8])

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertTrue(torch.allclose(solution, accuracy, atol=1e-4))

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (weighted)', name)

        metric = AccuracyOpenSet(average='weighted', unknown_label=2, precision=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(accuracy[6].float()), (2 + 1 + 0) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[7].float()), (2 + 0 + 1) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[8].float()), (2 + 0 + 2) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[9].float()), (1 + 0 + 2) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[9].float()), (0 + 0 + 3) / 8, places=3)

    def test_accuracy_balanced_thresholding(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [40, 60],
                  [90, 10],

                  [70, 30],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='balanced', unknown_label=2, precision=0)

        solution = torch.tensor([((2 + 1) / 5 + 0) / 2, (0 + 1) / 2])

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertTrue(torch.allclose(solution, accuracy, atol=1e-4))

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (balanced)', name)

        metric = AccuracyOpenSet(average='balanced', unknown_label=2, precision=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(accuracy[6].float()), ((2 + 1) / 5 + 0 / 3) / 2, places=3)
        self.assertAlmostEqual(float(accuracy[7].float()), ((2 + 0) / 5 + 1 / 3) / 2, places=3)
        self.assertAlmostEqual(float(accuracy[8].float()), ((2 + 0) / 5 + 2 / 3) / 2, places=3)
        self.assertAlmostEqual(float(accuracy[9].float()), ((1 + 0) / 5 + 2 / 3) / 2, places=3)

    def test_accuracy_binary_thresholding(self):
        logits = [[90, 10],
                  [40, 60],
                  [80, 20],

                  [40, 60],
                  [90, 10],

                  [70, 30],
                  [60, 40],
                  [90, 10]
                  ]

        targets = [0, 0, 0, 1, 1, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        metric = AccuracyOpenSet(average='binary', unknown_label=2, precision=0)

        solution = torch.tensor([5 / 8, 3 / 8])

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertTrue(torch.allclose(solution, accuracy, atol=1e-4))

        name = metric.get_name()

        self.assertEqual('Accuracy Open Set (binary)', name)

        metric = AccuracyOpenSet(average='binary', unknown_label=2, precision=1)

        accuracy = metric.compute(logits=logits, targets=targets)

        self.assertAlmostEqual(float(accuracy[6].float()), (5 + 0) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[7].float()), (3 + 1) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[8].float()), (3 + 2) / 8, places=3)
        self.assertAlmostEqual(float(accuracy[9].float()), (2 + 2) / 8, places=3)

    def test_assertions(self):
        with self.assertRaises(AssertionError):
            AccuracyOpenSet(average='test', unknown_label=1)

        with self.assertRaises(AssertionError):
            AccuracyOpenSet(average=123, unknown_label=1)

        with self.assertRaises(AssertionError):
            AccuracyOpenSet(unknown_label=-1)
