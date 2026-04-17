import unittest

import torch

from evaluation_framework.scoring.score import Score


class TestScore(unittest.TestCase):
    def test_basic_score(self):
        scores = {"S1": 123.0,
                  "S2": torch.tensor([1, 2, 3]),
                  "S3": torch.zeros((2, 3))}

        score = Score(scores)

        self.assertAlmostEqual(score['S1'], 123.0, places=5)
        self.assertTrue(torch.allclose(score['S2'], torch.tensor([1, 2, 3]), atol=1e-6))
        self.assertTrue(torch.allclose(score['S3'], torch.zeros((2, 3)), atol=1e-6))

        for index, (key, value) in enumerate(scores.items()):
            if index == 0:
                self.assertEqual(key, 'S1')
                self.assertAlmostEqual(value, 123.0, places=5)
            elif index == 1:
                self.assertEqual(key, 'S2')
                self.assertTrue(torch.allclose(value, torch.tensor([1, 2, 3]), atol=1e-6))
            elif index == 2:
                self.assertEqual(key, 'S3')
                self.assertTrue(torch.allclose(value, torch.zeros((2, 3)), atol=1e-6))

        self.assertEqual(len(scores), 3)

        slice = score[['S3', 'S1']]

        self.assertIsInstance(slice, dict)
        self.assertEqual(list(slice.keys()), ['S3', 'S1'])
        self.assertAlmostEqual(slice['S1'], 123.0, places=5)
        self.assertTrue(torch.allclose(slice['S3'], torch.zeros((2, 3)), atol=1e-6))

        slice = score['S1']

        self.assertIsInstance(slice, float)
        self.assertAlmostEqual(slice, 123.0, places=5)

        slice = score[['S1']]

        self.assertIsInstance(slice, dict)
        self.assertEqual(list(slice.keys()), ['S1'])
        self.assertAlmostEqual(slice['S1'], 123.0, places=5)

    def test_nested_score(self):
        nested_score = Score({"S1": 123.0,
                              "S2": torch.tensor([1, 2, 3]),
                              "S3": torch.zeros((2, 3))}
                             )

        score = Score({'Nested': nested_score,
                       'S4': 1.0,
                       'S5': torch.ones((2, 3))})

        self.assertIsInstance(score['Nested'], Score)
        self.assertIsInstance(score['S4'], float)
        self.assertIsInstance(score['S5'], torch.Tensor)

    def test_assertions(self):
        with self.assertRaises(AssertionError) as e:
            Score([123.0])

        with self.assertRaises(AssertionError) as e:
            Score({'S1': 'ABC'})

        with self.assertRaises(AssertionError) as e:
            Score({123: 123.0})

        with self.assertRaises(AssertionError) as e:
            Score({'S1': 123})

        scores = {"S1": 123.0,
                  "S2": torch.tensor([1, 2, 3]),
                  "S3": torch.zeros((2, 3))}

        score = Score(scores)

        with self.assertRaises(AssertionError) as e:
            value = score[0]

        with self.assertRaises(AssertionError) as e:
            value = score[[0]]
