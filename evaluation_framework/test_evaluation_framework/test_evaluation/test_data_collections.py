import unittest

import torch

from evaluation_framework.collections.dynamic_data_collection import DynamicDataCollection
from evaluation_framework.collections.static_data_collection import StaticDataCollection


class TestDataCollection(unittest.TestCase):
    def test_static_collection(self):
        target = {'logits': torch.zeros(3), 'targets': torch.ones((3, 2))}
        collection = StaticDataCollection(target)
        data = collection.get_data()

        for i, (k, v) in enumerate(data.items()):
            self.assertTrue(torch.allclose(v, target[k], atol=1e-6))
            self.assertEqual(k, list(target.keys())[i])

        self.assertEqual(target['logits'].shape, data['logits'].shape)
        self.assertEqual(target['targets'].shape, data['targets'].shape)

        self.assertNotEqual(id(target), id(data))

        with self.assertRaises(AssertionError) as e:
            collection = StaticDataCollection({'logits': 123.0, 'targets': torch.ones((3, 2))})

    def test_dynamic_collection(self):
        original_data = {'logits': torch.zeros(3), 'targets': torch.ones((3, 2))}
        collection = DynamicDataCollection(original_data)
        collection.add_data(**original_data)

        target = {'logits': torch.zeros(3 * 2), 'targets': torch.ones((3 * 2, 2))}
        data = collection.get_data()

        self.assertEqual(target['logits'].shape, data['logits'].shape)
        self.assertEqual(target['targets'].shape, data['targets'].shape)

        for i, (k, v) in enumerate(data.items()):
            self.assertTrue(torch.allclose(v, target[k], atol=1e-6))
            self.assertEqual(k, list(target.keys())[i])

        collection.add_data(**{'logits': torch.zeros(2)})
        data = collection.get_data()
        self.assertEqual(data['logits'].shape, torch.Size([8]))
        self.assertEqual(data['targets'].shape, torch.Size([6, 2]))
