import unittest
from typing import Any, Iterable

from ble.components.MultiNodeContainer import MultiNodeContainer


class TestMultiNodeContainer(unittest.TestCase):
    def test_init(self):
        container = MultiNodeContainer("Test")
        self.assertEqual(container.get_name(), "Test")
        self.assertEqual(container.component_type, Any)
        self.assertEqual(container.components, [])

        container = MultiNodeContainer("Test", component_type=int)
        self.assertEqual(container.component_type, int)

        container = MultiNodeContainer("Test", [1,2,3], component_type=int)
        self.assertEqual(container.component_type, int)
        self.assertEqual(container.components, [1,2,3])

        with self.assertRaises(AssertionError):
            container = MultiNodeContainer("Test", component_type=123)


    def test_setter(self):
        container = MultiNodeContainer("Test")
        container.set_components([1,2,3])
        self.assertEqual(container.components, [1,2,3])

        with self.assertRaises(AssertionError):
            container = MultiNodeContainer("Test", component_type=int)
            container.set_components(["a", "b"])

    def test_len(self):
        container = MultiNodeContainer("Test", [1,2,3], component_type=int)
        self.assertEqual(len(container), 3)

    def test_clear(self):
        container = MultiNodeContainer("Test", [1,2,3], component_type=int)
        container.clear()
        self.assertEqual(container.components, [])
        self.assertEqual(len(container), 0)


    def test_iter(self):
        container = MultiNodeContainer("Test", [1, 2, 3], component_type=int)

        for idx, element in enumerate(container):
            self.assertEqual(idx+1, element)

        for idx, element in enumerate(container):
            if idx == 1:
                break

        for idx, element in enumerate(container):
            self.assertEqual(idx + 1, element)

        self.assertIsInstance(container, Iterable)

    def test_get_item(self):
        container = MultiNodeContainer("Test", [1, 2, 3, 4, 5, 6], component_type=int)
        self.assertEqual(container[1:5:2], [2,4])
        self.assertEqual(container[2], 3)


