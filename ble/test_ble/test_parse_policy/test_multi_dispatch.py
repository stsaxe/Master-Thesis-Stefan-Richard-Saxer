import unittest
from abc import abstractmethod, ABC
from typing import override

from ble.components.advertising_data.AdvertisingData import Flags
from ble.utils.HelperMethods import HelperMethods
from ble.parse_policy.MultiDispatch import MultiDispatchSupport, multidispatchmethod


class TestBaseClassMultiDispatch(MultiDispatchSupport, HelperMethods, ABC):
    @multidispatchmethod
    def run(self, *args):
        raise Exception("Not Implemented")

    @abstractmethod
    @run.register(int)
    def run_int(self, integer: int):
        pass

    @abstractmethod
    @run.register(str)
    def run_str(self, string: str):
        pass

    @run.register(Flags)
    def run_flags(self, struct: Flags):
        pass

class TestSubclassMultiDispatch(TestBaseClassMultiDispatch):
    def __init__(self):
        super().__init__()
        self.integer = 10
        self.string = 'Test'

    @TestBaseClassMultiDispatch.run.register(Flags)
    def run_flags(self, struct: Flags):
        print("hello")

    @override
    @TestBaseClassMultiDispatch.run.register(str)
    def run_str(self, string: str):
        return self.string + string

    @override
    @TestBaseClassMultiDispatch.run.register(int)
    def run_int(self, integer: int):
        return 3 * integer + self.integer

    @TestBaseClassMultiDispatch.run.register(str, int)
    def run_str_int(self, string: str, integer: int):
        return self.string + string + f"{2*integer}"

class TestOtherSubclassMultiDispatch(TestBaseClassMultiDispatch):
    @override
    @TestBaseClassMultiDispatch.run.register(str)
    def run_str(self, string: str):
        return "ABC" + string

    @override
    @TestBaseClassMultiDispatch.run.register(int)
    def run_int(self, integer: int):
        return 5 * integer


class TestMultiDispatch(unittest.TestCase):
    def test_simple_case(self):
        first = TestSubclassMultiDispatch()

        self.assertEqual(first.run(1), 13)
        self.assertEqual(first.run("ABC"), "TestABC")

        second = TestOtherSubclassMultiDispatch()

        self.assertEqual(second.run(1), 5)
        self.assertEqual(second.run("ABC"), "ABCABC")

    def test_multiple_arguments(self):
        first = TestSubclassMultiDispatch()
        self.assertEqual(first.run("ABC", 123), "TestABC246")

    def test_not_implemented(self):
        first = TestSubclassMultiDispatch()
        with self.assertRaises(Exception):
            first.run(123, "ABC")

