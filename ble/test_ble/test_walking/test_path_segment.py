import unittest

from ble.walking.PathSegment import PathSegment
from ble.walking.Path import Path


class TestPathSegment(unittest.TestCase):
    def test_initialization(self) -> None:
        segment = PathSegment("test")

        self.assertEqual(segment.name, "test")

        keys = {"abc": ['a', 'b', 'c'], "xxx": "xyz"}

        segment = PathSegment("test", keys=keys)
        self.assertEqual(segment.name, "test")
        self.assertEqual(segment.keys, {"abc": ['a', 'b', 'c'], "xxx": ["xyz"]})

        complex_keys = {r"  \/(Ab Z-19_) ": r"  \/(AXy z19_-) ", "test": "  *   "}
        complex_name = r"  \/(aAb Z19_-) "

        segment = PathSegment(complex_name, keys=complex_keys)

        self.assertEqual(segment.name, r"\/(aab_z19_-)")
        self.assertEqual(segment.keys, {r"\/(ab_z-19_)": [r"\/(axy_z19_-)"], "test": ["*"]})

    def test_invalid_name_key_and_value(self):
        with self.assertRaises(AssertionError):
            keys = {"abc": ['a*', 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            keys = {"abc*": ['a*', 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            keys = {"*": ['a', 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            keys = {"abc": ['a.c', 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            keys = {"a.bc": ['ac', 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            segment = PathSegment("te.st")

        with self.assertRaises(AssertionError):
            keys = {"abc": [2, 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            keys = {123: ["b", 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):
            keys = {"abc": ["b", 'b', 'c'], "xxx": ["xyz"]}
            segment = PathSegment(123, keys=keys)

        with self.assertRaises(AssertionError):
            keys = {"test": ("b", 'b', 'c'), "xxx": ["xyz"]}
            segment = PathSegment("test", keys=keys)

        with self.assertRaises(AssertionError):

            segment = PathSegment("")

    def test_placeholder(self):
        segment = PathSegment("*")
        self.assertTrue(segment.is_placeholder())
        self.assertTrue(segment.is_single_placeholder())
        self.assertFalse(segment.is_multi_placeholder())

        segment = PathSegment("**")
        self.assertTrue(segment.is_placeholder())
        self.assertTrue(segment.is_multi_placeholder())
        self.assertFalse(segment.is_single_placeholder())

        with self.assertRaises(AssertionError):
            segment = PathSegment("*", keys = {"test": "abc"})

        with self.assertRaises(AssertionError):
            segment = PathSegment("**", keys = {"test": "abc"})

    def test_from_string(self):
        string = "PdU"
        segment = PathSegment()
        segment.from_string(string)

        self.assertEqual(segment.name, "pdu")


        string = r"pdU[Adv 123\type_-() = (adv\123)- IND, mode = legacy, Adv 123\type_-()  = SCAN_RSP, test = *]"

        segment = PathSegment()
        segment.from_string(string)

        self.assertEqual(segment.name, "pdu")
        self.assertEqual(segment.keys, {r"adv_123\type_-()": [r"(adv\123)-_ind", "scan_rsp"], "mode": ["legacy"], "test": ["*"]})

        string = "*"
        segment = PathSegment()
        segment.from_string(string)
        self.assertEqual(segment.name, "*")

        string = "**"
        segment = PathSegment()
        segment.from_string(string)
        self.assertEqual(segment.name, "**")

        with self.assertRaises(AssertionError):
            string = "pdu(adv_type = adv_ind)"
            segment = PathSegment()
            segment.from_string(string)

        with self.assertRaises(AssertionError):
            string = "pdu[adv_type = adv_ind"
            segment = PathSegment()
            segment.from_string(string)

    def test_matches_simple(self):
        left = PathSegment()
        left.from_string("pdu")

        right = PathSegment()
        right.from_string("PdU")

        self.assertTrue(right.matches(left))
        self.assertTrue(left.matches(right))


    def test_matches_placeholder(self):
        left = PathSegment()
        right = PathSegment()

        left.from_string("*")
        right.from_string("PdU")

        self.assertTrue(right.matches(left))
        self.assertTrue(left.matches(right))

        left.from_string("**")
        right.from_string("PdU")

        self.assertTrue(right.matches(left))
        self.assertTrue(left.matches(right))

        left.from_string("*")
        right.from_string("PdU[adv_typ = adv_ind]")

        self.assertTrue(right.matches(left))
        self.assertTrue(left.matches(right))

    def test_matches_single_key(self):
        left = PathSegment()
        right = PathSegment()

        left.from_string("pdu")
        right.from_string("pdu[adv_type = adv_ind]")

        self.assertTrue(left.matches(right))
        self.assertFalse(right.matches(left))


    def test_matches_multi_key(self):
        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("pdu[adv_type = adv_ind, mode = *, test=abc]")

        self.assertTrue(selector.matches(field_path))


        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("pdu[adv_type = adv_ind, adv_type = adv_direct_ind, mode = *, test=abc]")

        self.assertTrue(selector.matches(field_path))


        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("pdu[adv_type = adv_ind, mode = extended, test=abc]")

        self.assertFalse(selector.matches(field_path))


        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("pdu[adv_type = scan_request, mode = *, test=abc]")

        self.assertFalse(selector.matches(field_path))


        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("pdu[adv_type = adv_ind, mode = *, ABCD=abc]")

        self.assertFalse(selector.matches(field_path))

        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("pdu")

        self.assertTrue(selector.matches(field_path))

        field_path = PathSegment()
        field_path.from_string("pdu[adv_type = adv_ind, mode = legacy, adv_type=scan_rsp, test=*]")

        selector = PathSegment()
        selector.from_string("adv_struct[adv_type = adv_ind, mode = *, test=abc]")

        self.assertFalse(selector.matches(field_path))










