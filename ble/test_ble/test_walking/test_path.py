import unittest
from typing import Iterable

from ble.walking.PathSegment import PathSegment
from ble.walking.Path import Path


class TestPatt(unittest.TestCase):
    def test_initialization(self):
        path = Path()
        self.assertEqual(len(path), 0)
        self.assertEqual(path.segments, [])

        a = PathSegment("a")
        b = PathSegment("b")
        c = PathSegment("c")

        path.append(a)
        path.append(b)
        path.append(c)

    def test_copy(self):
        path = Path()

        a = PathSegment("a")
        b = PathSegment("b")
        c = PathSegment("c")

        path.append(a)
        path.append(b)
        path.append(c)

        copy_path = path.copy()

        self.assertNotEqual(id(path), id(copy_path))
        self.assertEqual(copy_path.segments, [a, b, c])




    def test_append_extend_pop_clear(self):
        path = Path()
        self.assertEqual(len(path), 0)
        self.assertEqual(path.segments, [])

        a = PathSegment("a")
        b = PathSegment("b")
        c = PathSegment("c")

        path.append(a)

        self.assertEqual(len(path), 1)
        self.assertEqual(path.segments, [a])

        second_path = Path()
        second_path.append(b)
        second_path.append(c)

        path.extend(second_path)

        self.assertEqual(len(path), 3)
        self.assertEqual(path.segments, [a, b, c])

        path.pop()
        self.assertEqual(path.segments, [a, b])

        path.clear()
        self.assertEqual(len(path), 0)
        self.assertEqual(path.segments, [])

    def test_iter(self):
        path = Path()

        self.assertIsInstance(path, Iterable)

        a = PathSegment("a")
        b = PathSegment("b")
        c = PathSegment("c")

        path.append(a)
        path.append(b)
        path.append(c)

        for idx, segment in enumerate(path):
            if idx == 0:
                self.assertEqual(segment, a)

            if idx == 1:
                self.assertEqual(segment, b)

            if idx == 2:
                self.assertEqual(segment, c)

        # this tests whether resetting works

        for idx, segment in enumerate(path):
            if idx == 0:
                self.assertEqual(segment, a)

            if idx == 1:
                break

        # this tests whether resetting works

        for idx, segment in enumerate(path):
            if idx == 0:
                self.assertEqual(segment, a)

            if idx == 1:
                self.assertEqual(segment, b)

            if idx == 2:
                self.assertEqual(segment, c)

    def test_get_item(self):
        path = Path()

        self.assertIsInstance(path, Iterable)

        a = PathSegment("a")
        b = PathSegment("b")
        c = PathSegment("c")
        d = PathSegment("d")
        e = PathSegment("e")
        f = PathSegment("f")

        path.append(a)
        path.append(b)
        path.append(c)
        path.append(d)
        path.append(e)
        path.append(f)

        segments = path[1:4:2]
        self.assertEqual(segments, [b, d])

    def test_from_strings_at_init(self):
        path = Path("a.b.c")
        self.assertEqual(path.segments[0].name, "a")
        self.assertEqual(path.segments[1].name, "b")
        self.assertEqual(path.segments[2].name, "c")

    def test_from_string(self):
        string = " a. **. c[test = xyz, test=abc].d[d = 123]"

        path = Path()
        path.from_string(string)

        self.assertEqual(path.segments[0].name, "a")
        self.assertEqual(path.segments[0].keys, dict())
        self.assertEqual(path.segments[1].name, "**")
        self.assertEqual(path.segments[1].keys, dict())

        self.assertEqual(path.segments[2].name, "c")
        self.assertEqual(path.segments[2].keys, {"test": ["xyz", "abc"]})

        self.assertEqual(path.segments[3].name, "d")
        self.assertEqual(path.segments[3].keys, {"d": ["123"]})

    def test_exact_match(self):
        field_path = Path()
        selector_path =  Path()


        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"

        field_path.from_string(string_field)
        selector_path.from_string(string_field)

        self.assertTrue(field_path.matches(selector_path))
        self.assertTrue(selector_path.matches(field_path))

    def test_complex_sub_matches(self):
        field_path = Path()
        selector_path = Path()

        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

        string_field = "**.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].*"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, adv_type = scan_rsp].**.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "**"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "**.pdu[mode=legacy].**"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "packet.pdu.adv_data.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))



        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, mode = test].adv_data.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

        string_field = "**"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "packet.**.data"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, adv_type = scan_rsp].*.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

    def test_negative_matches(self):
        field_path = Path()
        selector_path = Path()

        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "packet.pdu[adv_mode = *, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))

        field_path = Path()
        selector_path = Path()

        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "*.adv_data.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        field_path = Path()
        selector_path = Path()

        string_field = "packet.*.adv_struct[type=0xFF].data"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        field_path = Path()
        selector_path = Path()

        string_field = "packet.pdu[adv_type = ADV_IND, mode = legacy, adv_type = scan_rsp].adv_data.adv_struct[type=0xFF].data"
        string_selector = "packet.pdu[adv_type = *, mode = legacy, adv_type = scan_rsp].adv_data[type=test].adv_struct[type=0xFF].data"


        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))

    def test_star_paths(self):
        field_path = Path()
        selector_path = Path()

        string_field = "a.b.c.d"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "a.b.c.d"
        string_selector = "a.**.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "a.b.**"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "a.b.c.d"
        string_selector = "**.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "**"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "a.b.c.d"
        string_selector = "**"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "**.b.c.d"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "a.b.c.d"
        string_selector = "a.*.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

        string_field = "a.b.*.d"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

        string_field = "a.b.c.*"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))

        string_field = "a.b.c.d"
        string_selector = "*.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "**.c.d"
        string_selector = "*.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "*.*.*.*"
        string_selector = "*.b.**"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


        string_field = "**.c.d*"
        string_selector = "**.b.**"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertTrue(selector_path.matches(field_path))


    def test_negative_star_paths(self):
        field_path = Path()
        selector_path = Path()

        string_field = "a.b.c.d"
        string_selector = "*.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        string_field = "*.c.d"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        string_field = "a.b.c.d"
        string_selector = "a.*.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))

        string_field = "a.*.d"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        string_field = "a.b.*"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        string_field = "a.b.c.d"
        string_selector = "a.*"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


        string_field = "*"
        string_selector = "a.b.c.d"

        field_path.from_string(string_field)
        selector_path.from_string(string_selector)

        self.assertFalse(selector_path.matches(field_path))


































