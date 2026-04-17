import unittest

from ble.utils.StringCursor import StringCursor


class TestStringCursor(unittest.TestCase):
    def test_init(self):
        data ="AB1234"
        cursor = StringCursor(data)

        self.assertEqual(cursor.remaining(bytes=True), 3)
        self.assertEqual(cursor.remaining(), 6)

        data = "AB123"

        cursor = StringCursor(data)
        self.assertEqual(cursor.remaining(bytes=True), 2)
        self.assertEqual(cursor.remaining(), 5)

        self.assertEqual(cursor.data, data)

    def test_read(self):
        cursor = StringCursor("12345")
        self.assertEqual(cursor.remaining(), 5)
        self.assertEqual(cursor.read(1), "1")
        self.assertEqual(cursor.read(2), "23")
        self.assertEqual(cursor.remaining(), 2)
        self.assertEqual(cursor.read(2), "45")
        self.assertEqual(cursor.remaining(), 0)

    def test_read_assertion(self):
        cursor = StringCursor("12345")
        self.assertEqual(cursor.remaining(), 5)
        self.assertEqual(cursor.read(1), "1")
        self.assertEqual(cursor.remaining(), 4)
        with self.assertRaises(AssertionError):
            cursor.read(5)


    def test_read_bytes(self):
        cursor = StringCursor("12345678")
        self.assertEqual(cursor.read_bytes(1), "12")
        self.assertEqual(cursor.read_bytes(2), "3456")
        self.assertEqual(cursor.read_bytes(1), "78")
        self.assertEqual(cursor.remaining(), 0)

    def test_read_bytes_assertion(self):
        cursor = StringCursor("12345678")
        with self.assertRaises(AssertionError):
            cursor.read_bytes(5)

    def test_read_to_end(self):
        cursor = StringCursor("12345")
        cursor.read(1)

        self.assertEqual(cursor.read_to_end(), "2345")

        cursor = StringCursor("12345")
        cursor.read(2)

        self.assertEqual(cursor.read_to_end(byte_aware=True), "34")
        self.assertEqual(cursor.remaining(), 1)
        self.assertEqual(cursor.read(1), "5")

    def test_peek(self):
        cursor = StringCursor("1234567")
        cursor.read(1)
        self.assertEqual(cursor.peek(n=2, shift=2), "45")

        cursor = StringCursor("1234567")
        cursor.read(1)
        self.assertEqual(cursor.peek(n=4, shift=2), "4567")

        with self.assertRaises(AssertionError):
            cursor = StringCursor("1234567")
            cursor.read(1)
            cursor.peek(n=5, shift=2)

        with self.assertRaises(AssertionError):
            cursor = StringCursor("1234567")
            cursor.read(1)
            cursor.peek(n=4, shift=5)


    def test_peek_bytes(self):
        cursor = StringCursor("1234567")
        cursor.read(1)
        self.assertEqual(cursor.peek_bytes(n=2, shift=1), "4567")

        cursor = StringCursor("1234567")
        cursor.read(1)
        self.assertEqual(cursor.peek_bytes(n=1, shift=1), "45")

        with self.assertRaises(AssertionError):
            cursor = StringCursor("1234567")
            cursor.read(1)
            cursor.peek_bytes(n=4, shift=1)

        with self.assertRaises(AssertionError):
            cursor = StringCursor("1234567")
            cursor.read(1)
            cursor.peek_bytes(n=2, shift=5)
















