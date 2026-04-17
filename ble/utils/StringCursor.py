from ble.utils.HelperMethods import HelperMethods


class StringCursor(HelperMethods):
    def __init__(self, data: str):
        HelperMethods.check_valid_string(data, empty_allowed=True)
        self.data: str = data
        self.idx: int = 0

    def remaining(self, bytes: bool = False) -> int:
        if bytes:
            return (len(self.data) - self.idx) // 2
        else:
            return len(self.data) - self.idx

    def read(self, n: int) -> str:
        assert self.idx + n <= len(self.data), f"Unexpected end of data while reading {n} Bytes"
        r = self.data[self.idx:self.idx + n]
        self.idx += n
        return r

    def peek(self, n: int, shift: int = 0) -> str:
        assert self.idx + n + shift <= len(self.data), "peek is out of range for data"
        return self.data[self.idx + shift:self.idx + n + shift]

    def peek_bytes(self, n: int, shift: int = 0) -> str:
        return self.peek(2*n, 2*shift)

    def read_bytes(self, n: int = 1) -> str:
        return self.read(n * 2)

    def read_to_end(self, byte_aware: bool = False) -> str:
        r = self.data[self.idx:]
        self.idx = len(self.data)

        if byte_aware and len(r) % 2 != 0:
            r = r[:-1]
            self.idx -= 1
        return r
