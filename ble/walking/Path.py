from functools import lru_cache
from typing import Self

from ble.walking.PathSegment import PathSegment


class Path:
    def __init__(self, path: str = None) -> None:
        self.segments: list[PathSegment] = []
        self.__iter = 0

        if path is not None:
            self.from_string(path)

    def copy(self) -> Self:
        path = Path()
        path.segments = self.segments.copy()
        return path

    def from_string(self, path: str) -> None:
        path = path.strip()
        result = []

        for raw_segment in path.split("."):
            segment = PathSegment()
            segment.from_string(raw_segment)

            result.append(segment)

        self.clear()
        self.segments = result

    def append(self, segment: PathSegment) -> None:
        assert isinstance(segment, PathSegment), "segment must be a PathSegment"
        self.segments.append(segment)

    def extend(self, path: Self) -> None:
        for segment in path.segments:
            self.append(segment)

    def clear(self) -> None:
        self.segments.clear()

    def pop(self) -> None:
        self.segments.pop()

    def matches(self, other: Self):
        # the self object should be the scope path, the right other object should be a module / component path

        """
        Return a memoized rec(i, j) function such that rec(i, j) is True iff
        left[i:] and right[j:] have at least one common concrete expansion.

        Assumptions:
        - each element has a `.kind` attribute
        - '*' matches exactly one segment
        - '**' matches zero or more segments
        - ordinary segments are checked by `segments_compatible(a, b)`
        """

        def suffix_can_match_empty(path: Self, start: int) -> bool:
            for segment in path[start:]:
                if not segment.is_multi_placeholder():
                    return False

            return True

        left = self
        right = other

        @lru_cache(maxsize=None)
        def rec(i: int, j: int) -> bool:
            # both exhausted
            if i == len(left) and j == len(right):
                return True

            # one exhausted: the other must be able to match empty
            if i == len(left):
                return suffix_can_match_empty(right, j)
            if j == len(right):
                return suffix_can_match_empty(left, i)

            a = left[i]
            b = right[j]

            # both are **
            if a.is_multi_placeholder() and b.is_multi_placeholder():
                return (
                        rec(i + 1, j) or  # left ** matches empty
                        rec(i, j + 1)  # right ** matches empty
                )

            # left is **
            if a.is_multi_placeholder():
                return (
                        rec(i + 1, j) or  # left ** matches empty
                        rec(i, j + 1)  # left ** absorbs one segment from right
                )

            # right is **
            if b.is_multi_placeholder():
                return (
                        rec(i, j + 1) or  # right ** matches empty
                        rec(i + 1, j)  # right ** absorbs one segment from left
                )

            # both are single-segment patterns
            if not a.matches(b):
                return False

            return rec(i + 1, j + 1)

        return rec(0, 0)

    def __getitem__(self, item):
        return self.segments[item]

    def __len__(self):
        return len(self.segments)

    def __reset_iter(self):
        self.__iter = 0

    def __iter__(self):
        self.__reset_iter()
        return self

    def __next__(self):
        try:
            element = self.segments[self.__iter]
            self.__iter += 1
            return element

        except IndexError:
            raise StopIteration
