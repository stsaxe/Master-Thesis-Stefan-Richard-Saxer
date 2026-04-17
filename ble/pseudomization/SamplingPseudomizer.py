from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Generic, Sequence, TypeVar, override, List

from ble.pseudomization.AbstractRandomizer import AbstractRandomizer
from ble.utils.HelperMethods import HelperMethods

T = TypeVar("T")


@dataclass(frozen=True)
class WeightedRange:
    start: int
    end: int
    weight: int

    def __post_init__(self) -> None:
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("start and end must be ints")
        if not isinstance(self.weight, int):
            raise TypeError("weight must be an int")
        if self.start > self.end:
            raise ValueError("start must be <= end")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")


@dataclass(frozen=True)
class WeightedItem(Generic[T]):
    value: T
    weight: int

    def __post_init__(self) -> None:
        if not isinstance(self.weight, int):
            raise TypeError("weight must be an int")
        if self.weight <= 0:
            raise ValueError("weight must be > 0")


class PseudoRandomSampler(HelperMethods, AbstractRandomizer):
    def _u64(self, token: str) -> int:
        subkey = self._subkey(token)
        block = hmac.new(
            subkey,
            (0).to_bytes(4, "big"),
            hashlib.sha256,
        ).digest()
        return int.from_bytes(block[:8], "big")

    def _choose_index_by_weight(
        self,
        token: str,
        weights: Sequence[int],
    ) -> int:
        if not weights:
            raise ValueError("weights must not be empty")
        if any((not isinstance(w, int) or w <= 0) for w in weights):
            raise ValueError("all weights must be positive ints")

        total = sum(weights)
        draw = self._u64(token) % total

        acc = 0
        for i, w in enumerate(weights):
            acc += w
            if draw < acc:
                return i

        raise RuntimeError("weighted selection failed")

    def sample_int(
        self,
        token: str,
        ranges: Sequence[WeightedRange],
    ) -> int:
        if not ranges:
            raise ValueError("ranges must not be empty")

        bucket_index = self._choose_index_by_weight(
            token=f"{token}|bucket",
            weights=[r.weight for r in ranges],
        )
        chosen = ranges[bucket_index]

        span = chosen.end - chosen.start + 1
        offset = self._u64(f"{token}|value") % span
        return chosen.start + offset

    def sample_item(
        self,
        token: str,
        items: List[WeightedItem[T]],
    ) -> T:
        if not items:
            raise ValueError("items must not be empty")

        index = self._choose_index_by_weight(
            token=token,
            weights=[item.weight for item in items],
        )
        return items[index].value

    def sample_k_items_with_replacement(
        self,
        token_prefix: str,
        k: int,
        items: List[WeightedItem[T]],
    ) -> list[T]:
        if not isinstance(token_prefix, str) or not token_prefix:
            raise ValueError("token_prefix must be a non-empty string")
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative int")

        return [
            self.sample_item(
                token=f"{token_prefix}|{i}",
                items=items,
            )
            for i in range(k)
        ]

    def sample_k_items_without_replacement(
        self,
        token_prefix: str,
        k: int,
        items: Sequence[WeightedItem[T]],
    ) -> list[T]:
        if not isinstance(token_prefix, str) or not token_prefix:
            raise ValueError("token_prefix must be a non-empty string")
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non-negative int")
        if k > len(items):
            raise ValueError("k cannot exceed number of available items")

        pool = list(items)
        out: list[T] = []

        for i in range(k):
            idx = self._choose_index_by_weight(
                token=f"{token_prefix}|{i}",
                weights=[item.weight for item in pool],
            )
            out.append(pool[idx].value)
            del pool[idx]

        return out