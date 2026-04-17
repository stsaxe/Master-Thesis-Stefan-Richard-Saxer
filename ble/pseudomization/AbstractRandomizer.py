from __future__ import annotations

from abc import ABC
import hashlib
import hmac
from typing import override

from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class AbstractRandomizer(PseudomizerSupportInterface, ABC):
    def __init__(self, rotation_type: str | EpochRotation) -> None:
        self.seed: str = " "
        self.epoch: int = 0

        if isinstance(rotation_type, EpochRotation):
            self.rotation_type = rotation_type
        elif isinstance(rotation_type, str):
            self.rotation_type = EpochRotation(rotation_type)
        else:
            raise TypeError("rotation_type must be an EpochRotation or string")

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        seed = config.seed
        epoch = config.epoch

        if not isinstance(seed, str) or not seed:
            raise ValueError("seed must be a non-empty string")
        if not isinstance(epoch, int):
            raise TypeError("epoch must be an int")

        self.seed = seed
        self.epoch = epoch

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        if isinstance(rotation_type, str):
            rotation_type = EpochRotation(rotation_type)

        if self.rotation_type == rotation_type and self.rotation_type != EpochRotation.NEVER:
            self.epoch += 1

    def _subkey(self, token: str) -> bytes:
        if not isinstance(token, str) or not token:
            raise ValueError("token must be a non-empty string")

        key = self.seed.encode("utf-8")
        msg = f"{self.epoch}|{token}".encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).digest()
