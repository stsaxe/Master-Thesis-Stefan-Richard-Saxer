from abc import ABC, abstractmethod

from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.pseudomizer_config import PseudomizerConfig


class PseudomizerSupportInterface(ABC):
    @abstractmethod
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        pass

    @abstractmethod
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        pass