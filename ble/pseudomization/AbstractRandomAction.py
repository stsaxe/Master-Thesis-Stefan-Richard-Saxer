from abc import ABC
from typing import override

from ble.pseudomization.SamplingPseudomizer import PseudoRandomSampler
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.HexPseudomizer import HexPseudomizer
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.yaml.Action import ActionInterface


class AbstractRandomAction(ActionInterface, PseudomizerSupportInterface, ABC):
    def __init__(self, rotation_type: str | EpochRotation) -> None:
        self.pseudomizer: HexPseudomizer = HexPseudomizer(rotation_type)
        self.sampler: PseudoRandomSampler = PseudoRandomSampler(rotation_type)

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        self.pseudomizer.configure_pseudomizer(config)
        self.sampler.configure_pseudomizer(config)

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        self.pseudomizer.rotate_epoch(rotation_type)
        self.sampler.rotate_epoch(rotation_type)
