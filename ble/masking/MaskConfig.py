import copy
from typing import List, override

import yaml

from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.RuntimeInterface import RuntimeInterface
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.walking.Path import Path
from ble.masking.MaskScope import MaskScope



class MaskConfig(RuntimeInterface, PseudomizerSupportInterface):

    def __init__(self, scopes: List[MaskScope] = None, ctx: NodeInterface = None) -> None:
        if scopes is None:
            scopes = []
        self.ctx: NodeInterface = ctx
        self.scopes: List[MaskScope] = scopes
        self.active_scopes: List[MaskScope] = []
        self.current_path: Path = Path()
        self.global_seed: str = ""

        self._sort_scopes()

    def set_ctx(self, ctx: NodeInterface) -> None:
        self.ctx = copy.deepcopy(ctx)

    def reset_ctx(self) -> None:
        self.ctx = None

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        for scope in self.scopes:
            scope.rotate_epoch(rotation_type)

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        for scope in self.scopes:
            scope.configure_pseudomizer(config)

    def from_yaml(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            file = yaml.safe_load(f)

        assert isinstance(file, dict), " loaded yaml is not a dictionary"

        self.from_dict(file)

    def from_dict(self, data: dict):
        assert "scopes" in data.keys(), "scopes not found in config"
        assert "seed" in data.keys(), "seed not found in config"

        seed = data["seed"]
        self.global_seed = seed

        config = PseudomizerConfig()
        config.seed = seed
        config.epoch = 0

        for scope in data["scopes"]:
            mask_scope = MaskScope()
            mask_scope.from_dict(scope)

            self.scopes.append(mask_scope)

        self.configure_pseudomizer(config)

        self._sort_scopes()

    def _sort_scopes(self):
        self.scopes.sort(key=lambda scope: scope.get_priority(), reverse=False)

    @override
    def enter_node(self, node: NodeInterface) -> None:
        self.current_path.append(node.get_path_segment())

        for scope in self.scopes:
            scope.try_activation_for(self.current_path)

    @override
    def leave_node(self) -> None:
        for scope in self.scopes:
            scope.try_deactivation_for(self.current_path)

        self.current_path.pop()

    @override
    def process_fields(self, fields: list[AbstractField]) -> None:
        for scope in self.scopes:
            for field in fields:
                scope.apply_rules_to_field(field, self.current_path, self.ctx)
