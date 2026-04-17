from typing import Any, override

from ble.yaml.YamlRegistry import ACTION_REGISTRY
from ble.walking.Path import Path
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.yaml.AbstractRule import AbstractRule
from ble.pseudomization.AbstractRandomAction import AbstractRandomAction


class GenerationRule(AbstractRule, PseudomizerSupportInterface):
    def __init__(self) -> None:
        AbstractRule.__init__(self)

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        if isinstance(self.action, AbstractRandomAction):
            self.action.rotate_epoch(rotation_type)

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        if isinstance(self.action, AbstractRandomAction):
            self.action.configure_pseudomizer(config)

    @override
    def from_dict(self, data: dict) -> None:
        if "name" in data.keys():
            self._set_name(data["name"])

        assert "path" in data.keys(), "path is not provided"

        path_string = data["path"]
        path = Path()
        path.from_string(path_string)

        self._set_path(path)

        assert "action" in data.keys(), "action is not provided"
        assert "ref" in data['action'].keys(), "ref is not provided"
        ref = data["action"]["ref"]

        if "args" in data["action"].keys():
            kwargs = data["action"]["args"]

        else:
            kwargs = {}

        new_action = ACTION_REGISTRY.create(ref, **kwargs)


        self._set_action(new_action)

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        _ = self.action.execute(field, ctx)