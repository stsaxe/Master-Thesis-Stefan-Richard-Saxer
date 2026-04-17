from typing import override

from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.pseudomization.AbstractRandomAction import AbstractRandomAction
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.walking.Path import Path
from ble.yaml.AbstractRule import AbstractRule
from ble.yaml.Condition import ConditionInterface
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.yaml.YamlRegistry import CONDITION_REGISTRY, ACTION_REGISTRY, NullCondition


class MaskRule(AbstractRule, PseudomizerSupportInterface):
    def __init__(self) -> None:
        AbstractRule.__init__(self)
        self.priority: int = 0
        self.condition: ConditionInterface = NullCondition()

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        if isinstance(self.action, AbstractRandomAction):
            self.action.rotate_epoch(rotation_type)

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        if isinstance(self.action, AbstractRandomAction):
            self.action.configure_pseudomizer(config)

    def get_priority(self) -> int:
        return self.priority

    @override
    def from_dict(self, data: dict) -> None:
        if "priority" in data.keys():
            self.priority = int(data["priority"])

        if "name" in data.keys():
            self._set_name(data["name"])

        if "condition" in data.keys():
            assert "ref" in data["condition"].keys()

            condition_string = data["condition"]["ref"]

            if "args" in data["condition"].keys():
                kwargs = data["condition"]["args"]
            else:
                kwargs = {}

            new_condition = CONDITION_REGISTRY.create(condition_string, **kwargs)
            self._set_condition(new_condition)

        assert "path" in data.keys()

        path_string = data["path"]
        path = Path()

        path.from_string(path_string)
        self._set_path(path)

        assert "action" in data.keys()
        assert "ref" in data["action"].keys()

        action_string = data["action"]["ref"]


        if "args" in data["action"].keys():
            kwargs = data["action"]["args"]
        else:
            kwargs = {}

        new_action = ACTION_REGISTRY.create(action_string, **kwargs)

        assert isinstance(new_action, AbstractRandomAction), "action must be of type AbstractMaskAction"
        self._set_action(new_action)

    def _set_condition(self, condition: ConditionInterface) -> None:
        assert isinstance(condition, ConditionInterface), "condition must be of type ConditionInterface"
        self.condition = condition

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> None:
        try:
            condition = self.condition.apply(ctx)
        except:
            return

        if condition:
            _ = self.action.execute(field, ctx)
            self.rotate_epoch(EpochRotation.CALL)
