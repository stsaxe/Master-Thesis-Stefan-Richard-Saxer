from typing import List, override

from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.utils.HelperMethods import HelperMethods
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.pseudomization.PseudomizerSupportInterface import PseudomizerSupportInterface
from ble.walking.Path import Path
from ble.walking.PathSegment import PathSegment
from ble.masking.MaskRule import MaskRule
from ble.pseudomization.epoch_rotation_enum import EpochRotation


class MaskScope(HelperMethods, PseudomizerSupportInterface):
    def __init__(self, name: str = "", path: Path=Path(), rules: List[MaskRule]=None, priority: int = 0) -> None:
        if rules is None:
            rules = []
        self.name: str  = name
        self.priority: int = priority
        self.activation_path: Path = path
        self.rules: List[MaskRule] = rules
        self.active: bool = False

        self._set_name(name)
        self._set_priority(priority)
        self._set_rules(rules)
        self._set_path(path)

        self._sort_rules()

    @override
    def rotate_epoch(self, rotation_type: EpochRotation | str):
        for rule in self.rules:
            rule.rotate_epoch(rotation_type)

    @override
    def configure_pseudomizer(self, config: PseudomizerConfig) -> None:
        for rule in self.rules:
            rule.configure_pseudomizer(config)


    def from_dict(self, data: dict) -> None:
        if "name" in data.keys():
            self._set_name(data["name"])

        if "priority" in data.keys():
            self._set_priority(data["priority"])

        assert "path" in data.keys()
        assert "rules" in data.keys()

        path_string = data["path"]
        path = Path()
        path.from_string(path_string)
        self._set_path(path)

        rules = data["rules"]
        assert isinstance(rules, list), "rules must be a list"

        new_rules = []

        for rule in rules:
            new_rule = MaskRule()
            new_rule.from_dict(rule)
            new_rules.append(new_rule)

        self._set_rules(new_rules)
        self._sort_rules()

    def _sort_rules(self):
        self.rules.sort(key=lambda rule: rule.get_priority(), reverse=False)


    def is_active(self) -> bool:
        return self.active

    def try_activation_for(self, matching_path: Path) -> None:
        if self.get_path().matches(matching_path):
            self.active = True

    def try_deactivation_for(self, matching_path: Path) -> None:
        if self.get_path().matches(matching_path):
            self.deactivate()

    def deactivate(self) -> None:
        self.active = False


    def get_name(self) -> str:
        return self.name

    def get_path(self) -> Path:
        return self.activation_path.copy()

    def get_priority(self) -> int:
        return self.priority

    def _set_name(self, name: str) -> None:
        HelperMethods.check_valid_string(name, empty_allowed=True)
        self.name = name

    def _set_path(self, path: Path) -> None:
        assert isinstance(path, Path), "path must be a Path object"
        self.activation_path = path

    def _set_rules(self, rules: List[MaskRule]) -> None:
        assert isinstance(rules, list), "rules must be a list"
        for rule in rules:
            assert isinstance(rule, MaskRule), "rule must be a Rule"
        self.rules = rules

    def _set_priority(self, priority: int) -> None:
        assert isinstance(priority, int), "priority must be a int"
        self.priority = priority


    def apply_rules_to_field(self, field: AbstractField, current_path: Path, ctx: NodeInterface) -> None:
        if not self.is_active():
            return

        field_path = current_path.copy()
        field_path.append(PathSegment(name=field.get_name()))

        for rule in self.rules:
            match_path = self.get_path()
            match_path.extend(rule.get_path())

            if match_path.matches(field_path):
                rule.execute(field, ctx)
