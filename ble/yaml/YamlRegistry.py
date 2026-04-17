from typing import Optional, Callable, Any

from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.yaml.Action import ActionInterface
from ble.yaml.Condition import ConditionInterface


class YamlRegistry:
    def __init__(self):
        self._entries: dict[str, Any] = dict()

    def register(self, name: str):
        def deco(cls):
            if name in self._entries:
                raise KeyError(f"Duplicate registry entry: {name}")
            self._entries[name] = cls
            return cls
        return deco

    def create(self, ref: str, **kwargs):
        if ref not in self._entries:
            raise KeyError(f"Unknown ref: {ref}")
        cls = self._entries[ref]
        return cls(**kwargs)

    def __getitem__(self, item) -> type:
        if isinstance(item, str):
            return self._entries[item]
        else:
            raise KeyError(f"Invalid Key {item}, must be str")


class ActionRegistry(YamlRegistry):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(YamlRegistry, cls).__new__(cls)
        return cls.__instance


class ConditionRegistry(YamlRegistry):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(YamlRegistry, cls).__new__(cls)
        return cls.__instance




CONDITION_REGISTRY = ConditionRegistry()
ACTION_REGISTRY = ActionRegistry()


@ACTION_REGISTRY.register("NullAction")
class NullAction(ActionInterface):
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        return field.get_value()

@CONDITION_REGISTRY.register("NullCondition")
class NullCondition(ConditionInterface):
    def apply(self, ctx: NodeInterface) -> bool:
        return True

