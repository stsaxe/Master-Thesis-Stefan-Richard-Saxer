from abc import abstractmethod
from copy import deepcopy, copy
from typing import Any, override

from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.walking.Path import Path
from ble.yaml.Action import ActionInterface
from ble.yaml.YamlRegistry import NullAction


class AbstractRule(ActionInterface):
    def __init__(self) -> None:
        ActionInterface.__init__(self)
        self.action: ActionInterface = NullAction()
        self.path: Path = Path()
        self.name: str = ""


    def _set_action(self, action: ActionInterface) -> None:
        assert isinstance(action, ActionInterface), "action does not implement ActionInterface"
        self.action = action

    def _set_path(self, path: Path) -> None:
        assert isinstance(path, Path), "path does not implement Path"
        self.path = path

    def _set_name(self, name: str) -> None:
        assert isinstance(name, str), "name does not implement str"
        self.name = name


    def get_path(self) -> Path:
        return deepcopy(self.path)


    @abstractmethod
    def from_dict(self, data: dict) -> None:
        pass

    @abstractmethod
    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        pass