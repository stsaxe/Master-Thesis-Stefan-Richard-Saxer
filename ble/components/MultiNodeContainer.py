from collections.abc import Iterable
from typing import Any, override, Self

from ble.utils.HelperMethods import HelperMethods
from ble.interfaces.NodeInterface import NodeInterface


class MultiNodeContainer(NodeInterface, Iterable):

    def __init__(self, name: str, components: list[Any] = None, component_type: type = Any) -> None:
        NodeInterface.__init__(self, name)
        self.components: list = []
        self.__iter: int = 0
        self.__reset_iter()
        assert component_type is not None, "component_type cannot be None"
        assert component_type is Any or isinstance(component_type, type), "component_type must be a type"
        self.component_type: type = component_type

        if components is not None:
            self.set_components(components)

    def set_components(self, components: list[Any]) -> None:
        assert isinstance(components, list), "components must be a list"
        for component in components:
            self.append(component)

    def clear(self) -> None:
        self.components = []

    def append(self, component: Any) -> None:
        if self.component_type is not Any:
            assert isinstance(component, self.component_type), f"component must be an instance of {self.component_type}"
        self.components.append(component)

    def __reset_iter(self) -> None:
        self.__iter = 0

    @override
    def __iter__(self) -> Self:
        self.__reset_iter()
        return self

    def __next__(self) -> Any:
        try:
            self.__iter += 1
            return self.components[self.__iter - 1]

        except IndexError:
            raise StopIteration

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, item) -> Any:
        return self.components[item]
