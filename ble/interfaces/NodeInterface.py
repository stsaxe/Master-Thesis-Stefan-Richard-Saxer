from abc import ABC
from typing import Self

from ble.utils.HelperMethods import HelperMethods
from ble.fields.AbstractField import AbstractField
from ble.interfaces.RuntimeInterface import RuntimeInterface
from ble.walking.PathSegment import PathSegment
from collections.abc import Iterable


class NodeInterface(HelperMethods, ABC):
    def __init__(self, name: str) -> None:
        self.node_name: str = ""
        self.set_name(name)

    def get_path_segment(self) -> PathSegment:
        return PathSegment(self.get_name())

    def get_name(self) -> str:
        return self.node_name

    def set_name(self, name: str) -> None:
        HelperMethods.check_valid_string(name, empty_allowed=False)
        self.node_name = name

    def get_children(self) -> list[Self]:
        children = []
        for _, var in vars(self).items():
            # in case the child is a node, we do not have to check whether it is an iterable like a list
            # this is important for the MultiNode container, that is otherwise skipped as a child

            if isinstance(var, NodeInterface):
                children.append(var)
                continue

            elif isinstance(var, Iterable):
                for element in var:
                    if isinstance(element, NodeInterface):
                        children.append(element)

        return children

    def get_fields(self) -> list[AbstractField]:
        fields = []

        for _, var in vars(self).items():
            if isinstance(var, AbstractField):
                fields.append(var)

            elif isinstance(var, Iterable):
                for element in var:
                    if isinstance(element, AbstractField):
                        fields.append(element)

        return fields

    def walk(self, runtime: RuntimeInterface) -> RuntimeInterface:
        runtime.enter_node(self)

        runtime.process_fields(self.get_fields())

        for child in self.get_children():
            runtime = child.walk(runtime)

        runtime.leave_node()

        return runtime
