from typing import List, override

from ble import PathSegment
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.interfaces.RuntimeInterface import RuntimeInterface
from ble.walking.Path import Path


class RetrievalRuntime(RuntimeInterface):
    def __init__(self, target_path: Path | list[Path]) -> None:
        self.current_path: Path = Path()

        if isinstance(target_path, Path):
            self.target_path: List[Path] = [target_path]

        elif isinstance(target_path, list):
            for p in target_path:
                assert isinstance(p, Path), "target_path must be a Path object"

            self.target_path: List[Path] = target_path


        self.values: dict[Path, List[NodeInterface | AbstractField]]= dict()
        self.reset()

    def get_values(self) -> dict[Path, List[NodeInterface | AbstractField]]:
        return self.values.copy()

    def reset(self) -> None:
        for target_path in self.target_path:
            self.values[target_path] = []

    @override
    def enter_node(self, node: NodeInterface) -> None:
        self.current_path.append(node.get_path_segment())

        for path in self.target_path:
            if path.matches(self.current_path):
                self.values[path].append(node)

    @override
    def process_fields(self, fields: list[AbstractField]) -> None:
        for field in fields:
            for target_path in self.target_path:
                field_path = self.current_path.copy()
                field_path.append(PathSegment(field.get_name()))

                if target_path.matches(field_path):
                    self.values[target_path].append(field)

    @override
    def leave_node(self) -> None:
        self.current_path.pop()
