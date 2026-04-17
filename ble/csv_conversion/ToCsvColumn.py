from typing import List, Any

from ble import PathSegment, ActionInterface, ACTION_REGISTRY, HelperMethods
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.walking.Path import Path
from ble.yaml.YamlRegistry import NullAction


class ToCsvColumn(HelperMethods):
    def __init__(self):
        self.column: str = ''
        self.sources: List[Path] = []
        self.data: List[Any] = []
        self.transform: ActionInterface = NullAction()
        self.default_value: Any = ''


    def _set_column(self, column: str):
        HelperMethods.check_valid_string(column, empty_allowed=False)
        self.column = column

    def _set_transform(self, transform: ActionInterface):
        assert isinstance(transform, ActionInterface)
        self.transform = transform

    def from_dict(self, data: dict):
        assert "column" in data.keys()
        self._set_column(data["column"])

        if "default_value" in data.keys():
            self.default_value = data["default_value"]

        if "transform" in data.keys():
            assert "ref" in data["condition"].keys()
            assert "args" in data["condition"].keys()

            action_string = data["transform"]["args"]
            kwargs = data["transform"]["args"]

            new_transform = ACTION_REGISTRY.create(action_string)(**kwargs)
            self._set_transform(new_transform)

        assert "sources" in data.keys()

        for source in data["sources"]:
            assert "path" in source.keys()
            path_string = source["path"]

            path = Path()
            path.from_string(path_string)

            self.sources.append(path)


    def apply_field(self, field: AbstractField, current_path: Path, ctx: NodeInterface) -> None:
        field_path = current_path.copy()
        field_path.append(PathSegment(name=field.get_name()))

        for path in self.sources:
            if not path.matches(field_path):
                continue

            value = self.transform.execute(field, ctx)
            self.data.append(value)

    def to_column_value(self, seperator: str) -> str:
        if len(self.data) == 0:
            return  str(self.default_value)

        out = ""

        for idx, value in enumerate(self.data):
            out += str(value)
            if idx < len(self.data) - 1:
                out += seperator

        return str(out)








