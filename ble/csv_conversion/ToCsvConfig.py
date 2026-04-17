from typing import List, override

from ble import PathSegment, NodeInterface
from ble.csv_conversion.ToCsvColumn import ToCsvColumn
from ble.fields.AbstractField import AbstractField
from ble.interfaces.RuntimeInterface import RuntimeInterface
from ble.walking.Path import Path


class ToCsvConfig(RuntimeInterface):

    def __init__(self):
        self.seperator: str
        self.file_name: str
        self.columns: List[ToCsvColumn] = []
        self.current_path: Path = Path()
        self.ctx: NodeInterface = None

    @override
    def enter_node(self, node: NodeInterface) -> None:
        self.current_path.append(node.get_path_segment())


    def from_dict(self, data: dict) -> None:
        for column in data["columns"]:
            new_column = ToCsvColumn()
            new_column.from_dict(column)
            self.columns.append(new_column)




    @override
    def process_fields(self, fields: list[AbstractField]) -> None:
        for field in fields:
            for column in self.columns:
                column.apply_field(field, self.current_path, self.ctx)

    @override
    def leave_node(self) -> None:
        self.current_path.pop()

