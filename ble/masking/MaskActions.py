from typing import Any, override
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.pseudomization.AbstractRandomAction import AbstractRandomAction
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.yaml.YamlRegistry import ACTION_REGISTRY
from ble.fields.HexDataField import HexDataField


@ACTION_REGISTRY.register("mask_ble_address")
class MaskBleAddress(AbstractRandomAction):
    def __init__(self, rotation_type: str = EpochRotation.STREAM, token: str = None, length: int = 12) -> None:
        AbstractRandomAction.__init__(self, rotation_type)
        self.token: str = token
        self.length: int = length

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        assert len(field.get_value(prefix=False)) == self.length, f"Address must be {self.length} Bits long"
        assert isinstance(field, HexDataField), "field must be a HexDataField"

        if self.token is None:
            value = self.pseudomizer.pseudomize(field.get_name(), self.length)
        else:
            value = self.pseudomizer.pseudomize(self.token, self.length)

        field.set_value(value)


@ACTION_REGISTRY.register("mask_hex_data")
class MaskHexData(AbstractRandomAction):
    def __init__(self, rotation_type: str = EpochRotation.STREAM, start: int = 0, end: int = None, step: int = 1,
                 token: str = None) -> None:
        AbstractRandomAction.__init__(self, rotation_type)
        self.start: int = start
        self.end: int = end
        self.step: int = step
        self.token: str = token

    @staticmethod
    def _replace_in_slice(string: str, replace_value: str, start: int, end: int, step: int) -> str:
        slice_range = range(start, end, step)
        assert len(slice_range) == len(replace_value)

        new_string = list(string)

        for i, j in enumerate(slice_range):
            new_string[j] = replace_value[i]

        return "".join(i for i in new_string)

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        assert isinstance(field, HexDataField), "field must be a HexDataField"
        value = field.get_value(prefix=False)

        if self.end is None:
            end = len(value)
        else:
            end = (self.end + len(value)) % len(value)

        length = len(value[self.start:end:self.step])

        if self.token is None:
            mask_values = self.pseudomizer.pseudomize(field.get_name(), length)
        else:
            mask_values = self.pseudomizer.pseudomize(self.token, length)

        new_value = self._replace_in_slice(value, mask_values, self.start, end, self.step)
        field.set_value(new_value)
