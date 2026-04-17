from typing import Any

from ble import AbstractField, NodeInterface
from ble.interfaces.ParserInterface import ParserInterface
from ble.yaml.Action import ActionInterface
from ble.yaml.YamlRegistry import ACTION_REGISTRY

@ACTION_REGISTRY.register("extract_ble_address")
class extractAddress(ActionInterface):
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        address = field.get_value()
        address = str(address).lower()

        assert len(address) % 2 == 0, "address length must be of Byte Length"

        return ":".join(address[i:i + 2] for i in range(0, len(address), 2))


@ACTION_REGISTRY.register("extract_packet_length")
class extractPacketLength(ActionInterface):
    def __init__(self, include_nrf_header: bool=False):
            self.include_nrf_header = include_nrf_header

    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        assert isinstance(ctx, ParserInterface)

        if self.include_nrf_header:
            return ctx.get_length() + 26
        else:
            return ctx.get_length()


@ACTION_REGISTRY.register("convert_adv_channel_to_integer")
class convertChannelToInteger(ActionInterface):
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        value = field.get_value()
        value = int(value, 16)

        assert value in [37, 38, 39], "Advertising Channel must be 37, 38 or 39"

        return value

class extractCompanyID(ActionInterface):
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        company_id = field.get_value()
        company_id = int(company_id, 16)

