import copy
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, override, Dict, cast, List

from ble.fields.BitDataField import BitDataField
from ble.yaml.YamlRegistry import ACTION_REGISTRY
from ble.generation.GenStratRegistry import GenStratRegistry
from ble.utils.HelperMethods import HelperMethods
from ble.fields.HexDataField import HexDataField
from ble.components.MultiNodeContainer import MultiNodeContainer
from ble.parse_policy.StrictMode import StrictMode
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.components.pdu.AdvertisingPDUs import NullPDU
from ble.pseudomization.AbstractRandomAction import AbstractRandomAction
from ble.components.packet.Packet import Packet
from ble.fields.AbstractField import AbstractField
from ble.interfaces.NodeInterface import NodeInterface
from ble.pseudomization.SamplingPseudomizer import WeightedItem, WeightedRange
from ble.components.pdu.AdvertisingPDUs import PDU_REGISTRY
from ble.components.advertising_data.AdvertisingData import ADVERTISING_REGISTRY
from ble.errors.ParseError import ParseError
from ble.generation.gen_data.BleDeviceNames import BLE_Device_Names, LEN_BLE_DEVICE_NAMES

GEN_STRAT_REGISTRY = GenStratRegistry()


@lru_cache(maxsize=2)
def load_bluetooth_uuids() -> Dict[int, str]:
    from ble.generation.gen_data.BluetoothIDs import UUIDs
    return cast(Dict[int, str], UUIDs)


@lru_cache(maxsize=2)
def load_bluetooth_company_ids() -> Dict[int, str]:
    from ble.generation.gen_data.BluetoothIDs import COMPANY_IDs
    return cast(Dict[int, str], COMPANY_IDs)


class AbstractGenStrat(AbstractRandomAction, HelperMethods, ABC):
    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None) -> None:
        AbstractRandomAction.__init__(self, rotation_type)
        self.token: str = token


@GEN_STRAT_REGISTRY.register("**.packet")
@ACTION_REGISTRY.register("default_gen_strat_pdu")
class GenStratPDU(AbstractGenStrat):
    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None,
                 transitions: list[list[int]] = None) -> None:
        AbstractGenStrat.__init__(self, rotation_type, token)

        self.transitions = None

        if isinstance(transitions, List):
            for tuple_item in transitions:
                assert isinstance(tuple_item, list), "List items must be a List"
                assert len(tuple_item) == 2, "List item must have length 2"

                assert isinstance(tuple_item[0], int), "First item must be an int"
                assert isinstance(tuple_item[1], int), "Second item must be an int"

            self.transitions = transitions

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        assert isinstance(ctx, Packet), "ctx is not a Packet"
        token = self.token if self.token is not None else ctx.get_name()
        pdu_lut = PDU_REGISTRY.get_registry()

        if self.transitions is None:
            weighted_items = [WeightedItem(t, 1) for _, t in pdu_lut.items() if not (isinstance(t, NullPDU))]


        else:
            weighted_items = [WeightedItem(pdu_lut[t[1]], t[0]) for t in self.transitions]

        pdu = self.sampler.sample_item(token, weighted_items)

        ctx.pdu = pdu()


class AbstractGenStratAdvStructs(AbstractGenStrat, ABC):
    context: str = None

    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None,
                 transitions: list[list[int | List[int]]] = None) -> None:
        AbstractGenStrat.__init__(self, rotation_type, token)

        self.transitions = None

        if isinstance(transitions, List):
            for tuple_item in transitions:
                assert isinstance(tuple_item, list), "List items must be a List"
                assert len(tuple_item) == 2, "List item must have length 2"

                assert isinstance(tuple_item[0], int), "First item must be an int"
                assert isinstance(tuple_item[1], list), "Second item must be an List"

                for item in tuple_item[1]:
                    assert isinstance(item, int), "List items must be a List"

            self.transitions = transitions

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        assert isinstance(ctx, MultiNodeContainer), "ctx is not a MultiNodeContainer"

        token = self.token if self.token is not None else ctx.get_name()

        if self.transitions is None:
            adv_count_distribution = [WeightedRange(1, 1, 500), WeightedRange(2, 2, 250), WeightedRange(3, 3, 125),
                                      WeightedRange(4, 4, 125)]

            adv_count = self.sampler.sample_int(token, adv_count_distribution)

            adv_structs = [WeightedItem(i(), 1) for i in ADVERTISING_REGISTRY.get_registry().values()]

            counter = 0
            sampled_structs = []


            while counter <= 10:
                counter += 1

                sampled_structs = self.sampler.sample_k_items_with_replacement(str(token) + str(counter),
                                                                               adv_count,
                                                                               adv_structs)

                try:
                     StrictMode.verify_adv_struct_occurrences(sampled_structs, self.context)
                     break

                except ParseError:
                    sampled_structs = []
                    continue

        else:
            adv_struct_distribution = [WeightedItem(i[1], i[0]) for i in self.transitions]

            adv_structs = self.sampler.sample_item(token, adv_struct_distribution)

            sampled_structs = [ADVERTISING_REGISTRY[struct]() for struct in adv_structs]

        ctx.clear()

        for adv_struct in sampled_structs:
            ctx.append(adv_struct)


@GEN_STRAT_REGISTRY.register("**.scan_response_data")
@ACTION_REGISTRY.register("default_gen_strat_scan_rsp_data_structs")
class GenStratScanRspStructs(AbstractGenStratAdvStructs):
    context = "SRD"


@GEN_STRAT_REGISTRY.register("**.advertising_data")
@ACTION_REGISTRY.register("default_gen_strat_adv_data_structs")
class GenStratAdvertisingStructs(AbstractGenStratAdvStructs):
    context = "AD"


class AbstractGenStratUuidList(AbstractGenStrat, ABC):
    bit: int = None

    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None,
                 transitions: list[list[int]] = None) -> None:
        AbstractGenStrat.__init__(self, rotation_type, token)

        self.transitions = None

        if isinstance(transitions, List):
            for tuple_item in transitions:
                assert isinstance(tuple_item, list), "List items must be a List"
                assert len(tuple_item) == 2, "List item must have length 2"

                assert isinstance(tuple_item[0], int), "First item must be an int"
                assert isinstance(tuple_item[1], int), "Second item must be an int"

            self.transitions = transitions

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        assert isinstance(ctx, MultiNodeContainer), "ctx is not a MultiNodeContainer"
        token = self.token if self.token is not None else ctx.get_name()

        if self.transitions is None:

            uuid_count_dist = [WeightedRange(1, 1, 80), WeightedRange(2, 2, 15), WeightedRange(3, 3, 5)]
            uuid_count = self.sampler.sample_int(token, uuid_count_dist)

        else:
            uuid_count_dist = [WeightedItem(i[1], i[0]) for i in self.transitions]
            uuid_count = self.sampler.sample_item(token, uuid_count_dist)

        ctx.clear()
        for i in range(uuid_count):
            ctx.append(HexDataField(f"{self.bit} bit UUID", target_byte_length=self.bit // 8))


@GEN_STRAT_REGISTRY.register("**.16_bit_uuids")
@ACTION_REGISTRY.register("default_gen_strat_16_bit_uuid_count")
class GenStratUuidList16Bit(AbstractGenStratUuidList):
    bit = 16


@GEN_STRAT_REGISTRY.register("**.32_bit_uuids")
@ACTION_REGISTRY.register("default_gen_strat_32_bit_uuid_count")
class GenStratUuidList32Bit(AbstractGenStratUuidList):
    bit = 32


@GEN_STRAT_REGISTRY.register("**.128_bit_uuids")
@ACTION_REGISTRY.register("default_gen_strat_128_bit_uuid_count")
class GenStratUuidList128Bit(AbstractGenStratUuidList):
    bit = 128


# this operates on a field and not on a node interface object --> no registration allowed!
# paths could be duplicate for many fields

@ACTION_REGISTRY.register("default_gen_strat_field")
class DefaultGenStratField(AbstractGenStrat):

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        token = self.token if self.token is not None else field.get_name()

        target_length = field.get_target_bit_length()

        if target_length is None:
            if isinstance(field, BitDataField):
                length_distribution = [WeightedRange(1, 64, 1)]
                target_length = self.sampler.sample_int(token, length_distribution)

            elif isinstance(field, HexDataField):
                # 8 to 240 Bits
                length_distribution = [WeightedRange(1 * 8, 30 * 8, 1)]

                target_length = self.sampler.sample_int(token, length_distribution)

                target_length = (target_length // 8) * 8


        hex_string = self.pseudomizer.pseudomize(token, (target_length + 8) // 4)

        bin_string = "".join(HelperMethods.hex_to_bin(i, pad='nibble') for i in hex_string)

        assert len(bin_string[:target_length]) == target_length

        field.set_value(bin_string[:target_length], bin=True)


@ACTION_REGISTRY.register("generate_device_name_from_list")
class GenerateDeviceName(AbstractGenStrat):
    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        token = self.token if self.token is not None else field.get_name()

        index = self.sampler.sample_int(token, [WeightedRange(0, LEN_BLE_DEVICE_NAMES - 1, 1)])

        field.set_value(BLE_Device_Names[index].encode("utf-8").hex(), bin=False)


class AbstractRealID(AbstractGenStrat, ABC):
    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None,
                 transitions: list[list[int]] = None) -> None:
        AbstractGenStrat.__init__(self, rotation_type, token)

        self.transitions = None

        if isinstance(transitions, List):
            for tuple_item in transitions:
                assert isinstance(tuple_item, list), "List items must be a List"
                assert len(tuple_item) == 2, "List item must have length 2"

                assert isinstance(tuple_item[0], int), "First item must be an int"
                assert isinstance(tuple_item[1], int), "Second item must be an int"

            self.transitions = transitions

    @staticmethod
    @abstractmethod
    def _load_id_lut() -> Dict[int, str]:
        pass

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        token = self.token if self.token is not None else field.get_name()

        if self.transitions is None:
            ids = list(self._load_id_lut().keys())
            length = len(ids)

            sampled_index = self.sampler.sample_int(token, [WeightedRange(0, length - 1, 1)])
            vendor_id = HelperMethods.int_to_hex(ids[sampled_index], pad='byte')
            vendor_id = HelperMethods.clean_hex_value(vendor_id)

        else:
            vendor_id_distribution = [WeightedItem(i[1], i[0]) for i in self.transitions]
            
            vendor_id = self.sampler.sample_item(token, vendor_id_distribution)

            vendor_id = HelperMethods.int_to_hex(vendor_id, pad='byte')

        if len(vendor_id) == 2:
            vendor_id = "00" + vendor_id


        assert len(vendor_id) == 4, "vendor_id is not of length 4"

        field.set_value(vendor_id[:4], bin=False)


@ACTION_REGISTRY.register("real_company_id")
class RealCompanyID(AbstractRealID):
    @staticmethod
    @override
    def _load_id_lut() -> Dict[int, str]:
        return load_bluetooth_company_ids()



@ACTION_REGISTRY.register("real_uuid")
class RealUUID(AbstractRealID):
    @staticmethod
    @override
    def _load_id_lut() -> Dict[int, str]:
        return load_bluetooth_uuids()


@ACTION_REGISTRY.register("dult_protocol")
class DultProtocolStrategy(AbstractGenStrat):
    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None,
                 network_id: str | list[str] = None, nearby: bool = None, payload_byte_length: int = None) -> None:
        AbstractGenStrat.__init__(self, rotation_type, token)

        self.network_id = None
        self.payload_byte_length = payload_byte_length
        self.nearby = nearby

        if isinstance(network_id, str):
            self.network_id = [network_id]

        else:
            self.network_id = network_id

        if not self.network_id is None:
            network_ids = []
            for _, network_id in enumerate(self.network_id):
                network_id = HelperMethods.clean_hex_value(network_id)

                assert len(network_id) == 2, "network_id is not of length 2"

                network_ids.append(network_id)

            self.network_id = network_ids

        if isinstance(payload_byte_length, int):
            self.payload_byte_length = payload_byte_length

    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        token = self.token if self.token is not None else field.get_name()

        hex_string = ""

        if self.network_id is None:
            network_id = self.pseudomizer.pseudomize(token, length=2)
        else:
            samples = [WeightedItem(i, 1) for i in self.network_id]
            network_id = self.sampler.sample_item(token, samples)

        assert len(network_id) == 2, "network_id is not of length 2"

        hex_string += network_id

        if self.nearby is not None:
            nearby = str(int(self.nearby))

        else:
            nearby = str(int(self.sampler.sample_int(token, [WeightedRange(0, 1, 1)])))

        hex_string += "0" + nearby

        if self.payload_byte_length is not None:
            length = self.payload_byte_length
        else:
            length = self.sampler.sample_int(token, [WeightedRange(0, 22, 1)])

        hex_string += self.pseudomizer.pseudomize(token, length * 2 + 2)[2:]

        field.set_value(hex_string, bin=False)


@ACTION_REGISTRY.register("real_channel")
class RealChannel(AbstractGenStrat):
    def __init__(self, rotation_type: str | EpochRotation = EpochRotation.CALL, token: str = None,
                 transitions: List[List[int]] = None):
        AbstractGenStrat.__init__(self, rotation_type, token)
        self.transitions = None

        if isinstance(transitions, List):
            for tuple_item in transitions:
                assert isinstance(tuple_item, list), "List items must be a List"
                assert len(tuple_item) == 2, "List item must have length 2"
                assert isinstance(tuple_item[0], int), "First item must be an int"
                assert isinstance(tuple_item[1], int), "Second item must be an int"
                assert 0 <= tuple_item[1] <= 255, "Channel must be in range 0 - 255"

            self.transitions = transitions

    @override
    def execute(self, field: AbstractField, ctx: NodeInterface) -> Any:
        token = self.token if self.token is not None else field.get_name()

        if self.transitions is None:
            distribution = [WeightedItem(37 + i, 1) for i in range(3)]

        else:
            distribution = [WeightedItem(i[1], i[0]) for i in self.transitions]

        channel = self.sampler.sample_item(token, distribution)

        channel = HelperMethods.int_to_hex(channel, pad='byte')

        assert len(channel) == 2, "channel is not of length 1 Byte"

        field.set_value(channel, bin=False)
