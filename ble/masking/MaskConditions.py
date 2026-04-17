from ble.walking.RetrievalRuntime import RetrievalRuntime
from ble.utils.HelperMethods import HelperMethods
from ble.components.advertising_data.AdvertisingData import ManufacturerSpecific, ServiceData16Bits
from ble.interfaces.NodeInterface import NodeInterface
from ble.walking.Path import Path
from ble.yaml.Condition import ConditionInterface
from ble.yaml.YamlRegistry import CONDITION_REGISTRY


@CONDITION_REGISTRY.register("is_apple_continuity")
class AppleContinuityCondition(ConditionInterface, HelperMethods):
    def __init__(self, continuity_type: str | int | list[str | int]):
        if isinstance(continuity_type, str) or isinstance(continuity_type, int):
            self.continuity_type = [HelperMethods.clean_hex_value(str(continuity_type))]

        elif isinstance(continuity_type, list):
            self.continuity_type = []
            for item in continuity_type:
                assert isinstance(item, str) or isinstance(item, int)
                self.continuity_type.append(HelperMethods.clean_hex_value(str(item)))

    def apply(self, ctx: NodeInterface) -> bool:
        target_path_string = "**.adv_struct[adv_type=0xff]"
        target_path = Path()
        target_path.from_string(target_path_string)

        runtime = RetrievalRuntime(target_path)

        _ = ctx.walk(runtime)

        adv_structs = runtime.get_values()[target_path]

        if len(adv_structs) != 1:
            return False

        adv_struct = adv_structs[0]

        if not (isinstance(adv_struct, ManufacturerSpecific)):
            return False

        company_id = adv_struct.company_id.get_value()
        data = adv_struct.data.get_value()

        if company_id != "004C":
            return False

        if len(data) < 2:
            return False

        found_type = data[0:2]

        for continuity_type in self.continuity_type:
            if found_type == continuity_type:
                return True

        return False


@CONDITION_REGISTRY.register("is_samsung_find")
class SamsungSmartThingsFindCondition(ConditionInterface, HelperMethods):
    def apply(self, ctx: NodeInterface) -> bool:
        target_path_string = "**.adv_struct[adv_type=0x16]"
        target_path = Path()
        target_path.from_string(target_path_string)

        runtime = RetrievalRuntime(target_path)

        _ = ctx.walk(runtime)

        adv_structs = runtime.get_values()[target_path]

        if len(adv_structs) != 1:
            return False
        adv_struct = adv_structs[0]

        if not isinstance(adv_struct, ServiceData16Bits):
            return False

        uuid = adv_struct.uuid.get_value()

        if uuid == "FD5A":
            return True

        return False


@CONDITION_REGISTRY.register("is_dult")
class DultStandardCondition(ConditionInterface, HelperMethods):
    def __init__(self, network_id: str):
        self.network_id = HelperMethods.clean_hex_value(str(network_id))

    def apply(self, ctx: NodeInterface) -> bool:
        target_path_string = "**.adv_struct[adv_type=0x16]"
        target_path = Path()
        target_path.from_string(target_path_string)

        runtime = RetrievalRuntime(target_path)

        _ = ctx.walk(runtime)

        adv_structs = runtime.get_values()[target_path]

        if len(adv_structs) != 1:
            return False

        adv_struct = adv_structs[0]

        if not isinstance(adv_struct, ServiceData16Bits):
            return False

        uuid = adv_struct.uuid.get_value()

        if uuid != "FCB2":
            return False

        data = adv_struct.data.get_value()

        if len(data) < 2:
            return False

        if data[0:2] != self.network_id:
            return False

        return True



@CONDITION_REGISTRY.register("is_google_find_my_hub")
class GoogleFindMyHubCondition(ConditionInterface, HelperMethods):
    def apply(self, ctx: NodeInterface) -> bool:
        target_path_string = "**.adv_struct[adv_type=0x16]"
        target_path = Path()
        target_path.from_string(target_path_string)

        runtime = RetrievalRuntime(target_path)

        _ = ctx.walk(runtime)

        adv_structs = runtime.get_values()[target_path]

        if len(adv_structs) != 1:
            return False

        adv_struct = adv_structs[0]

        if not isinstance(adv_struct, ServiceData16Bits):
            return False

        uuid = adv_struct.uuid.get_value()
        if uuid != "FEAA":
            return False

        data = adv_struct.data.get_value()

        if len(data) < 2:
            return False

        fhn_type = data[0:2]

        if fhn_type not in ['40', '41']:
            return False
        else:
            return True




@CONDITION_REGISTRY.register("is_tile_find")
class TileFindMyCondition(ConditionInterface, HelperMethods):
    def apply(self, ctx: NodeInterface) -> bool:
        target_path_string = "**.adv_struct[adv_type=0x16]"
        target_path = Path()
        target_path.from_string(target_path_string)

        runtime = RetrievalRuntime(target_path)

        _ = ctx.walk(runtime)

        adv_structs = runtime.get_values()[target_path]

        if len(adv_structs) != 1:
            return False

        adv_struct = adv_structs[0]

        if not isinstance(adv_struct, ServiceData16Bits):
            return False

        uuid = adv_struct.uuid.get_value()

        if uuid == "FEED":
            return True

        return False
