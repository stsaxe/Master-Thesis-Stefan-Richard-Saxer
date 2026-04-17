import unittest

from ble import HexDataField, ACTION_REGISTRY, DefaultGenStratField, BitDataField, BLE_Device_Names
from ble.components.MultiNodeContainer import MultiNodeContainer
from ble.pseudomization.SamplingPseudomizer import WeightedItem, WeightedRange
from ble.components.advertising_data.AdvertisingData import ManufacturerSpecific, Flags
from ble.components.pdu.AdvertisingPDUs import AdvInd, NullPDU, AdvDirectInd, ScanReq
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.components.packet.Packet import Packet
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.walking.Path import Path
from ble.generation.GenerationStrategies import GenStratPDU, GEN_STRAT_REGISTRY, GenStratAdvertisingStructs, \
    GenStratScanRspStructs, GenStratUuidList16Bit, GenStratUuidList32Bit, GenStratUuidList128Bit, GenerateDeviceName, \
    RealUUID, RealCompanyID, load_bluetooth_company_ids, load_bluetooth_uuids, DultProtocolStrategy, RealChannel

from unittest.mock import patch, Mock

from playground.temp2 import AdvIndPDU


class TestGenStrats(unittest.TestCase):
    def test_abstract_gen_strat(self):
        strat = GenStratPDU(token="TOKEN", rotation_type='stream')

        self.assertEqual(strat.token, "TOKEN")
        self.assertEqual(strat.sampler.rotation_type, EpochRotation.STREAM)
        self.assertEqual(strat.pseudomizer.rotation_type, EpochRotation.STREAM)

        config = PseudomizerConfig()
        config.seed = "TEST"
        config.epoch = 1

        strat.configure_pseudomizer(config)

        self.assertEqual(strat.sampler.seed, "TEST")
        self.assertEqual(strat.pseudomizer.seed, "TEST")

        self.assertEqual(strat.sampler.epoch, 1)
        self.assertEqual(strat.pseudomizer.epoch, 1)

        strat.rotate_epoch("call")
        self.assertEqual(strat.sampler.epoch, 1)
        self.assertEqual(strat.pseudomizer.epoch, 1)

        strat.rotate_epoch("stream")
        self.assertEqual(strat.sampler.epoch, 2)
        self.assertEqual(strat.pseudomizer.epoch, 2)

    def test_gen_pdu_creation(self):
        path_string = '**.packet'
        path = Path()
        path.from_string(path_string)

        strat = GEN_STRAT_REGISTRY.create(path)

        self.assertIsInstance(strat, GenStratPDU)

    def test_gen_pdu_default(self):
        strat = GenStratPDU(rotation_type='stream')
        packet = Packet()

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = AdvInd

        _ = strat.execute(None, packet)

        self.assertIsInstance(packet.pdu, AdvInd)

        args, kwargs = strat.sampler.sample_item.call_args

        pdu_types = [t.value for t in args[1]]
        self.assertTrue(NullPDU not in pdu_types)

        weights = set([t.weight for t in args[1]])
        self.assertEqual({1}, weights)

        self.assertEqual(args[0], 'Packet')

    def test_gen_pdu_transitions(self):
        strat = GenStratPDU(rotation_type='stream', token='TEST', transitions=[[100, 0b0000], [50, 1], [20, 3]])

        self.assertEqual(strat.transitions, [[100, 0], [50, 1], [20, 3]])

        packet = Packet()

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = AdvInd

        _ = strat.execute(None, packet)

        self.assertIsInstance(packet.pdu, AdvInd)

        args, kwargs = strat.sampler.sample_item.call_args

        pdu_types = [t.value for t in args[1]]
        self.assertEqual(pdu_types, [AdvInd, AdvDirectInd, ScanReq])

        weights = list([t.weight for t in args[1]])
        self.assertEqual([100, 50, 20], weights)

        self.assertEqual(args[0], 'TEST')

    def test_gen_adv_structs_creation(self):
        path_string = '**.advertising_data'
        path = Path()
        path.from_string(path_string)

        strat = GEN_STRAT_REGISTRY.create(path)

        self.assertIsInstance(strat, GenStratAdvertisingStructs)

    def test_gen_scan_rsp_structs_creation(self):
        path_string = '**.scan_response_data'
        path = Path()
        path.from_string(path_string)

        strat = GEN_STRAT_REGISTRY.create(path)

        self.assertIsInstance(strat, GenStratScanRspStructs)

    def test_gen_adv_structs_default(self):
        strat = GenStratAdvertisingStructs()

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 2
        strat.sampler.sample_k_items_with_replacement.return_value = [ManufacturerSpecific(), Flags()]

        ctx = MultiNodeContainer("TestContainer")

        _ = strat.execute(None, ctx)

        self.assertIsInstance(ctx[0], ManufacturerSpecific)
        self.assertIsInstance(ctx[1], Flags)

        args, kwargs = strat.sampler.sample_int.call_args
        self.assertEqual(args[1], [WeightedRange(1, 1, 500), WeightedRange(2, 2, 250), WeightedRange(3, 3, 125),
                                   WeightedRange(4, 4, 125)])

        self.assertEqual(args[0], 'TestContainer')

    def test_gen_adv_structs_transitions(self):
        strat = GenStratAdvertisingStructs(token='TEST', transitions=[[100, [0xFF, 0x01]], [50, [0x0a, 0x01]]])

        self.assertEqual(strat.transitions, [[100, [255, 0x01]], [50, [0x0a, 0x01]]])

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = [255, 1]

        ctx = MultiNodeContainer("TestContainer")

        _ = strat.execute(None, ctx)

        args, kwargs = strat.sampler.sample_item.call_args

        self.assertIsInstance(ctx[0], ManufacturerSpecific)
        self.assertIsInstance(ctx[1], Flags)

        self.assertEqual(args[0], 'TEST')
        self.assertEqual(args[1], [WeightedItem(value=[0xFF, 1], weight=100), WeightedItem(value=[0x0a, 1], weight=50)])

    def test_gen_adv_structs_invalid_selection(self):
        strat = GenStratAdvertisingStructs()

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 2
        strat.sampler.sample_k_items_with_replacement.return_value = [Flags(), Flags()]

        ctx = MultiNodeContainer("TestContainer")

        _ = strat.execute(None, ctx)

        self.assertEqual(len(ctx), 0)

    def test_gen_adv_structs_invalid_selection_srd(self):
        strat = GenStratScanRspStructs()

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 2
        strat.sampler.sample_k_items_with_replacement.return_value = [ManufacturerSpecific(), Flags()]

        ctx = MultiNodeContainer("TestContainer")

        _ = strat.execute(None, ctx)

        self.assertEqual(len(ctx), 0)

    def test_gen_16_bit_uuid_list_creation(self):
        path_string = '**.16_bit_uuids'
        path = Path()
        path.from_string(path_string)

        strat = GEN_STRAT_REGISTRY.create(path)

        self.assertIsInstance(strat, GenStratUuidList16Bit)
        self.assertEqual(strat.bit, 16)

    def test_gen_32_bit_uuid_list_creation(self):
        path_string = '**.32_bit_uuids'
        path = Path()
        path.from_string(path_string)

        strat = GEN_STRAT_REGISTRY.create(path)

        self.assertIsInstance(strat, GenStratUuidList32Bit)
        self.assertEqual(strat.bit, 32)

    def test_gen_128_bit_uuid_list_creation(self):
        path_string = '**.128_bit_uuids'
        path = Path()
        path.from_string(path_string)

        strat = GEN_STRAT_REGISTRY.create(path)

        self.assertIsInstance(strat, GenStratUuidList128Bit)
        self.assertEqual(strat.bit, 128)

    def test_gen_strat_uuid_list_default(self):
        strat = GenStratUuidList16Bit()

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 2

        ctx = MultiNodeContainer("TestContainer")

        _ = strat.execute(None, ctx)

        self.assertEqual(len(ctx), 2)
        self.assertEqual(ctx[0].get_name(), "16 bit UUID")
        self.assertIsInstance(ctx[0], HexDataField)
        self.assertEqual(ctx[0].get_target_bit_length(), 16)

        args, kwargs = strat.sampler.sample_int.call_args

        self.assertEqual(args[0], 'TestContainer')
        self.assertEqual(args[1], [WeightedRange(1, 1, 80), WeightedRange(2, 2, 15), WeightedRange(3, 3, 5)])

    def test_gen_strat_uuid_list_transitions(self):
        strat = GenStratUuidList16Bit(token='TEST', transitions=[[100, 1], [50, 3], [20, 2]])
        self.assertEqual(strat.transitions, [[100, 1], [50, 3], [20, 2]])

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = 3

        ctx = MultiNodeContainer("TestContainer")
        ctx.append("Test")

        _ = strat.execute(None, ctx)

        self.assertEqual(len(ctx), 3)

        args, kwargs = strat.sampler.sample_item.call_args

        self.assertEqual(args[0], 'TEST')
        self.assertEqual(args[1], [WeightedItem(1, 100), WeightedItem(3, 50), WeightedItem(2, 20)])

    def test_default_field_gen_strat(self):
        strat = ACTION_REGISTRY.create("default_gen_strat_field")
        self.assertIsInstance(strat, DefaultGenStratField)

    def test_field_gen_strat_hex_field_no_target_length(self):
        field = HexDataField("TOKEN", target_bit_length=None)

        strat = DefaultGenStratField()

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 8 * 3 + 1

        strat.pseudomizer.seed = "TEST"
        strat.pseudomizer.epoch = 0

        _ = strat.execute(field, None)

        self.assertEqual(field.get_value(), "CD31EE")

    def test_field_gen_strat_hex_field_with_target_length(self):
        field = HexDataField("TOKEN", target_byte_length=3)

        strat = DefaultGenStratField()

        strat.pseudomizer.seed = "TEST"
        strat.pseudomizer.epoch = 0

        _ = strat.execute(field, None)

        self.assertEqual(field.get_value(), "CD31EE")

    def test_field_gen_strat_bit_field_no_target_length(self):
        field = BitDataField("TOKEN", target_bit_length=None)

        strat = DefaultGenStratField()

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 23

        strat.pseudomizer.seed = "TEST"
        strat.pseudomizer.epoch = 0

        _ = strat.execute(field, None)

        self.assertEqual(field.get_value(bin=True), "11001101001100011110111")

    def test_field_gen_strat_bit_field_with_target_length(self):
        field = BitDataField("TOKEN", target_bit_length=5)

        strat = DefaultGenStratField()

        strat.pseudomizer.seed = "TEST"
        strat.pseudomizer.epoch = 0

        _ = strat.execute(field, None)

        self.assertEqual(field.get_value(bin=True), "11001")

    def test_device_name_strat_Creation(self):
        strat = ACTION_REGISTRY.create("generate_device_name_from_list")
        self.assertIsInstance(strat, GenerateDeviceName)

    def test_device_name_strat(self):
        strat = GenerateDeviceName()

        field = HexDataField("DeviceName", target_bit_length=None)
        strat.execute(field, None)
        device_name = bytes.fromhex(field.get_value()).decode('utf-8')

        self.assertTrue(device_name in BLE_Device_Names)

    def test_real_uuid_strat_Creation(self):
        strat = ACTION_REGISTRY.create("real_uuid")
        self.assertIsInstance(strat, RealUUID)

    def test_real_company_id_strat_Creation(self):
        strat = ACTION_REGISTRY.create("real_company_id")
        self.assertIsInstance(strat, RealCompanyID)

    def test_real_comapny_id_default(self):
        strat = RealCompanyID()
        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = -2

        field = HexDataField("CompanyID", target_bit_length=None)

        _ = strat.execute(field, None)
        self.assertEqual(field.get_value(), "0001")

        args, kwargs = strat.sampler.sample_int.call_args
        self.assertEqual(args[0], 'CompanyID')

    def test_real_uuid_default(self):
        strat = RealUUID()
        print(load_bluetooth_uuids())

        strat.sampler = Mock()
        strat.sampler.sample_int.return_value = 0

        field = HexDataField("UUID", target_bit_length=None)

        _ = strat.execute(field, None)
        self.assertEqual(field.get_value(), "FEFF")

    def test_real_company_id_transitions(self):
        strat = RealUUID(token='TOKEN', transitions=[[100, 0x0001], [50, 0xAB12], [20, 1234]])

        self.assertEqual(strat.transitions, [[100, 0x0001], [50, 0xAB12], [20, 1234]])

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = 0xab12

        field = HexDataField("CompanyID", target_bit_length=None)

        _ = strat.execute(field, None)

        self.assertEqual(field.get_value(), "AB12")

        args, kwargs = strat.sampler.sample_item.call_args
        self.assertEqual(args[0], 'TOKEN')
        self.assertEqual(args[1], [WeightedItem(1, 100), WeightedItem(0xAB12, 50), WeightedItem(1234, 20)])


    def test_create_dult(self):
        strat = ACTION_REGISTRY.create("dult_protocol")
        self.assertIsInstance(strat, DultProtocolStrategy)

    def test_default_dult(self):
        strat = DultProtocolStrategy()

        def fake_sample_item(token, weighted_items):
            if weighted_items ==[WeightedRange(0, 1, 1)]:
                return 1
            if weighted_items == [WeightedRange(0, 22, 1)]:
                return 4
            raise AssertionError(f"Unexpected argument")

        strat.sampler = Mock()
        strat.sampler.sample_int.side_effect = fake_sample_item

        strat.pseudomizer.seed = "TEST"
        strat.pseudomizer.epoch = 0

        field = HexDataField("TOKEN")
        _ = strat.execute(field, None)

        self.assertEqual(len(field.get_value()), 2 * (4 + 1 + 1))

        self.assertEqual(field.get_value()[:2], 'CD')
        self.assertEqual(field.get_value()[2:4], '01')
        self.assertEqual(field.get_value()[4:], '31EECF8C')

    def test_custom_dult(self):

        strat = DultProtocolStrategy(network_id=['02', 'AB', '03'], nearby=False, payload_byte_length=3)
        self.assertEqual(strat.network_id, ['02', 'AB', '03'])
        self.assertEqual(strat.nearby, False)
        self.assertEqual(strat.payload_byte_length, 3)

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = 'AB'

        strat.pseudomizer.seed = "TEST"
        strat.pseudomizer.epoch = 0

        field = HexDataField("TOKEN")
        _ = strat.execute(field, None)

        self.assertEqual(len(field.get_value()), 2 * (3 + 1 + 1))

        self.assertEqual(field.get_value()[:2], 'AB')
        self.assertEqual(field.get_value()[2:4], '00')
        self.assertEqual(field.get_value()[4:], '31EECF')



    def test_default_channel(self):
        strat = RealChannel()

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = 37

        field = HexDataField("Channel", target_bit_length=None)

        _ = strat.execute(field, None)
        self.assertEqual(field.get_value(), "25")

        args, kwargs = strat.sampler.sample_item.call_args
        self.assertEqual(args[0], 'Channel')
        self.assertEqual(args[1], [WeightedItem(37, 1), WeightedItem(38, 1), WeightedItem(39, 1)])


    def test_custom_channel(self):
        strat = RealChannel(transitions=[[100, 25], [200, 36]], token='TOKEN')
        self.assertEqual(strat.transitions, [[100, 25], [200, 36]])

        strat.sampler = Mock()
        strat.sampler.sample_item.return_value = 36

        field = HexDataField("Channel", target_bit_length=None)

        _ = strat.execute(field, None)
        self.assertEqual(field.get_value(), "24")

        args, kwargs = strat.sampler.sample_item.call_args
        self.assertEqual(args[0], 'TOKEN')
        self.assertEqual(args[1], [WeightedItem(25, 100), WeightedItem(36, 200)])



