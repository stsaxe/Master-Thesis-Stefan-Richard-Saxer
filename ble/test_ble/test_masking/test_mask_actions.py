import unittest

from ble.components.packet.Packet import Packet
from ble.fields.HexDataField import HexDataField
from ble.masking.MaskActions import MaskBleAddress, MaskHexData
from ble.pseudomization.epoch_rotation_enum import EpochRotation
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from ble.yaml.YamlRegistry import ACTION_REGISTRY


class TestMaskActions(unittest.TestCase):
    def test_mask_address_standard (self):
        action = MaskBleAddress(rotation_type="stream", length=10)

        field = HexDataField("TOKEN", "123456789A")

        action.execute(field, Packet())
        self.assertEqual(field.get_value(), "880CB13437")
        self.assertEqual(len(field.get_value()), 10)


        config = PseudomizerConfig()
        config.seed = 'TEST'
        config.epoch = 0

        action.configure_pseudomizer(config)
        action.execute(field, Packet())
        self.assertEqual(field.get_value(), "CD31EECF8C")


    def test_mask_address_create_registry(self):
        action = ACTION_REGISTRY.create("mask_ble_address", rotation_type='packet', token='TEST')

        self.assertIsInstance(action, MaskBleAddress)
        self.assertEqual(action.token, 'TEST')
        self.assertEqual(action.pseudomizer.rotation_type, EpochRotation.PACKET)

    def test_mask_address_create_registry_kwargs(self):

        kwargs = {"rotation_type": "packet", "token": "TEST", "length": 10}

        action = ACTION_REGISTRY.create("mask_ble_address", **kwargs)

        self.assertIsInstance(action, MaskBleAddress)
        self.assertEqual(action.token, 'TEST')
        self.assertEqual(action.length, 10)
        self.assertEqual(action.pseudomizer.rotation_type, EpochRotation.PACKET)


    def test_mask_address__custom_token(self):
        field = HexDataField("MyField", "123456789ABC")


        action = MaskBleAddress(rotation_type="stream", token="TOKEN")
        action.pseudomizer.seed = 'TEST'

        action.execute(field, Packet())

        self.assertEqual(field.get_value(), "CD31EECF8CAB")

    def test_mask_address_rotate_epoch (self):
        action = MaskBleAddress(rotation_type="stream")
        action.pseudomizer.seed = 'TEST'

        field = HexDataField("TOKEN", "123456789ABC")

        action.execute(field, Packet())
        self.assertEqual(field.get_value(), "CD31EECF8CAB")
        self.assertEqual(len(field.get_value()), 12)

        # test idempotency
        action.rotate_epoch(rotation_type="packet")
        action.execute(field, Packet())
        self.assertEqual(field.get_value(), "CD31EECF8CAB")
        self.assertEqual(len(field.get_value()), 12)

        # rotate epoch
        action.rotate_epoch(rotation_type="stream")
        action.execute(field, Packet())
        self.assertEqual(field.get_value(), "570BFE475367")
        self.assertEqual(len(field.get_value()), 12)

    def test_mask_address_assertions(self):
        action = MaskBleAddress(rotation_type="stream")
        action.pseudomizer.seed = 'TEST'

        field = HexDataField("TOKEN", "123456789ABCEF")

        with self.assertRaises(AssertionError):
            action.execute(field, Packet())


    def test_mask_hex_data_field_simple(self):
        field = HexDataField("MyField", "123456789ABC")

        action = MaskHexData(rotation_type="stream", token="TOKEN")
        action.pseudomizer.seed = 'TEST'

        action.execute(field, Packet())

        self.assertEqual(field.get_value(), "CD31EECF8CAB")

    def test_mask_hex_data_field_complex(self):
        field = HexDataField("MyField", "123456789ABC")

        action = MaskHexData(rotation_type="stream", token="TOKEN", start= 1,end = -3, step = 2)
        action.pseudomizer.seed = 'TEST'

        action.execute(field, Packet())

        self.assertEqual(field.get_value(), "1C3D53719ABC")

    def test_mask_hex_data_field_complex_v2(self):
        field = HexDataField("MyField", "123456789ABC")

        action = MaskHexData(rotation_type="stream", token="TOKEN", start=1, end=8, step=2)
        action.pseudomizer.seed = 'TEST'

        action.execute(field, Packet())

        self.assertEqual(field.get_value(), "1C3D53719ABC")


    def test_mask_hex_data_field_complex_with_field_token(self):
        field = HexDataField("TOKEN", "123456789ABC")

        action = MaskHexData(rotation_type="stream", start=1, end=8, step=2)
        action.pseudomizer.seed = 'TEST'

        action.execute(field, Packet())

        self.assertEqual(field.get_value(), "1C3D53719ABC")

    def test_mask_hex_data_create_registry(self):
        action = ACTION_REGISTRY.create("mask_hex_data", rotation_type='packet', token='TEST', start = 1, step = 4)

        self.assertIsInstance(action, MaskHexData)
        self.assertEqual(action.token, 'TEST')
        self.assertEqual(action.start, 1)
        self.assertEqual(action.end, None)
        self.assertEqual(action.step, 4)
        self.assertEqual(action.pseudomizer.rotation_type, EpochRotation.PACKET)


















