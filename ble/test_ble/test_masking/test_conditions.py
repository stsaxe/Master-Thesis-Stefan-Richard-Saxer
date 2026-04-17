import unittest

from ble.components.packet.Packet import Packet
from ble.masking.MaskConditions import AppleContinuityCondition, SamsungSmartThingsFindCondition, TileFindMyCondition, \
    GoogleFindMyHubCondition, DultStandardCondition


class TestMaskConditionContinuity(unittest.TestCase):
    def test_condition_is_continuity(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition("12")

        self.assertTrue(condition.apply(pkt))

    def test_condition_is_continuity_list_false(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition(["10", '11', '14'])

        self.assertFalse(condition.apply(pkt))

    def test_condition_is_continuity_list(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4C00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition(["10", '11', '12'])

        self.assertTrue(condition.apply(pkt))

    def test_condition_is_continuity_wrong_type(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4c00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition("13")

        self.assertFalse(condition.apply(pkt))

    def test_condition_is_continuity_wrong_type_integer(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4c00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition(13)

        self.assertFalse(condition.apply(pkt))

    def test_condition_is_continuity_integer(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FF4c00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition(12)

        self.assertTrue(condition.apply(pkt))

    def test_condition_wrong_comapny_id(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06FFABCD123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition("12")

        self.assertFalse(condition.apply(pkt))

    def test_condition_double_adv_type(self):
        pdu_string = "6014" + "ABCD12345678" + '06FF4c00123456' + '06FF4c00123456'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition("12")

        self.assertFalse(condition.apply(pkt))


    def test_condition_adv_too_short(self):
        pdu_string = "6010" + "ABCD12345678" + '02011A' + '020ACB' + '03FF4C00'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = AppleContinuityCondition("12")

        self.assertFalse(condition.apply(pkt))

class TestSamsungSmartThingsFindCondition(unittest.TestCase):
    def test_smart_things_find(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06165AFD151234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = SamsungSmartThingsFindCondition()
        self.assertTrue(condition.apply(pkt))

    def test_smart_things_find_wrong_uuid(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '06165ABD151234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = SamsungSmartThingsFindCondition()
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_double_data(self):
        pdu_string = "6014" + "ABCD12345678" + '06165AFD151234' + '06165AFD151234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = SamsungSmartThingsFindCondition()
        self.assertFalse(condition.apply(pkt))


class TestTileFindMyCondition(unittest.TestCase):
    def test_smart_things_find(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616EDFE151234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = TileFindMyCondition()
        self.assertTrue(condition.apply(pkt))

    def test_smart_things_find_wrong_uuid(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616ABFE151234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = TileFindMyCondition()
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_double_data(self):
        pdu_string = "6014" + "ABCD12345678" + '0616EDFE151234' + '0616EDFE151234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = TileFindMyCondition()
        self.assertFalse(condition.apply(pkt))

class TestGoogleFindMyCondition(unittest.TestCase):
    def test_smart_things_find(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616AAFE401234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = GoogleFindMyHubCondition()
        self.assertTrue(condition.apply(pkt))

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616AAFE411234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = GoogleFindMyHubCondition()
        self.assertTrue(condition.apply(pkt))

    def test_smart_things_find_wrong_uuid(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616ABFE411234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = GoogleFindMyHubCondition()
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_wrong_data_prefix(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616AAFE121234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = GoogleFindMyHubCondition()
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_wrong_data_length(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0316AAFE'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='tolerant')
        pkt.update()

        condition = GoogleFindMyHubCondition()
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_double_data(self):
        pdu_string = "6014" + "ABCD12345678" + '0616AAFE401234' + '0616AAFE401234'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = GoogleFindMyHubCondition()
        self.assertFalse(condition.apply(pkt))



class TestDultStandardCondition(unittest.TestCase):
    def test_smart_things_find(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616B2FC01ABCD'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = DultStandardCondition(network_id="01")
        self.assertTrue(condition.apply(pkt))

        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616B2FCABABCD'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = DultStandardCondition(network_id="AB")
        self.assertTrue(condition.apply(pkt))



    def test_smart_things_find_wrong_uuid(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616B5FC01ABCD'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = DultStandardCondition(network_id="01")
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_wrong_data_prefix(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0616B2FC03ABCD'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = DultStandardCondition(network_id="01")
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_wrong_data_length(self):
        pdu_string = "6013" + "ABCD12345678" + '02011A' + '020ACB' + '0316B2FC'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='tolerant')
        pkt.update()

        condition = DultStandardCondition(network_id="01")
        self.assertFalse(condition.apply(pkt))

    def test_smart_things_find_double_data(self):
        pdu_string = "6014" + "ABCD12345678" + '0616B2FC01ABCD' + '0616B2FC01ABCD'
        access_address = "12345678"
        crc = "AB12CD"

        pkt = Packet()
        pkt.from_string(access_address + pdu_string + crc, parse_mode='normal')

        condition = DultStandardCondition(network_id="01")
        self.assertFalse(condition.apply(pkt))



