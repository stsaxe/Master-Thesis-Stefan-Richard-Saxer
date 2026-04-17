from __future__ import annotations

import datetime
from typing import Self, TYPE_CHECKING, override, Iterable, Any

import pandas as pd
from scapy.layers.bluetooth4LE import BTLE

from ble import HelperMethods
from ble.components.MultiNodeContainer import MultiNodeContainer
from ble.components.packet.Packet import Packet
from scapy.utils import hexstr, PcapWriter
from tqdm import tqdm
from scapy.all import PcapNgReader

from ble.errors.OverLengthError import OverLengthError
from ble.errors.ParseError import ParseError
from ble.generation.GenConfig import GenConfig
from ble.masking.MaskConfig import MaskConfig

if TYPE_CHECKING:
    from ble.parse_policy.ParsePolicyInterface import ParsePolicyInterface


class BleStream(Iterable):
    def __init__(self):
        self.packets: MultiNodeContainer = MultiNodeContainer(name="Stream", component_type=Packet)
        self.__iter = 0
        self.__reset_iter()


    def to_string_list(self, include_delta_time_ms: bool = False, include_channel: bool = False,
                       include_rssi: bool = False, prefix: bool = False) -> list[str]:

        pkt_list = []

        time_prev = 0.0

        for idx, pkt in enumerate(self.packets):
            out = "0x" * prefix + ""

            if include_delta_time_ms:
                if idx == 0:
                    time_prev = pkt.time
                    time_delta = '00000000'


                else:
                    current_time = pkt.time
                    time_delta = int((current_time - time_prev).total_seconds() * 1_000_000)
                    time_prev = current_time

                    assert 0 <= time_delta <= 2 ** 32 - 1

                    time_delta = hex(time_delta)[2:].zfill(8).upper()

                out += time_delta

            if include_channel:
                assert pkt.channel.get_length() > 0
                out += pkt.channel.to_string()

            if include_rssi:
                assert pkt.rssi.get_length() > 0
                out += pkt.rssi.to_string()

            out += pkt.to_string()
            pkt_list.append(out)

        return pkt_list

    def add_packet(self, packet: Packet, update: bool = False) -> Self:
        assert isinstance(packet, Packet)

        if update:
            packet.update()

        self.packets.append(packet)

        return self

    def from_pcap_file(self, path: str, parse_mode: str | ParsePolicyInterface = 'normal', update: bool = False,
                       ignore_error: bool = False, fill_empty_packet: bool = False) -> Self:
        assert isinstance(path, str), "path is not a string"

        self.clear()

        reader = PcapNgReader(path)

        for idx, pkt in tqdm(enumerate(reader)):
            hex_string = hexstr(pkt, onlyhex=1)
            hex_string = hex_string.replace(" ", "")

            parsed_packet = Packet()

            try:
                parsed_packet.from_string(hex_string[34:], parse_mode=parse_mode)
                if update:
                    parsed_packet.update()

            except ParseError or ValueError or AssertionError as e:
                if not ignore_error:
                    raise e
                else:
                    if fill_empty_packet:
                        parsed_packet = Packet()
                    else:
                        continue

            parsed_packet.set_time(float(pkt.time))
            parsed_packet.set_channel(hex_string[18:20])
            parsed_packet.set_rssi(hex_string[20:22])

            self.add_packet(parsed_packet, update=False)

        self.sort_by_time(ascending=True)

        return self

    def sort_by_time(self, ascending: bool = True) -> Self:
        self.packets.components.sort(key=lambda pkt: pkt.time, reverse=not ascending)
        return self

    def merge(self, other: Self) -> Self:
        self.packets.components.extend(other.packets.components)
        return self

    def clear(self) -> Self:
        self.packets.clear()
        return self

    def update(self) -> None:
        for pkt in self.packets:
            pkt.update()

    def mask(self, mask_config: MaskConfig) -> MaskConfig:
        for packet in tqdm(self.packets):
            packet.mask(mask_config)

            mask_config.rotate_epoch(rotation_type="packet")

        return mask_config

    def to_pcap_file(self, path: str) -> None:
        assert isinstance(path, str), "path is not a string"

        self.sort_by_time(ascending=True)

        writer = PcapWriter(path, sync=True)

        for _, pkt in tqdm(enumerate(self.packets)):
            pkt_scapy = BTLE(bytes.fromhex(pkt.to_string()))
            pkt_scapy.time = pkt.get_time()
            writer.write(pkt_scapy)

        writer.close()

    def to_pandas(self) -> pd.DataFrame:
        df_data = []

        for pkt in self.packets:
            channel = pkt.get_channel(integer=True)
            rssi = pkt.get_rssi(integer=True)
            time = pkt.get_time()

            try:
                source_address = pkt.pdu.advertising_address.get_value()
            except:
                source_address = ""

            row = [time, source_address, channel, rssi, pkt.to_string()]
            df_data.append(row)

        df = pd.DataFrame(df_data, columns=['Time', 'Source', 'Channel', 'RSSI', 'Hex Data'])

        return df

    def from_pandas(self, df: pd.DataFrame, parse_mode: str | ParsePolicyInterface = 'normal', update: bool = False,
                    ignore_error: bool = False, fill_empty_packet: bool = False) -> Self:

        assert isinstance(df, pd.DataFrame), "df is not a DataFrame"

        self.clear()

        for idx, row in df.iterrows():
            time = row['Time']
            channel = row['Channel']
            rssi = row['RSSI']
            hex_string = row['Hex Data']

            try:
                pkt = Packet()
                pkt.from_string(hex_string, parse_mode=parse_mode)

                if update:
                    pkt.update()

            except ParseError or ValueError or AssertionError as e:
                if not ignore_error:
                    raise e
                else:
                    if fill_empty_packet:
                        pkt = Packet()
                    else:
                        continue

            pkt.set_time(time)
            pkt.set_channel(channel)
            pkt.set_rssi(rssi)

            self.add_packet(pkt)

        self.sort_by_time(ascending=True)

        return self

    def __len__(self) -> int:
        return len(self.packets)

    def generate(self, config: GenConfig, num_of_packets: int) -> Self:
        assert isinstance(config, GenConfig)
        assert isinstance(num_of_packets, int)

        self.clear()

        for _ in tqdm(range(num_of_packets)):
            pkt = Packet()

            while True:
                try:
                    pkt.generate(config)
                    break
                except OverLengthError:
                    continue

            config.rotate_epoch(rotation_type="packet")
            self.add_packet(pkt, update=False)

        return self

    def __reset_iter(self) -> None:
        self.__iter = 0

    @override
    def __iter__(self) -> Self:
        self.__reset_iter()
        return self

    def __next__(self) -> Any:
        try:
            self.__iter += 1
            return self.packets[self.__iter - 1]

        except IndexError:
            raise StopIteration
