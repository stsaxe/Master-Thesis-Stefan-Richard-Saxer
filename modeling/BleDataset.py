from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from ble import MaskConfig, Packet
from ble.pseudomization.pseudomizer_config import PseudomizerConfig
from bpe.bpe import BleBytePairEncoder
from data_masking.MaskLabelLut import MASK_LABEL_LUT
from data_processing.LabelLut import LABEL_OTHER_DEVICE

import itertools

import random

class BLEStreamDataset(Dataset):
    def __init__(self,
                 dataset_type: str,
                 packet_df: pd.DataFrame,
                 sequence_table: pd.DataFrame,
                 stream_index: pd.DataFrame,
                 max_sequence_length: int,
                 max_token_length: int,
                 tokenizer_dict: dict,
                 mask_config_path: str,
                 mask_label_lut: dict = MASK_LABEL_LUT,
                 hex_column: str = "Hex Data",
                 label_column: str = "Label",
                 time_delta_column: str = "Time Delta",
                 channel_column: str = "Channel",
                 sequence_id_column: str = "Sequence_ID",
                 packet_pos_column: str = "Packet_in_Sequence",
                 start_row_column: str = "Start_Row",
                 num_packets_column: str = "Num_of_Packets",
                 stream_start_column: str = "Start_Packet_IDX",
                 length_column: str = "Length",
                 label_unknown: str = LABEL_OTHER_DEVICE,
                 augment_data: bool = False,
                 set_time_delta_zero: bool = False,
                 adapt_sequence_length: bool = False,
                 deterministic: bool = True
                 ):

        assert dataset_type in ['test', 'train', 'validation']
        self.augment_data = augment_data
        self.set_time_delta_zero = set_time_delta_zero
        self.adapt_sequence_length = adapt_sequence_length
        self.deterministic = deterministic

        self.data_set_type = dataset_type
        self.max_sequence_length = max_sequence_length
        self.max_token_length = max_token_length

        self.packet_df = packet_df.copy()
        self.sequence_table = sequence_table.copy()
        self.stream_index = stream_index.copy()

        self.tokenizer = BleBytePairEncoder.from_state_dict(tokenizer_dict)

        self.hex_column = hex_column
        self.time_delta_column = time_delta_column
        self.channel_column = channel_column
        self.label_column = label_column
        self.sequence_id_column = sequence_id_column
        self.packet_pos_column = packet_pos_column
        self.start_row_column = start_row_column
        self.num_packets_column = num_packets_column
        self.stream_start_column = stream_start_column
        self.length_column = length_column

        self.stream_index = self.stream_index.reset_index(drop=True).copy()
        self.sequence_table = self.sequence_table.set_index(self.sequence_id_column, drop=False)

        labels = sorted(list(self.packet_df[self.label_column].unique()))
        labels.remove(LABEL_OTHER_DEVICE)

        self.num_of_known_classes = len(labels)
        self.label_unknown = label_unknown

        self.known_label_to_id = {l: i for i, l in enumerate(sorted(labels))}

        self.mask_config_label_lut: dict[str, MaskConfig] = dict()

        for label in list(self.known_label_to_id.keys()) + [label_unknown]:
            label_mask_config_path = mask_label_lut[label]

            mask_config = MaskConfig()
            mask_config.from_yaml(mask_config_path + label_mask_config_path)

            self.mask_config_label_lut[label] = mask_config

    def _tokenize_stream(self, pkt_list: list):
        return self.tokenizer.encode_many(pkt_list)

    def _pad_stream(self, tokens: list[int], target_length: int) -> list[int]:
        if len(tokens) < target_length:
            pad_len = target_length - len(tokens)
            return tokens + [self.tokenizer.PAD_ID] * pad_len

        else:
            return tokens[:target_length]

    def __len__(self) -> int:
        return len(self.stream_index)

    def _get_stream_packets(self, idx: int) -> tuple:
        stream_row = self.stream_index.iloc[idx]

        seq_id = int(stream_row[self.sequence_id_column])
        start_packet_idx = int(stream_row[self.stream_start_column])

        length = int(stream_row[self.length_column])

        if length > self.max_sequence_length:
            length = self.max_sequence_length

        if seq_id not in self.sequence_table.index:
            raise KeyError(f"Sequence_ID {seq_id} not found in sequence_table")

        seq_row = self.sequence_table.loc[seq_id]
        start_row = int(seq_row[self.start_row_column])
        num_packets = int(seq_row[self.num_packets_column])

        if start_packet_idx < 0:
            raise IndexError(f"Negative Start_Packet_IDX: {start_packet_idx}")
        if length <= 0:
            raise IndexError(f"Non-positive stream length: {length}")
        if start_packet_idx + length > num_packets:
            raise IndexError(
                f"Requested stream exceeds sequence bounds: "
                f"Sequence_ID={seq_id}, start={start_packet_idx}, length={length}, "
                f"num_packets={num_packets}"
            )

        global_start = start_row + start_packet_idx
        global_end = global_start + length


        if self.augment_data:
            if self.deterministic:
                random.seed(idx)
                np.random.seed(idx)

            stream_packets = self.packet_df.iloc[global_start:global_end, :][[
                self.time_delta_column, self.channel_column, self.hex_column]].copy(deep=True)
            if self.deterministic:
                stream_packets = stream_packets.reset_index().sample(frac=0.8, random_state=idx).sort_index().set_index("index")
            else:
                stream_packets = stream_packets.reset_index().sample(frac=0.8).sort_index().set_index("index")

            stream_packets[self.time_delta_column] = 1_000_000 * np.random.lognormal(mean=1, sigma=1, size=len(stream_packets))
            stream_packets[self.time_delta_column] = stream_packets[self.time_delta_column].astype(int)

            stream_packets = stream_packets[[self.time_delta_column, self.channel_column, self.hex_column]]


        else:
            stream_packets = self.packet_df.iloc[global_start:global_end, :][[
                self.time_delta_column, self.channel_column, self.hex_column]].copy()

        if self.set_time_delta_zero:
            stream_packets[self.time_delta_column] = 0


        stream_packets = list(stream_packets.itertuples(index=False, name=None))

        if len(stream_packets) != length and not self.augment_data:
            raise IndexError(
                f"Retrieved slice length mismatch: expected {length}, got {len(stream_packets)}"
            )

        return stream_packets, stream_row, seq_row

    def _mask_stream(self, packets: list, idx: int, label: str):
        mask_config = self.mask_config_label_lut[label]

        if self.deterministic:
            random.seed(idx)

        config = PseudomizerConfig()
        config.seed = mask_config.global_seed + self.data_set_type
        config.epoch = random.randint(0, 1_000_000)

        mask_config.configure_pseudomizer(config)

        masked_packets = []

        for pkt_tuple in packets:
            pkt = Packet()
            pre_mask = pkt_tuple[2]
            pkt.from_string(pkt_tuple[2], parse_mode='tolerant')
            pkt.mask(mask_config)
            post_mask = pkt.to_string()

            assert pre_mask != post_mask


            mask_config.rotate_epoch('packet')

            masked_packets.append((pkt_tuple[0], pkt_tuple[1], pkt.to_string()))

        return masked_packets

    def __getitem__(self, idx: int):
        stream_packets, stream_row, seq_row = self._get_stream_packets(idx)

        label_value = stream_row[self.label_column]

        if label_value not in self.known_label_to_id and label_value != self.label_unknown:
            raise KeyError(f"Label {label_value!r} not found in label_to_id mapping")

        stream_packets = self._mask_stream(stream_packets, idx, label_value)

        tokenized_packets = self._tokenize_stream(stream_packets)

        tokens = list(itertools.chain(*tokenized_packets))

        if self.adapt_sequence_length:
            if self.deterministic:
                random.seed(idx)
            fraction = random.uniform(0.02, 1)
            tokens = tokens[:int(fraction * int(self.max_token_length))]

        tokenized_packets = self._pad_stream(tokens, self.max_token_length)
        tokenized_packets = torch.tensor(tokenized_packets).long()

        if label_value == self.label_unknown:
            target = torch.ones(self.num_of_known_classes).float() / self.num_of_known_classes
        else:
            target = F.one_hot(torch.tensor(self.known_label_to_id[label_value], dtype=torch.long),
                               num_classes=self.num_of_known_classes).to(torch.float32)

        if label_value == self.label_unknown:
            label_id = self.num_of_known_classes
        else:
            label_id = self.known_label_to_id[label_value]

        return {"tokens": tokenized_packets, "target": target, "label": label_value, "label_id": label_id, "masked packets": stream_packets}
