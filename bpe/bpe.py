import math
import random
import multiprocessing as mp
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable, Any
from tqdm import tqdm
import os


def count_payload_pairs_worker(args):
    sequences, mergeable_token_ids = args
    mergeable = mergeable_token_ids
    counts = Counter()

    for seq in sequences:
        n = len(seq)
        for i in range(n - 1):
            a = seq[i]
            b = seq[i + 1]
            if a in mergeable and b in mergeable:
                counts[(a, b)] += 1

    return counts


def apply_payload_merge_worker(args):
    sequences, pair, new_token_id = args
    a, b = pair
    out = []

    for seq in sequences:
        merged = []
        i = 0
        n = len(seq)

        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                merged.append(new_token_id)
                i += 2
            else:
                merged.append(seq[i])
                i += 1

        out.append(merged)

    return out

@dataclass
class BPEConfig:
    target_vocab_size: int = 1024
    min_pair_count: int = 2
    random_seed: int = 0
    time_num_bytes: int = 4

class BleBytePairEncoder:
    def __init__(self, config: BPEConfig) -> None:
        self.config = config
        self.num_workers = os.cpu_count()

        self.PAD_ID = 0
        self.EOS_ID = 1
        self.UNK_ID = 2
        self.UNK_CHANNEL_ID = 3

        self.special_token_to_id = {
            "[PAD]": self.PAD_ID,
            "[EOS]": self.EOS_ID,
            "[UNK]": self.UNK_ID,
            "[UNK_CHANNEL]": self.UNK_CHANNEL_ID,
        }

        self.id_to_special_token = {v: k for k, v in self.special_token_to_id.items()}

        self.time_num_bytes = config.time_num_bytes
        self.TIME_BASE = len(self.special_token_to_id)
        self.TIME_BLOCK_SIZE = 256

        # Payload byte token block
        self.PAYLOAD_BASE = self.TIME_BASE + self.TIME_BLOCK_SIZE
        self.PAYLOAD_BLOCK_SIZE = 256

        # Channel tokens are dynamic
        self.CHANNEL_BASE = self.PAYLOAD_BASE + self.PAYLOAD_BLOCK_SIZE
        self.channel_value_to_id: Dict[int, int] = {}
        self.id_to_channel_value: Dict[int, int] = {}

        # BPE merges
        self.merges: List[Tuple[Tuple[int, int], int]] = []
        self.merge_parents: Dict[int, Tuple[int, int]] = {}

        # Tokens allowed to participate in payload merges
        self.mergeable_payload_token_ids = set(
            self.PAYLOAD_BASE + i for i in range(256)
        )

        self.vocab_size = self.CHANNEL_BASE

    def time_token_id(self, byte_value: int) -> int:
        return self.TIME_BASE + byte_value

    def payload_byte_token_id(self, byte_value: int) -> int:
        return self.PAYLOAD_BASE + byte_value

    def register_channels(self, channels: Iterable[int]):
        for ch in sorted(set(channels)):
            if ch not in self.channel_value_to_id:
                token_id = self.vocab_size
                self.channel_value_to_id[ch] = token_id
                self.id_to_channel_value[token_id] = ch
                self.vocab_size += 1

    def channel_token_id(self, channel: int) -> int:
        return self.channel_value_to_id.get(channel, self.UNK_CHANNEL_ID)

    @staticmethod
    def clean_hex_value(value: str) -> str:
        assert isinstance(value, str)
        assert len(value) >= 0

        special_symbols = ["_", ":", " ", "-"]
        value = value.strip().lower()

        for symbol in special_symbols:
            value = value.replace(symbol, "")

        if value.startswith("0x") or value.startswith("0X"):
            value = value[2:]

        value = value.upper()

        if any(c not in "0123456789ABCDEF" for c in value):
            raise ValueError(f"Invalid hex string: {value}")

        return value

    @staticmethod
    def normalize_hex_string(hex_string: str) -> str:
        out = BleBytePairEncoder.clean_hex_value(hex_string)
        assert len(out) % 2 == 0, f"Hex string must have even length: {hex_string}"
        return out

    @classmethod
    def hex_to_bytes(cls, hex_string: str) -> List[int]:
        s = cls.normalize_hex_string(hex_string)
        return [int(s[i:i + 2], 16) for i in range(0, len(s), 2)]

    def get_max_time_val(self) -> int:
        max_val = (1 << (8 * self.time_num_bytes)) - 1
        return max_val

    def encode_time_tokens(self, delta_time: int) -> List[int]:
        if delta_time < 0:
            raise ValueError("delta_time must be >= 0")

        if delta_time > self.get_max_time_val():
            raise ValueError(
                f"delta_time {delta_time} does not fit into {self.time_num_bytes} bytes"
            )

        raw = delta_time.to_bytes(
            self.time_num_bytes,
            byteorder='big',
            signed=False,
        )
        return [self.time_token_id(b) for b in raw]

    def encode_base(
            self,
            delta_time: int,
            channel: int,
            packet_hex: str) -> List[int]:

        payload = self.hex_to_bytes(packet_hex)

        seq = []

        seq.extend(self.encode_time_tokens(delta_time))
        seq.append(self.channel_token_id(channel))
        seq.extend(self.payload_byte_token_id(b) for b in payload)
        seq.append(self.EOS_ID)

        return seq

    def _prepare_training_sequences(self, records: List[Tuple[int, int, str]], ) -> List[List[int]]:
        self.register_channels(ch for _, ch, _ in records)

        return [self.encode_base(dt, ch, hx) for dt, ch, hx in records]

    def _make_shards(self, sequences: List[List[int]]) -> List[List[List[int]]]:
        if len(sequences) == 0:
            return []

        num_workers = min(self.num_workers, len(sequences))
        chunk_size = math.ceil(len(sequences) / num_workers)

        return [sequences[i:i + chunk_size] for i in range(0, len(sequences), chunk_size)]


    def fit(self, records: List[Tuple[int, int, str]], verbose: bool = True) -> None:
        sequences = self._prepare_training_sequences(records)

        if verbose:
            print(f"Prepared {len(sequences):,} sequences")
            print(f"Known channels: {len(self.channel_value_to_id):,}")
            print(f"Base vocab before merges: {self.vocab_size:,}")
            print(f"Target vocab size: {self.config.target_vocab_size:,}")
            print(f"Workers: {self.num_workers}")

        target_merges = self.config.target_vocab_size - self.vocab_size
        if target_merges <= 0:
            if verbose:
                print("Target vocab size already reached by base vocab.")
            return

        for _ in tqdm(range(target_merges)):
            shards = self._make_shards(sequences)

            worker_args = [(shard, self.mergeable_payload_token_ids) for shard in shards]

            with mp.Pool(processes=self.num_workers) as pool:
                counters = pool.map(count_payload_pairs_worker, worker_args)

            total_counts = Counter()
            for c in counters:
                total_counts.update(c)

            if not total_counts:
                if verbose:
                    print("No more mergeable payload pairs.")
                break

            best_pair, best_count = total_counts.most_common(1)[0]

            if best_count < self.config.min_pair_count:
                if verbose:
                    print(
                        f"Stopping: best payload pair count {best_count} "
                        f"< min_pair_count {self.config.min_pair_count}"
                    )
                break

            new_token_id = self.vocab_size
            self.vocab_size += 1
            self.merges.append((best_pair, new_token_id))
            self.merge_parents[new_token_id] = best_pair
            self.mergeable_payload_token_ids.add(new_token_id)

            worker_args = [(shard, best_pair, new_token_id) for shard in shards]

            with mp.Pool(processes=self.num_workers) as pool:
                merged_shards = pool.map(apply_payload_merge_worker, worker_args)

            sequences = [seq for shard in merged_shards for seq in shard]

        if verbose:
            print(f"Training complete. Final vocab size: {self.vocab_size:,}")
            print(f"Learned payload merges: {len(self.merges):,}")


    def _apply_merges(self, seq: List[int]) -> List[int]:
        for (a, b), new_token_id in self.merges:
            merged = []
            i = 0
            n = len(seq)

            while i < n:
                if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                    merged.append(new_token_id)
                    i += 2
                else:
                    merged.append(seq[i])
                    i += 1

            seq = merged

        return seq

    def encode(self, delta_time: int, channel: int, packet_hex: str) -> List[int]:
        seq = self.encode_base(delta_time, channel, packet_hex)
        return self._apply_merges(seq)

    def encode_many(self, records: List[Tuple[int, int, str]]) -> List[List[int]]:
        return [self.encode(dt, ch, hx) for dt, ch, hx in records]



    def token_to_string(self, token_id: int) -> str:
        if token_id in self.id_to_special_token:
            return self.id_to_special_token[token_id]

        if self.TIME_BASE <= token_id < self.TIME_BASE + self.TIME_BLOCK_SIZE:
            val = token_id - self.TIME_BASE
            return f"[T_{val:02X}]"

        if self.PAYLOAD_BASE <= token_id < self.PAYLOAD_BASE + self.PAYLOAD_BLOCK_SIZE:
            val = token_id - self.PAYLOAD_BASE
            return f"{val:02X}"

        if token_id in self.id_to_channel_value:
            return f"[CH_{self.id_to_channel_value[token_id]}]"

        if token_id in self.merge_parents:
            return f"<MERGE_{token_id}>"

        return "[UNK]"

    def inspect_sequence(self, token_ids: List[int]) -> List[str]:
        return [self.token_to_string(t) for t in token_ids]

    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        return self.inspect_sequence(token_ids)


    def _is_time_token(self, token_id: int) -> bool:
        return self.TIME_BASE <= token_id < self.TIME_BASE + self.TIME_BLOCK_SIZE

    def _is_payload_base_token(self, token_id: int) -> bool:
        return self.PAYLOAD_BASE <= token_id < self.PAYLOAD_BASE + self.PAYLOAD_BLOCK_SIZE

    def _is_channel_token(self, token_id: int) -> bool:
        return token_id in self.id_to_channel_value or token_id == self.UNK_CHANNEL_ID

    def _payload_token_to_byte(self, token_id: int) -> int:
        if not self._is_payload_base_token(token_id):
            raise ValueError(f"Token {token_id} is not a base payload byte token")
        return token_id - self.PAYLOAD_BASE

    def _expand_payload_token(self, token_id: int) -> List[int]:
        if self._is_payload_base_token(token_id):
            return [token_id]

        if token_id in self.merge_parents:
            left, right = self.merge_parents[token_id]
            return self._expand_payload_token(left) + self._expand_payload_token(right)

        raise ValueError(f"Token {token_id} is not a decodable payload token")

    def decode_payload_hex(self, token_ids: List[int]) -> str:
        expanded = []
        for t in token_ids:
            expanded.extend(self._expand_payload_token(t))

        byte_vals = [self._payload_token_to_byte(t) for t in expanded]
        return "".join(f"{b:02X}" for b in byte_vals)


    def decode_packet(self, token_ids: List[int], allow_missing_eos: bool = False) -> Dict[str, Any]:
        seq = list(token_ids)

        if len(seq) > 0 and seq[-1] == self.EOS_ID:
            seq = seq[:-1]
        elif not allow_missing_eos:
            raise ValueError("Packet does not end with [EOS]")

        if len(seq) < self.time_num_bytes + 1:
            raise ValueError("Sequence too short to contain time and channel")

        # Decode fixed-width time
        time_tokens = seq[:self.time_num_bytes]
        for t in time_tokens:
            if not self._is_time_token(t):
                raise ValueError(f"Expected time token, got {self.token_to_string(t)}")

        time_bytes = bytes(t - self.TIME_BASE for t in time_tokens)
        delta_time = int.from_bytes(
            time_bytes,
            byteorder='big',
            signed=False,
        )

        # Decode channel
        ch_token = seq[self.time_num_bytes]
        if ch_token == self.UNK_CHANNEL_ID:
            channel = None
        elif ch_token in self.id_to_channel_value:
            channel = self.id_to_channel_value[ch_token]
        else:
            raise ValueError(f"Expected channel token, got {self.token_to_string(ch_token)}")

        # Decode payload
        payload_tokens = seq[self.time_num_bytes + 1:]
        packet_hex = self.decode_payload_hex(payload_tokens) if payload_tokens else ""

        return {
            "Time Delta": delta_time,
            "Channel": channel,
            "Hex Data": packet_hex,
            }


    def decode_stream(self, token_ids: List[int], allow_incomplete_last_packet: bool = True) -> List[Dict[str, Any]]:
        packets = []

        current = []
        for t in token_ids:
            current.append(t)
            if t == self.EOS_ID:
                packets.append(self.decode_packet(current, allow_missing_eos=False))
                current = []

        if len(current) > 0 and allow_incomplete_last_packet:
            packets.append(self.decode_packet(current, allow_missing_eos=True))

        return packets


    def pad_batch(self, batch_ids: List[List[int]], max_length: Optional[int] = None, truncation: bool = True,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        if not batch_ids:
            return [], []

        if max_length is None:
            max_length = max(len(x) for x in batch_ids)

        padded = []
        attention_mask = []

        for seq in batch_ids:
            if truncation and len(seq) > max_length:
                seq = seq[:max_length]

            pad_len = max_length - len(seq)
            padded.append(seq + [self.PAD_ID] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        return padded, attention_mask


    def state_dict(self) -> Dict:
        return {
            "config": self.config.__dict__,
            "channel_value_to_id": self.channel_value_to_id,
            "id_to_channel_value": self.id_to_channel_value,
            "merges": self.merges,
            "merge_parents": self.merge_parents,
            "vocab_size": self.vocab_size,
            "mergeable_payload_token_ids": list(self.mergeable_payload_token_ids),
        }

    @classmethod
    def from_state_dict(cls, state: Dict) -> "BleBytePairEncoder":
        obj = cls(BPEConfig(**state["config"]))
        obj.channel_value_to_id = {int(k): int(v) for k, v in state["channel_value_to_id"].items()}
        obj.id_to_channel_value = {int(k): int(v) for k, v in state["id_to_channel_value"].items()}
        obj.merges = [(tuple(pair), int(new_id)) for pair, new_id in state["merges"]]
        obj.merge_parents = {int(k): tuple(v) for k, v in state["merge_parents"].items()}
        obj.vocab_size = int(state["vocab_size"])
        obj.mergeable_payload_token_ids = set(int(x) for x in state["mergeable_payload_token_ids"])
        return obj


