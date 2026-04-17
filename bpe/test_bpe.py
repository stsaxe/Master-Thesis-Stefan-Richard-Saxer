import unittest
from bpe.bpe import BPEConfig, BleBytePairEncoder


class TestDedicatedTimeChannelBPE(unittest.TestCase):
    @staticmethod
    def make_tokenizer(**kwargs):
        config = BPEConfig(
            target_vocab_size=kwargs.pop("target_vocab_size", 1024),
            min_pair_count=kwargs.pop("min_pair_count", 2),
            random_seed=kwargs.pop("random_seed", 0),
            time_num_bytes=kwargs.pop("time_num_bytes", 4)
        )

        return BleBytePairEncoder(config)

    def test_config(self):
        config = BPEConfig()

        self.assertEqual(config.target_vocab_size, 1024)
        self.assertEqual(config.min_pair_count, 2)
        self.assertEqual(config.random_seed, 0)
        self.assertEqual(config.time_num_bytes, 4)

    def test_init(self):
        tokenizer = self.make_tokenizer(target_vocab_size=50, min_pair_count=3, random_seed=1, time_num_bytes=8)

        self.assertEqual(tokenizer.config.target_vocab_size, 50)
        self.assertEqual(tokenizer.config.min_pair_count, 3)
        self.assertEqual(tokenizer.config.random_seed, 1)
        self.assertEqual(tokenizer.config.time_num_bytes, 8)

    def test_normalize_hex_string_plain(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("0201060AFF"), "0201060AFF")

    def test_normalize_hex_string_little(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("0201060afF"), "0201060AFF")

    def test_normalize_hex_string_prefix(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("0x1201060AFF"), "1201060AFF")

    def test_normalize_hex_string_zeros(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("00001201060AFF"), "00001201060AFF")

    def test_normalize_hex_string_with_spaces(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("02 01 06 0A FF"), "0201060AFF")

    def test_normalize_hex_string_with_colons(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("02:01:06:0A:FF"), "0201060AFF")

    def test_normalize_hex_string_with_underscore(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.normalize_hex_string("_0201_06_0AFF"), "0201060AFF")

    def test_normalize_hex_string_odd_length_raises(self):
        tok = self.make_tokenizer()
        with self.assertRaises(AssertionError):
            tok.normalize_hex_string("ABC")

    def test_hex_to_bytes(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.hex_to_bytes("0201060AFF"), [2, 1, 6, 10, 255])

    def test_time_token_encoding_big_endian(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        encoded = tok.encode_time_tokens(66219)  # 0x0001024AB

        expected = [
            tok.time_token_id(0),
            tok.time_token_id(1),
            tok.time_token_id(2),
            tok.time_token_id(171),
        ]

        self.assertEqual(encoded, expected)

    def test_time_token_negative_raises(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        with self.assertRaises(ValueError):
            tok.encode_time_tokens(-1)

    def test_time_token_overflow_raises(self):
        tok = self.make_tokenizer(time_num_bytes=1)
        with self.assertRaises(ValueError):
            tok.encode_time_tokens(256)

    def test_register_channels(self):
        tok = self.make_tokenizer()
        tok.register_channels([36, 37, 39, 36, 36, 37])

        self.assertIn(36, tok.channel_value_to_id)
        self.assertIn(37, tok.channel_value_to_id)
        self.assertIn(39, tok.channel_value_to_id)
        self.assertEqual(len(tok.channel_value_to_id), 3)

    def test_channel_token_known(self):
        tok = self.make_tokenizer()
        tok.register_channels([7, 17, 99])

        self.assertEqual(tok.id_to_channel_value[tok.channel_token_id(17)], 17)

    def test_channel_token_unknown_returns_unk_channel(self):
        tok = self.make_tokenizer()
        tok.register_channels([37, 38, 39])

        self.assertEqual(tok.channel_token_id(123), tok.UNK_CHANNEL_ID)

    def test_encode_base(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38, 39, 40])

        seq = tok.encode_base(300, 38, "0201AB")

        expected = [
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x01),
            tok.time_token_id(0x2C),
            tok.channel_token_id(38),
            tok.payload_byte_token_id(2),
            tok.payload_byte_token_id(1),
            tok.payload_byte_token_id(171),
            tok.EOS_ID,
        ]
        self.assertEqual(seq, expected)

    def test_encode_base_wth_unknown_channel_and_empty_sequence(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38, 39, 40])

        seq = tok.encode_base(300, 41, "")

        expected = [
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x01),
            tok.time_token_id(0x2C),
            tok.UNK_CHANNEL_ID,

            tok.EOS_ID,
        ]
        self.assertEqual(seq, expected)

    def test_prepare_training_sequences_registers_channels(self):
        tok = self.make_tokenizer()
        records = [
            (0, 37, "AA"),
            (1, 38, "BB"),
            (2, 39, "CC"),
        ]

        seqs = tok._prepare_training_sequences(records)

        self.assertEqual(len(seqs), 3)
        self.assertSetEqual(set(tok.channel_value_to_id.keys()), {37, 38, 39})

    def test_fit_learns_one_merge_for_repeated_payload(self):
        tok = self.make_tokenizer(
            target_vocab_size=1_0000,
            min_pair_count=2,
            num_workers=1,
        )
        records = [
            (0, 37, "AABB"),
            (1, 38, "AABB"),
            (2, 39, "AABB"),
        ]

        tok.fit(records, verbose=False)

        self.assertEqual(len(tok.merges), 1)

        pair, new_token = tok.merges[0]
        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)

        self.assertEqual(pair, (aa, bb))
        self.assertIn(new_token, tok.merge_parents)
        self.assertEqual(tok.merge_parents[new_token], (aa, bb))

    def test_fit_does_not_learn_merge_when_pair_count_below_threshold(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            min_pair_count=3,
            num_workers=1,
        )
        records = [
            (0, 37, "AABB"),
            (1, 38, "AABB"),
            (1, 39, "1234"),
            (1, 38, "CCDD"),
        ]

        tok.fit(records, verbose=False)
        self.assertEqual(len(tok.merges), 0)

    def test_fit_only_merges_payload_tokens(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            min_pair_count=2,
            num_workers=1,
            time_num_bytes=4,
        )

        records = [
            (300, 37, "AABB"),
            (300, 37, "AABB"),
            (300, 37, "AABB"),
        ]

        tok.fit(records, verbose=False)

        self.assertEqual(len(tok.merges), 1)

        for (a, b), new_token in tok.merges:
            self.assertIn(a, tok.mergeable_payload_token_ids)
            self.assertIn(b, tok.mergeable_payload_token_ids)
            self.assertIn(new_token, tok.mergeable_payload_token_ids)

    def test_fit_increases_vocab_size(self):
        tok = self.make_tokenizer(
            target_vocab_size=700,
            min_pair_count=2,
            num_workers=1,
        )
        records = [
            (0, 37, "AABB"),
            (1, 38, "AABB"),
            (2, 39, "ACBC"),
        ]

        before = tok.vocab_size
        tok.fit(records, verbose=False)
        after = tok.vocab_size

        self.assertEqual(after, before + len(tok.channel_value_to_id) + len(tok.merges))

    def test_fit_respects_target_vocab_size(self):
        _tok = self.make_tokenizer()
        size = _tok.vocab_size

        tok = self.make_tokenizer(
            target_vocab_size=size + 3 + 2,
            min_pair_count=2,
            num_workers=1,
        )

        records = [
            (0, 37, "AABBAABB"),
            (1, 38, "AABB1234"),
            (2, 39, "1234CDEF"),
            (2, 39, "1234CDEF"),
        ]

        tok.fit(records, verbose=False)
        self.assertEqual(tok.vocab_size, size + 5)


    def test_fit_respects_smaller_target_vocab_size(self):
        _tok = self.make_tokenizer()
        size = _tok.vocab_size

        tok = self.make_tokenizer(
            target_vocab_size=size + 3 + 1,
            min_pair_count=2,
            num_workers=1,
        )

        records = [
            (0, 37, "AABBAABB"),
            (1, 38, "AABB1234"),
            (2, 39, "1234CDEF"),
        ]

        tok.fit(records, verbose=False)
        self.assertEqual(tok.vocab_size, size + 4)


    def test_apply_merges_replaces_payload(self):
        tok = self.make_tokenizer(num_workers=1, time_num_bytes=4)
        tok.register_channels([38, 39])

        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)
        new_token = tok.vocab_size
        tok.vocab_size += 1

        tok.merges = [((aa, bb), new_token)]
        tok.merge_parents[new_token] = (aa, bb)
        tok.mergeable_payload_token_ids.add(new_token)

        base = tok.encode_base(300, 38, "AABBCC")
        out = tok._apply_merges(base)

        expected = [
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x01),
            tok.time_token_id(0x2C),
            tok.channel_token_id(38),
            new_token,
            tok.payload_byte_token_id(0xCC),
            tok.EOS_ID,
        ]
        self.assertEqual(out, expected)

    def test_apply_merges_order_matters_and_supports_hierarchical_merges(self):
        tok = self.make_tokenizer(num_workers=1, time_num_bytes=4)
        tok.register_channels([38])

        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)
        cc = tok.payload_byte_token_id(0xCC)

        m1 = tok.vocab_size
        tok.vocab_size += 1
        m2 = tok.vocab_size
        tok.vocab_size += 1

        tok.merges = [
            ((aa, bb), m1),
            ((m1, cc), m2),
        ]
        tok.merge_parents[m1] = (aa, bb)
        tok.merge_parents[m2] = (m1, cc)
        tok.mergeable_payload_token_ids.update({m1, m2})

        base = tok.encode_base(0, 38, "AABBCC01")
        out = tok._apply_merges(base)

        expected = [
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.channel_token_id(38),
            m2,
            tok.payload_byte_token_id(1),
            tok.EOS_ID,
        ]
        self.assertEqual(out, expected)

    def test_apply_merges_does_not_cross_channel_payload_boundary(self):
        tok = self.make_tokenizer(num_workers=1, time_num_bytes=4)
        tok.register_channels([38])

        # Try to define an illegal merge manually:
        # (channel_token, first_payload_token) -> new_token
        ch = tok.channel_token_id(38)
        aa = tok.payload_byte_token_id(0xAA)
        illegal_merge = tok.vocab_size
        tok.vocab_size += 1

        tok.merges = [((ch, aa), illegal_merge)]

        base = tok.encode_base(0, 38, "AA")
        out = tok._apply_merges(base)

        self.assertNotEqual(base, out)

    def test_fit_never_creates_channel_payload_merge(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            min_pair_count=2,
            num_workers=1,
            time_num_bytes=4,
        )

        records = [
            (0, 38, "AABB"),
            (0, 38, "AABB"),
            (0, 38, "AACC"),
        ]
        tok.fit(records, verbose=False)

        out = tok.encode_base(0, 38, "AADD")

        expected = [
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.channel_token_id(38),
            tok.payload_byte_token_id(0xAA),
            tok.payload_byte_token_id(0xDD),
            tok.EOS_ID,
        ]

        self.assertEqual(out, expected)


    def test_encode_applies_learned_merges(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            min_pair_count=2,
            num_workers=1,
            time_num_bytes=4,
        )

        size = tok.vocab_size

        records = [
            (0, 37, "AABB1234"),
            (1, 38, "AABB34AB"),
            (2, 38, "AABBBC12EF"),
        ]
        tok.fit(records, verbose=False)

        out = tok.encode(0xAABB, 38, "12AABB34")

        # Payload should be shorter than raw 2-byte tokenization if merge learned
        # Base length would be:
        # 4 time + 1 channel + 2 payload + + Merged + [EOS] = 9
        self.assertEqual(len(out), 9)

        # Should still contain packet structure
        # size is + 2 because of channels
        expected = [
            tok.time_token_id(0x00),
            tok.time_token_id(0x00),
            tok.time_token_id(0xAA),
            tok.time_token_id(0xBB),
            tok.channel_token_id(38),
            tok.payload_byte_token_id(0x12),
            size+2,
            tok.payload_byte_token_id(0x34),
            tok.EOS_ID,
        ]

        self.assertEqual(out, expected)


    def test_token_to_string_special(self):
        tok = self.make_tokenizer()
        self.assertEqual(tok.token_to_string(tok.EOS_ID), "[EOS]")

    def test_token_to_string_time(self):
        tok = self.make_tokenizer()
        tid = tok.time_token_id(0x2C)
        self.assertEqual(tok.token_to_string(tid), "[T_2C]")

    def test_token_to_string_payload(self):
        tok = self.make_tokenizer()
        pid = tok.payload_byte_token_id(0xAA)
        self.assertEqual(tok.token_to_string(pid), "AA")

    def test_token_to_string_channel(self):
        tok = self.make_tokenizer()
        tok.register_channels([38, 39])
        cid = tok.channel_token_id(38)
        self.assertEqual(tok.token_to_string(cid), "[CH_38]")



class TestDedicatedTimeChannelBPEDecodeAndPadding(unittest.TestCase):
    @staticmethod
    def make_tokenizer(**kwargs):
        config = BPEConfig(
            target_vocab_size=kwargs.pop("target_vocab_size", 1024),
            min_pair_count=kwargs.pop("min_pair_count", 2),
            random_seed=kwargs.pop("random_seed", 0),
            time_num_bytes=kwargs.pop("time_num_bytes", 4)
        )

        return BleBytePairEncoder(config)


    def add_manual_merge(self, tok, left, right):
        new_token = tok.vocab_size
        tok.vocab_size += 1
        tok.merges.append(((left, right), new_token))
        tok.merge_parents[new_token] = (left, right)
        tok.mergeable_payload_token_ids.add(new_token)
        return new_token


    def test_decode_tokens_returns_human_readable_tokens(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        seq = tok.encode_base(300, 38, "AABB")
        decoded = tok.decode_tokens(seq)

        self.assertEqual(decoded[0:4], ["[T_00]", "[T_00]", "[T_01]", "[T_2C]"])
        self.assertEqual(decoded[4], "[CH_38]")
        self.assertEqual(decoded[5:7], ["AA", "BB"])
        self.assertEqual(decoded[7], "[EOS]")

    def test_decode_payload_hex_base_tokens(self):
        tok = self.make_tokenizer()
        payload_tokens = [
            tok.payload_byte_token_id(0xAA),
            tok.payload_byte_token_id(0xBB),
            tok.payload_byte_token_id(0xCC),
        ]

        self.assertEqual(tok.decode_payload_hex(payload_tokens), "AABBCC")

    def test_decode_payload_hex_recursive_merges(self):
        tok = self.make_tokenizer()

        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)
        cc = tok.payload_byte_token_id(0xCC)

        m1 = self.add_manual_merge(tok, aa, bb)   # AABB
        m2 = self.add_manual_merge(tok, m1, cc)   # AABBCC

        self.assertEqual(tok.decode_payload_hex([m2]), "AABBCC")



    def test_decode_packet_roundtrip_base_no_merges(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        seq = tok.encode_base(300, 38, "AABBCC")
        decoded = tok.decode_packet(seq, allow_missing_eos=True)

        self.assertEqual(decoded["Time Delta"], 300)
        self.assertEqual(decoded["Channel"], 38)
        self.assertEqual(decoded["Hex Data"], "AABBCC")


    def test_decode_packet_roundtrip_with_merges(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)
        cc = tok.payload_byte_token_id(0xCC)

        m1 = self.add_manual_merge(tok, aa, bb)   # AABB
        m2 = self.add_manual_merge(tok, m1, cc)   # AABBCC

        seq = [
            *tok.encode_time_tokens(300),
            tok.channel_token_id(38),
            m2,
            tok.EOS_ID,
        ]
        decoded = tok.decode_packet(seq, allow_missing_eos=True)

        self.assertEqual(decoded["Time Delta"], 300)
        self.assertEqual(decoded["Channel"], 38)
        self.assertEqual(decoded["Hex Data"], "AABBCC")



    def test_decode_packet_missing_eos_raises_by_default(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        seq = tok.encode_base(300, 38, "AABB")
        seq = seq[:-1]  # remove EOS

        with self.assertRaises(ValueError):
            tok.decode_packet(seq)

    def test_decode_packet_with_eos(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        seq = tok.encode_base(300, 38, "AABB")

        decoded = tok.decode_packet(seq, allow_missing_eos=False)

        self.assertEqual(decoded["Time Delta"], 300)
        self.assertEqual(decoded["Channel"], 38)
        self.assertEqual(decoded["Hex Data"], "AABB")


    def test_decode_packet_unknown_channel_decodes_to_none(self):
        tok = self.make_tokenizer(time_num_bytes=4)

        seq = [
            *tok.encode_time_tokens(7),
            tok.UNK_CHANNEL_ID,
            tok.payload_byte_token_id(0xAA),
            tok.EOS_ID,
        ]
        decoded = tok.decode_packet(seq)

        self.assertEqual(decoded["Time Delta"], 7)
        self.assertIsNone(decoded["Channel"])
        self.assertEqual(decoded["Hex Data"], "AA")

    def test_decode_packet_raises_if_time_token_is_invalid(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        seq = tok.encode_base(1, 38, "AA")
        seq[1] = tok.payload_byte_token_id(0xFF)  # corrupt first time token

        with self.assertRaises(ValueError):
            tok.decode_packet(seq)

    def test_decode_packet_raises_if_channel_token_is_invalid(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        seq = tok.encode_base(1, 38, "AA")

        channel_index = tok.time_num_bytes
        seq[channel_index] = tok.payload_byte_token_id(0x11)  # invalid channel token

        with self.assertRaises(ValueError):
            tok.decode_packet(seq)


    def test_decode_stream_multiple_packets(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([37, 38])

        seq1 = tok.encode_base(10, 37, "AABB")
        seq2 = tok.encode_base(20, 38, "CCDD")
        stream = seq1 + seq2

        packets = tok.decode_stream(stream)

        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[0]["Time Delta"], 10)
        self.assertEqual(packets[0]["Channel"], 37)
        self.assertEqual(packets[0]["Hex Data"], "AABB")

        self.assertEqual(packets[1]["Time Delta"], 20)
        self.assertEqual(packets[1]["Channel"], 38)
        self.assertEqual(packets[1]["Hex Data"], "CCDD")

    def test_decode_stream_incomplete_last_packet_allowed(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([37, 38])

        seq1 = tok.encode_base(10, 37, "AABB")
        seq2 = tok.encode_base(20, 38, "CCDD")[:-1]  # remove EOS from second packet
        stream = seq1 + seq2

        packets = tok.decode_stream(stream, allow_incomplete_last_packet=True)

        self.assertEqual(len(packets), 2)
        self.assertEqual(packets[1]["Time Delta"], 20)
        self.assertEqual(packets[1]["Channel"], 38)
        self.assertEqual(packets[1]["Hex Data"], "CCDD")

    def test_decode_stream_incomplete_last_packet_disallowed(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([37])

        seq = tok.encode_base(10, 37, "AABB")[:-1]  # remove EOS

        packets = tok.decode_stream(seq, allow_incomplete_last_packet=False)
        self.assertEqual(packets, [])

    def test_pad_batch_basic(self):
        tok = self.make_tokenizer()
        batch = [
            [10, 11, 12],
            [20, 21],
        ]

        padded, mask = tok.pad_batch(batch)

        self.assertEqual(padded, [
            [10, 11, 12],
            [20, 21, tok.PAD_ID],
        ])
        self.assertEqual(mask, [
            [1, 1, 1],
            [1, 1, 0],
        ])

    def test_pad_batch_with_explicit_max_length(self):
        tok = self.make_tokenizer()
        batch = [
            [10, 11],
            [20],
        ]

        padded, mask = tok.pad_batch(batch, max_length=4)

        self.assertEqual(padded, [
            [10, 11, tok.PAD_ID, tok.PAD_ID],
            [20, tok.PAD_ID, tok.PAD_ID, tok.PAD_ID],
        ])
        self.assertEqual(mask, [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ])

    def test_pad_batch_truncates_when_enabled(self):
        tok = self.make_tokenizer()
        batch = [
            [1, 2, 3, 4],
            [5, 6],
        ]

        padded, mask = tok.pad_batch(batch, max_length=3, truncation=True)

        self.assertEqual(padded, [
            [1, 2, 3],
            [5, 6, tok.PAD_ID],
        ])
        self.assertEqual(mask, [
            [1, 1, 1],
            [1, 1, 0],
        ])

    def test_pad_batch_without_truncation_can_produce_inconsistent_length(self):
        tok = self.make_tokenizer()
        batch = [
            [1, 2, 3, 4],
            [5, 6],
        ]

        padded, mask = tok.pad_batch(batch, max_length=3, truncation=False)

        self.assertEqual(padded[0], [1, 2, 3, 4])
        self.assertEqual(padded[1], [5, 6, tok.PAD_ID])

        self.assertEqual(mask[0], [1, 1, 1, 1])
        self.assertEqual(mask[1], [1, 1, 0])

    def test_pad_batch_empty_batch(self):
        tok = self.make_tokenizer()
        padded, mask = tok.pad_batch([])

        self.assertEqual(padded, [])
        self.assertEqual(mask, [])

import unittest


class TestDedicatedTimeChannelBPEStateDict(unittest.TestCase):
    @staticmethod
    def make_tokenizer(**kwargs):
        config = BPEConfig(
            target_vocab_size=kwargs.pop("target_vocab_size", 1024),
            min_pair_count=kwargs.pop("min_pair_count", 2),
            random_seed=kwargs.pop("random_seed", 0),
            time_num_bytes=kwargs.pop("time_num_bytes", 4)
        )

        return BleBytePairEncoder(config)

    def add_manual_merge(self, tok, left, right):
        new_token = tok.vocab_size
        tok.vocab_size += 1
        tok.merges.append(((left, right), new_token))
        tok.merge_parents[new_token] = (left, right)
        tok.mergeable_payload_token_ids.add(new_token)
        return new_token

    def test_state_dict_contains_expected_top_level_keys(self):
        tok = self.make_tokenizer()
        state = tok.state_dict()

        expected_keys = {
            "config",
            "channel_value_to_id",
            "id_to_channel_value",
            "merges",
            "merge_parents",
            "vocab_size",
            "mergeable_payload_token_ids",
        }
        self.assertSetEqual(set(state.keys()), expected_keys)

    def test_roundtrip_empty_tokenizer_preserves_basic_state(self):
        tok = self.make_tokenizer(
            target_vocab_size=777,
            min_pair_count=5,
            random_seed=123,
            time_num_bytes=8,
        )

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        self.assertEqual(restored.config.target_vocab_size, tok.config.target_vocab_size)
        self.assertEqual(restored.config.min_pair_count, tok.config.min_pair_count)
        self.assertEqual(restored.config.random_seed, tok.config.random_seed)
        self.assertEqual(restored.config.time_num_bytes, tok.config.time_num_bytes)

        self.assertEqual(restored.channel_value_to_id, tok.channel_value_to_id)
        self.assertEqual(restored.id_to_channel_value, tok.id_to_channel_value)
        self.assertEqual(restored.merges, tok.merges)
        self.assertEqual(restored.merge_parents, tok.merge_parents)
        self.assertEqual(restored.vocab_size, tok.vocab_size)
        self.assertSetEqual(restored.mergeable_payload_token_ids, tok.mergeable_payload_token_ids)

    def test_roundtrip_preserves_registered_channels(self):
        tok = self.make_tokenizer()
        tok.register_channels([38, 7, 17, 38, 200])

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        self.assertEqual(restored.channel_value_to_id, tok.channel_value_to_id)
        self.assertEqual(restored.id_to_channel_value, tok.id_to_channel_value)

        for ch in [7, 17, 38, 200]:
            self.assertEqual(restored.channel_token_id(ch), tok.channel_token_id(ch))

    def test_roundtrip_preserves_manual_merges_and_merge_parents(self):
        tok = self.make_tokenizer()
        tok.register_channels([38])

        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)
        cc = tok.payload_byte_token_id(0xCC)

        m1 = self.add_manual_merge(tok, aa, bb)
        m2 = self.add_manual_merge(tok, m1, cc)

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        self.assertEqual(restored.merges, tok.merges)
        self.assertEqual(restored.merge_parents, tok.merge_parents)
        self.assertEqual(restored.vocab_size, tok.vocab_size)
        self.assertSetEqual(restored.mergeable_payload_token_ids, tok.mergeable_payload_token_ids)

        self.assertIn(m1, restored.merge_parents)
        self.assertIn(m2, restored.merge_parents)
        self.assertEqual(restored.merge_parents[m1], (aa, bb))
        self.assertEqual(restored.merge_parents[m2], (m1, cc))

    def test_roundtrip_after_fit_preserves_encoding(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            min_pair_count=2,
            time_num_bytes=4,
        )

        records = [
            (0, 37, "AABBCC"),
            (1, 38, "AABBCC"),
            (2, 39, "AABBCC"),
            (3, 17, "AABBCC"),
        ]
        tok.fit(records, verbose=False)

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        test_record = (300, 38, "AABBCC")
        original_encoded = tok.encode(*test_record)
        restored_encoded = restored.encode(*test_record)

        self.assertEqual(restored_encoded, original_encoded)

    def test_roundtrip_after_fit_preserves_decode_packet(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            min_pair_count=2,
            time_num_bytes=4,
        )

        records = [
            (0, 37, "AABBCC"),
            (1, 38, "AABBCC"),
            (2, 39, "AABBCC"),
        ]
        tok.fit(records, verbose=False)

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        encoded = tok.encode(300, 38, "AABBCC")
        decoded = restored.decode_packet(encoded)

        self.assertEqual(decoded["Time Delta"], 300)
        self.assertEqual(decoded["Channel"], 38)
        self.assertEqual(decoded["Hex Data"], "AABBCC")

    def test_roundtrip_preserves_decode_payload_hex_for_recursive_merges(self):
        tok = self.make_tokenizer()
        tok.register_channels([38])

        aa = tok.payload_byte_token_id(0xAA)
        bb = tok.payload_byte_token_id(0xBB)
        cc = tok.payload_byte_token_id(0xCC)

        m1 = self.add_manual_merge(tok, aa, bb)
        m2 = self.add_manual_merge(tok, m1, cc)

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        self.assertEqual(restored.decode_payload_hex([m2]), "AABBCC")

    def test_roundtrip_preserves_token_to_string_behavior(self):
        tok = self.make_tokenizer(time_num_bytes=4)
        tok.register_channels([38])

        aa = tok.payload_byte_token_id(0xAA)
        m1 = self.add_manual_merge(tok, aa, tok.payload_byte_token_id(0xBB))

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        self.assertEqual(restored.token_to_string(restored.time_token_id(0x2C)), "[T_2C]")
        self.assertEqual(restored.token_to_string(restored.channel_token_id(38)), "[CH_38]")
        self.assertEqual(restored.token_to_string(aa), "AA")
        self.assertEqual(restored.token_to_string(m1), f"<MERGE_{m1}>")

    def test_state_dict_roundtrip_preserves_unknown_channel_behavior(self):
        tok = self.make_tokenizer()
        tok.register_channels([37, 38])

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        self.assertEqual(restored.channel_token_id(999), restored.UNK_CHANNEL_ID)

    def test_restored_tokenizer_can_encode_many(self):
        tok = self.make_tokenizer(
            target_vocab_size=600,
            num_workers=1,
            min_pair_count=2,
            time_num_bytes=4,
        )

        records = [
            (0, 37, "AABB"),
            (1, 38, "AABB"),
            (2, 39, "AABB"),
        ]
        tok.fit(records, verbose=False)

        state = tok.state_dict()
        restored = BleBytePairEncoder.from_state_dict(state)

        batch = [
            (5, 37, "AABB"),
            (6, 38, "AABB"),
        ]

        self.assertEqual(restored.encode_many(batch), tok.encode_many(batch))

