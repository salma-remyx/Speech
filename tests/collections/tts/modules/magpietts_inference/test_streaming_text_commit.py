# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for causal text commitment (streaming text segmentation) and its wiring
into the MagpieInferenceRunner chunk loop.
"""

import pytest
import torch

from examples.tts.magpietts_inference import create_argument_parser
from nemo.collections.tts.modules.magpietts_inference.inference import MagpieInferenceConfig, MagpieInferenceRunner
from nemo.collections.tts.modules.magpietts_inference.utils import _build_magpie_config
from nemo.collections.tts.parts.utils.streaming_text_commit import (
    StreamingCommitConfig,
    commit_streaming_text,
)


def _commit(text, language="en", **overrides):
    config = StreamingCommitConfig(**overrides) if overrides else None
    return commit_streaming_text(text=text, language=language, config=config)


class TestCausalTextCommitment:
    """Policy tests: the committer only ever sees the prefix it was fed."""

    @pytest.mark.unit
    def test_strong_punctuation_commits_before_stream_ends(self):
        segments, stats = _commit("Hello there friend. How are you today?")
        assert segments == ["Hello there friend.", "How are you today?"]
        # First segment became synthesizable after 3 units, not after the full stream.
        assert stats["units_before_first_commit"] == 3.0
        assert stats["strong_boundary_commits"] == 2.0

    @pytest.mark.unit
    def test_commits_are_prefix_invariant(self):
        """Committed segments must not depend on text that has not arrived yet."""
        prefix = "One two three four. Then five six seven"
        segments_with_future, _ = _commit(prefix + ". UNSEEN FUTURE WORDS")
        segments_without_future, _ = _commit(prefix + ".")
        assert segments_with_future[:2] == segments_without_future[:2]
        assert segments_with_future[0] == "One two three four."

        units_a = _commit(prefix + ".")[1]["units_before_first_commit"]
        units_b = _commit(prefix + ". plus more unseen text")[1]["units_before_first_commit"]
        assert units_a == units_b

    @pytest.mark.unit
    def test_capacity_bounds_segment_size_without_punctuation(self):
        text = " ".join(f"word{i}" for i in range(60))
        segments, stats = _commit(text, max_segment_units=10)
        assert max(len(segment.split()) for segment in segments) <= 10
        assert len(segments) >= 5
        assert stats["forced_capacity_commits"] >= 4

    @pytest.mark.unit
    def test_capacity_cut_prefers_boundary_within_holdback(self):
        text = " ".join(f"word{i}" for i in range(9)) + " tail, next then"
        segments, stats = _commit(text, max_segment_units=10, holdback_units=3)
        assert segments[0].endswith("tail,")
        assert stats["forced_capacity_commits"] == 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "text,expected_segments",
        [
            ("I met Dr. Smith at the station yesterday evening.", 1),  # abbreviation period
            ("The constant is 3. It continues after the dot here.", 1),  # number still forming
            ("The notes (see page three. It helps.) were useful later on.", 2),  # boundary inside brackets
        ],
        ids=["abbreviation", "forming_number", "open_bracket"],
    )
    def test_uncertain_tails_are_kept_provisional(self, text, expected_segments):
        segments, stats = _commit(text)
        assert len(segments) == expected_segments
        assert stats["suppressed_boundaries"] >= 1.0
        # The ambiguous period never became a commit boundary.
        for segment in segments:
            assert not segment.endswith("Dr.") and not segment.endswith("3.")

    @pytest.mark.unit
    def test_weak_boundary_dissolved_by_following_conjunction(self):
        text = " ".join(["word"] * 16) + ", and then we kept on talking more."
        segments, stats = _commit(text)
        # The comma was accepted provisionally, then dissolved by "and": no commit there.
        assert len(segments) == 1
        assert stats["weak_boundary_commits"] == 0.0
        assert stats["provisional_boundaries_dissolved"] == 1.0

    @pytest.mark.unit
    def test_weak_boundary_commits_once_next_unit_resolves_it(self):
        text = " ".join(["word"] * 16) + ", ultimately we finished the task."
        segments, stats = _commit(text)
        assert len(segments) == 2
        assert segments[0].endswith(",")
        assert stats["weak_boundary_commits"] == 1.0

    @pytest.mark.unit
    def test_min_segment_units_suppresses_starved_segments(self):
        segments, _ = _commit("Hi. everyone here agreed on the plan quickly.")
        assert segments == ["Hi. everyone here agreed on the plan quickly."]

    @pytest.mark.unit
    def test_mandarin_character_units(self):
        text = "今天天气很好。我们一起去公园散步聊天喝茶吧。"
        segments, _ = _commit(text, language="zh")
        assert segments == ["今天天气很好。", "我们一起去公园散步聊天喝茶吧。"]


class _StubTokenizer:
    """Minimal stand-in for AggregatedTTSTokenizer (word-count token ids)."""

    tokenizers = {"english_phoneme": object(), "mandarin_phoneme": object()}

    def encode(self, text, tokenizer_name):
        if tokenizer_name == "mandarin_phoneme":
            return list(range(100, 100 + len(text)))
        return list(range(1, len(text.split()) + 1))


class _StubModel:
    """Minimal stand-in for MagpieTTSModel for chunk-rewrite wiring."""

    def __init__(self, eos_id=7):
        self.tokenizer = _StubTokenizer()
        self.eos_id = eos_id


class TestStreamingCommitRunnerWiring:
    """Integration tests for the runner hook and the example CLI wiring."""

    @pytest.mark.unit
    def test_runner_recommits_batch_chunks_causally(self):
        runner = MagpieInferenceRunner(
            model=_StubModel(eos_id=7),
            config=MagpieInferenceConfig(streaming_commit_config=StreamingCommitConfig()),
        )
        batch = {
            "raw_texts": ["First sentence here. Second sentence follows."],
            "languages": ["en"],
            "chunked_tokens": [[torch.tensor([1, 2, 3])]],  # offline dataset chunks
            "chunked_tokens_lens": [[3]],
        }

        stats = runner.apply_streaming_commitment(batch)

        # Two causally committed segments replace the single offline chunk.
        assert len(batch["chunked_tokens"][0]) == 2
        assert batch["chunked_tokens_lens"][0] == [4, 4]
        for tokens, token_len in zip(batch["chunked_tokens"][0], batch["chunked_tokens_lens"][0]):
            assert tokens.dtype == torch.int32
            assert tokens.numel() == token_len
            assert tokens[-1].item() == 7  # EOS appended like an offline chunk
        assert stats["segments"] == 2.0
        assert stats["strong_boundary_commits"] == 2.0
        assert "StreamCommit25" in runner.config.build_identifier()

    @pytest.mark.unit
    def test_runner_pads_ragged_chunk_lists_like_collate(self):
        runner = MagpieInferenceRunner(
            model=_StubModel(eos_id=7),
            config=MagpieInferenceConfig(streaming_commit_config=StreamingCommitConfig()),
        )
        batch = {
            "raw_texts": ["Two short sentences. Committed early.", "Only one sentence here."],
            "languages": ["en", "en"],
            "chunked_tokens": [[torch.tensor([1])], [torch.tensor([2])]],
            "chunked_tokens_lens": [[1], [1]],
        }

        runner.apply_streaming_commitment(batch)

        # Sample 0 commits twice, sample 1 once: sample 1 must be padded so the
        # runner's chunk loop can index [chunk_idx] on every sample.
        assert [len(chunks) for chunks in batch["chunked_tokens"]] == [2, 2]
        padding = batch["chunked_tokens"][1][1]
        assert padding.tolist() == [7]
        assert batch["chunked_tokens_lens"][1] == [5, 1]

    @pytest.mark.unit
    def test_runner_is_noop_without_config(self):
        runner = MagpieInferenceRunner(model=_StubModel(), config=MagpieInferenceConfig())
        offline_chunk = torch.tensor([1, 2, 3])
        batch = {
            "raw_texts": ["First sentence here. Second sentence follows."],
            "languages": ["en"],
            "chunked_tokens": [[offline_chunk]],
            "chunked_tokens_lens": [[3]],
        }

        assert runner.apply_streaming_commitment(batch) == {}
        assert batch["chunked_tokens"] == [[offline_chunk]]
        assert "StreamCommit" not in runner.config.build_identifier()

    @pytest.mark.unit
    def test_runner_requires_raw_texts(self):
        runner = MagpieInferenceRunner(
            model=_StubModel(), config=MagpieInferenceConfig(streaming_commit_config=StreamingCommitConfig())
        )
        with pytest.raises(KeyError, match="raw_texts"):
            runner.apply_streaming_commitment({})

    @pytest.mark.unit
    def test_example_cli_flags_build_streaming_commit_config(self):
        parser = create_argument_parser()
        base_args = ["--codecmodel_path", "codec.nemo", "--datasets_json_path", "evalset.json", "--out_dir", "out"]

        args = parser.parse_args(base_args + ["--streaming_commit", "--streaming_commit_capacity", "12"])
        config = _build_magpie_config(args)
        assert config.streaming_commit_config is not None
        assert config.streaming_commit_config.max_segment_units == 12
        assert "StreamCommit12" in config.build_identifier()

        config_off = _build_magpie_config(parser.parse_args(base_args))
        assert config_off.streaming_commit_config is None
