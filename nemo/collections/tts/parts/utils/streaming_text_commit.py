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
Causal text commitment for streaming TTS inference.

Segmentation policy for text that arrives incrementally (e.g. from a streaming
LLM): decides, using only the tokens observed so far, when enough text has
arrived to commit a segment for synthesis. This replaces the offline
"sentence-split the full text up front" behaviour of
``chunk_text_for_inference`` (which requires the complete utterance and is
therefore only pseudo-streaming) with an online policy built from three rules:

- **Punctuation-aware commitment**: a segment is committed at sentence-ending
  punctuation as soon as it arrives, so time-to-first-audio tracks the first
  sentence instead of the whole utterance.
- **Uncertainty-aware buffering**: ambiguous tails are kept provisional. A
  period that probably does not end a sentence (abbreviations such as ``Dr.``,
  numbers still forming such as ``3.``), trailing conjunctions, hyphenated
  compounds in progress, and unclosed brackets all hold the boundary open.
- **Capacity-adaptive segmentation**: when the buffer reaches capacity the
  committer cuts at the best recent boundary (if one lies within the holdback
  window) and otherwise force-commits, bounding latency and context size.

Adapted from the *causal commitment* mechanism of "X2Streaming-TTS: Causal
Token-Level Text-to-Speech from Streaming Text with Speech-State Inheritance"
(arXiv:2608.18661). The paper estimates prefix uncertainty with a language
model; this implementation substitutes parameter-free lexical proxies (Section
helpers at the bottom of this file). Acoustic continuity across committed
segments is provided by the caller's ``ChunkState`` (speech-state inheritance),
which already carries text/encoder/attention history between chunks.
"""

import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import torch

from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    get_tokenizer_for_language,
    has_phoneme_text_spans,
    tokenize_text_with_phoneme_spans,
)


@dataclass
class StreamingCommitConfig:
    """Parameters of the causal text commitment policy.

    All size parameters are in *arrival units*: whitespace-separated words for
    most languages, single characters for zh/ja (see ``character_unit_scale``).

    Attributes:
        max_segment_units: Capacity of the provisional buffer. Reaching it
            forces a commit (at the best nearby boundary if one exists).
        min_segment_units: Segments below this size are not committed at
            punctuation, to avoid starving the synthesizer of context.
        weak_boundary_min_units: Buffer size from which a comma/semicolon is
            accepted as a commit boundary once it survives one lookahead unit.
        holdback_units: When cutting at capacity, boundaries further than this
            many units from the tail are ignored (they would leave a nearly
            empty remainder).
        strong_punctuation: Characters that end a segment immediately.
        weak_punctuation: Characters that end a segment provisionally.
        character_unit_scale: Multiplier applied to the size parameters for
            character-unit languages (zh, ja), where one unit is one character.
    """

    max_segment_units: int = 25
    min_segment_units: int = 3
    weak_boundary_min_units: int = 16
    holdback_units: int = 4
    strong_punctuation: Tuple[str, ...] = (".", "!", "?", "…", "。", "！", "？", "।", "॥")
    weak_punctuation: Tuple[str, ...] = (",", ";", ":", "，", "、", "；", "：")
    character_unit_scale: float = 2.0


class CausalTextCommitter:
    """Online segmenter over an asynchronously arriving text stream.

    Feed arrival units (words, or characters for zh/ja) with :meth:`feed`; each
    call returns the text of a committed segment, or ``None`` while the buffer
    is still provisional. The commit decision depends only on units already
    fed, never on future input. Call :meth:`flush` at end of stream.
    """

    def __init__(self, config: Optional[StreamingCommitConfig] = None, separator: str = " "):
        """Initialize the committer.

        Args:
            config: Policy parameters; defaults to ``StreamingCommitConfig()``.
            separator: String used to join committed units (space for word
                streams, empty for character streams).
        """
        self.config = config if config is not None else StreamingCommitConfig()
        self._separator = separator
        self._buffer: List[str] = []
        self._pending_weak_idx: Optional[int] = None
        self._segments = 0
        self._strong_commits = 0
        self._weak_commits = 0
        self._capacity_commits = 0
        self._forced_commits = 0
        self._suppressed_boundaries = 0
        self._dissolved_boundaries = 0
        self._units_seen = 0
        self._units_at_first_commit: Optional[int] = None
        self._segment_units: List[int] = []

    def feed(self, unit: str) -> Optional[str]:
        """Consume one arriving text unit and return a committed segment if any.

        Args:
            unit: The newly arrived word or character.

        Returns:
            Committed segment text, or ``None`` if nothing was committed.
        """
        self._units_seen += 1
        self._buffer.append(unit)

        # A weak boundary from the previous unit stays provisional until the
        # next unit shows whether the clause continues (e.g. "came, and ...").
        if self._pending_weak_idx is not None:
            if _continues_clause(unit):
                self._pending_weak_idx = None
                self._dissolved_boundaries += 1
            else:
                return self._commit_upto(self._pending_weak_idx, reason="weak")

        # Capacity: bound the buffer by cutting at the best nearby boundary.
        if len(self._buffer) >= self.config.max_segment_units:
            cut = self._best_boundary_in_holdback()
            if cut is None:
                self._forced_commits += 1
            return self._commit_upto(cut if cut is not None else len(self._buffer) - 1, reason="capacity")

        tail = self._buffer[-1]
        strength = _boundary_strength(tail, self.config)
        if strength and not _is_lexically_ambiguous(tail) and not _has_open_bracket("".join(self._buffer)):
            if strength == "strong" and len(self._buffer) >= self.config.min_segment_units:
                return self._commit_upto(len(self._buffer) - 1, reason="strong")
            if strength == "weak" and len(self._buffer) >= self.config.weak_boundary_min_units:
                self._pending_weak_idx = len(self._buffer) - 1
        elif strength:
            self._suppressed_boundaries += 1
        return None

    def flush(self) -> List[str]:
        """Commit whatever remains at end of stream.

        Returns:
            List with a single final segment (possibly empty if all text was
            already committed).
        """
        self._pending_weak_idx = None  # no latency left to save at end of stream
        if not self._buffer:
            return []
        return [self._commit_upto(len(self._buffer) - 1, reason="eos")]

    @property
    def stats(self) -> Dict[str, float]:
        """Commit statistics for this stream.

        Returns:
            Dict with segment counts per commit reason, the number of
            punctuation boundaries kept provisional by the uncertainty and
            minimum-size guards, and ``units_before_first_commit`` (a
            text-level time-to-first-audio proxy: how many units had arrived
            when the first segment became synthesizable).
        """
        return {
            "segments": float(self._segments),
            "strong_boundary_commits": float(self._strong_commits),
            "weak_boundary_commits": float(self._weak_commits),
            "capacity_commits": float(self._capacity_commits),
            "forced_capacity_commits": float(self._forced_commits),
            "suppressed_boundaries": float(self._suppressed_boundaries),
            "provisional_boundaries_dissolved": float(self._dissolved_boundaries),
            "units_total": float(self._units_seen),
            "units_before_first_commit": float(
                self._units_at_first_commit if self._units_at_first_commit is not None else self._units_seen
            ),
            "mean_segment_units": (float(sum(self._segment_units) / len(self._segment_units)))
            if self._segment_units
            else 0.0,
        }

    def _commit_upto(self, idx: int, reason: str) -> str:
        """Split the buffer at ``idx`` and return the committed prefix text."""
        segment_units = self._buffer[: idx + 1]
        self._buffer = self._buffer[idx + 1 :]
        self._pending_weak_idx = None
        self._segments += 1
        if reason == "strong":
            self._strong_commits += 1
        elif reason == "weak":
            self._weak_commits += 1
        elif reason == "capacity":
            self._capacity_commits += 1
        self._segment_units.append(len(segment_units))
        if self._units_at_first_commit is None:
            self._units_at_first_commit = self._units_seen
        return self._separator.join(segment_units)

    def _best_boundary_in_holdback(self) -> Optional[int]:
        """Index of the best commit boundary within the holdback window, if any."""
        strongest: Optional[int] = None
        weakest: Optional[int] = None
        window_start = max(self.config.min_segment_units - 1, len(self._buffer) - 1 - self.config.holdback_units)
        for idx in range(len(self._buffer) - 1, window_start - 1, -1):
            unit = self._buffer[idx]
            if _is_lexically_ambiguous(unit):
                continue
            strength = _boundary_strength(unit, self.config)
            if strength == "strong" and strongest is None:
                strongest = idx
            elif strength == "weak" and weakest is None:
                weakest = idx
        if strongest is not None:
            return strongest
        return weakest


def commit_streaming_text(
    text: str,
    language: str = "en",
    config: Optional[StreamingCommitConfig] = None,
) -> Tuple[List[str], Dict[str, float]]:
    """Segment text as if it arrived unit by unit, committing causally.

    Args:
        text: Full text of the utterance (used only as the arrival stream; the
            policy never looks past the units already fed).
        language: Language code, selecting word- vs character-level arrival
            units ("zh"/"ja" are character-based).
        config: Policy parameters; defaults to ``StreamingCommitConfig()``.

    Returns:
        Tuple of (committed segments, commit statistics).
    """
    if config is None:
        config = StreamingCommitConfig()
    if language in ("zh", "ja"):
        config = replace(
            config,
            max_segment_units=max(1, int(config.max_segment_units * config.character_unit_scale)),
            min_segment_units=max(1, int(config.min_segment_units * config.character_unit_scale)),
            weak_boundary_min_units=max(1, int(config.weak_boundary_min_units * config.character_unit_scale)),
        )
        units, separator = [ch for ch in text if not ch.isspace()], ""
    else:
        units, separator = text.split(), " "

    committer = CausalTextCommitter(config=config, separator=separator)
    segments: List[str] = []
    for unit in units:
        committed = committer.feed(unit)
        if committed is not None and committed.strip():
            segments.append(committed)
    segments.extend(segment for segment in committer.flush() if segment.strip())
    return segments, committer.stats


def apply_streaming_commitment_to_batch(
    batch: Dict[str, Any],
    text_tokenizer: Any,
    eos_token_id: int,
    config: Optional[StreamingCommitConfig] = None,
    enable_phoneme_text_input: bool = False,
    phoneme_tokenizer: Any = None,
    text_phoneme_token_offset: Optional[int] = None,
    bop_marker: str = "<bop>",
    eop_marker: str = "<eop>",
) -> Dict[str, float]:
    """Re-derive ``chunked_tokens`` of an inference batch by causal commitment.

    Rewrites ``batch['chunked_tokens']`` and ``batch['chunked_tokens_lens']``
    from ``batch['raw_texts']`` so that the downstream chunk loop synthesizes
    causally committed segments instead of the dataset's offline sentence
    chunks. Each committed segment is tokenized exactly like an offline chunk
    (named tokenizer + trailing EOS), so state carried across chunks by
    ``ChunkState`` is inherited unchanged.

    Text containing phoneme-span markers is kept as a single chunk, mirroring
    the offline rule that span boundaries must not be split.

    Args:
        batch: Inference batch dictionary; must contain ``raw_texts`` and is
            modified in place.
        text_tokenizer: Aggregated tokenizer providing ``tokenizers`` and
            ``encode(text, tokenizer_name)``.
        eos_token_id: EOS token appended to every committed segment.
        config: Policy parameters; defaults to ``StreamingCommitConfig()``.
        enable_phoneme_text_input: Whether inline phoneme spans are enabled.
        phoneme_tokenizer: Tokenizer for phoneme spans (as in the dataset).
        text_phoneme_token_offset: Phoneme token ID offset (as in the dataset).
        bop_marker: Marker opening an inline phoneme span.
        eop_marker: Marker closing an inline phoneme span.

    Returns:
        Dict of commit statistics averaged over the batch.

    Raises:
        KeyError: If ``batch`` has no ``raw_texts`` entries to commit.
    """
    if "raw_texts" not in batch:
        raise KeyError("Causal text commitment requires batch['raw_texts'] (produced by the inference dataset).")
    if config is None:
        config = StreamingCommitConfig()

    available_tokenizers = list(getattr(text_tokenizer, "tokenizers", {}).keys())
    languages = batch.get("languages") or ["en"] * len(batch["raw_texts"])

    chunked_tokens: List[List[torch.Tensor]] = []
    chunked_tokens_lens: List[List[int]] = []
    per_sample_stats: List[Dict[str, float]] = []
    for text, language in zip(batch["raw_texts"], languages):
        if has_phoneme_text_spans(text, bop_marker=bop_marker, eop_marker=eop_marker):
            segments, stats = [text], {"segments": 1.0}
        else:
            segments, stats = commit_streaming_text(text=text, language=language, config=config)

        sample_tokens: List[torch.Tensor] = []
        sample_lens: List[int] = []
        for segment in segments:
            tokens = tokenize_text_with_phoneme_spans(
                text_tokenizer=text_tokenizer,
                text_str=segment,
                tokenizer_name=get_tokenizer_for_language(language, available_tokenizers),
                enable_phoneme_text_input=enable_phoneme_text_input,
                phoneme_tokenizer=phoneme_tokenizer,
                text_phoneme_token_offset=text_phoneme_token_offset,
                bop_marker=bop_marker,
                eop_marker=eop_marker,
            )
            tokens = tokens + [eos_token_id]
            sample_tokens.append(torch.tensor(tokens, dtype=torch.int32))
            sample_lens.append(len(tokens))

        if not sample_tokens:  # empty text: mirror the dataset's single-EOS chunk
            sample_tokens = [torch.tensor([eos_token_id], dtype=torch.int32)]
            sample_lens = [1]

        chunked_tokens.append(sample_tokens)
        chunked_tokens_lens.append(sample_lens)
        per_sample_stats.append(stats)

    batch["chunked_tokens"] = chunked_tokens
    batch["chunked_tokens_lens"] = chunked_tokens_lens

    # Pad ragged chunk lists to the batch maximum, exactly like the dataset's
    # collate_fn, so the runner's chunk loop can index every sample.
    max_num_chunks = max((len(sample_tokens) for sample_tokens in chunked_tokens), default=0)
    for sample_tokens, sample_lens in zip(chunked_tokens, chunked_tokens_lens):
        num_padding = max_num_chunks - len(sample_tokens)
        sample_tokens.extend(torch.tensor([eos_token_id], dtype=torch.int32) for _ in range(num_padding))
        sample_lens.extend([1] * num_padding)

    num_samples = max(len(per_sample_stats), 1)
    aggregate = {
        key: float(sum(stats.get(key, 0.0) for stats in per_sample_stats) / num_samples)
        for key in (
            "segments",
            "strong_boundary_commits",
            "weak_boundary_commits",
            "forced_capacity_commits",
            "suppressed_boundaries",
            "provisional_boundaries_dissolved",
            "units_before_first_commit",
        )
    }
    aggregate["num_samples"] = float(len(per_sample_stats))
    return aggregate


# Private helpers: parameter-free proxies for the paper's uncertainty estimate.


def _boundary_strength(unit: str, config: StreamingCommitConfig) -> str:
    """Classify a unit as ending with "strong", "weak", or no commit boundary."""
    core = _strip_trailing_closers(unit)
    if not core:
        return ""
    if core[-1] in config.strong_punctuation:
        return "strong"
    if core[-1] in config.weak_punctuation:
        return "weak"
    return ""


def _is_lexically_ambiguous(unit: str) -> bool:
    """Whether a unit's trailing punctuation probably does not end a clause.

    Proxies for prefix uncertainty: abbreviations and initials ("Dr.", "J."),
    numbers that may still be forming ("3." of "3.14", "1," of "1,000"), and
    hyphenated compounds in progress ("twenty-").
    """
    core = _strip_trailing_closers(unit).lower()
    if not core:
        return False
    if core.endswith("-"):
        return True
    if _FORMING_NUMBER_RE.search(core):
        return True
    if _INITIAL_RE.fullmatch(core):
        return True
    return core.endswith(".") and core.rstrip(".") in _ABBREVIATIONS


def _continues_clause(unit: str) -> bool:
    """Whether a newly arrived unit signals that the clause is not finished."""
    if unit and unit[0] in _OPEN_BRACKETS:
        return True
    return _strip_trailing_closers(unit).lower().strip("".join(_CLOSERS)) in _HOLDBACK_WORDS


def _strip_trailing_closers(unit: str) -> str:
    """Drop trailing closing quotes/brackets so punctuation beneath them is visible."""
    return unit.rstrip("".join(_CLOSERS))


def _has_open_bracket(text: str) -> bool:
    """Whether the buffered text has an unclosed bracket or quote."""
    depth = 0
    for char in text:
        if char in _OPENING_BRACKETS:
            depth += 1
        elif char in _CLOSING_BRACKETS:
            depth = max(depth - 1, 0)
    return depth > 0 or text.count('"') % 2 == 1


_OPENING_BRACKETS = frozenset("([{“‘«「『")
_CLOSING_BRACKETS = frozenset(")]}”’»」』")
_OPEN_BRACKETS = _OPENING_BRACKETS | frozenset('"')
_CLOSERS = _CLOSING_BRACKETS | frozenset('"')

_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st vs etc fig eq approx inc ltd dept univ vol ed al no".split()
)
_HOLDBACK_WORDS = frozenset(
    """
    and or but nor so yet for because although though while when since that which who whom whose
    where if then than as also however therefore moreover meanwhile instead
    """.split()
)
_FORMING_NUMBER_RE = re.compile(r"\d[.,]$")
_INITIAL_RE = re.compile(r"[a-z]\.")
