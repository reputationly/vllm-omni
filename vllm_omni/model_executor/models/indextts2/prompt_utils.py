# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prompt helpers for IndexTTS2 talker prefill."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from vllm_omni.model_executor.models.indextts2.preprocess_utils import resolve_model_file
from vllm_omni.model_executor.models.indextts2.text_processing_v2_5 import (
    prepare_indextts25_text,
)
from vllm_omni.model_executor.models.indextts2.tokenizer import IndexTTS2Tokenizer
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
)

_V2_CONDITIONING_PREFIX_TOKENS = 34
_V25_CONDITIONING_PREFIX_TOKENS = 3
_V2_TEXT_WRAPPER_TOKENS = 2
_V25_TEXT_WRAPPER_TOKENS = 2
_V25_TEXT_WRAPPER_TOKEN_IDS = frozenset({0, 1})
_START_MEL_TOKENS = 1

# Fallback when the checkpoint config does not expose ``gpt.max_text_tokens``.
# Both the IndexTTS 2 and IndexTTS 2.5 released checkpoints ship 600.
_DEFAULT_MAX_TEXT_TOKENS = 600

# Only needs to outlive the validate()->build() hop of the requests currently
# in flight; stage 0 runs with max_num_seqs=4, so this is deliberately small.
# Keys hold request text, which the request object, the access log and the
# rendered output already retain for at least as long.
_TEXT_TOKEN_COUNT_CACHE_SIZE = 32


def _resolve_bpe_model_path(model_id_or_path: str) -> str:
    path = resolve_model_file(model_id_or_path, "bpe.model")
    if path is None:
        raise FileNotFoundError(f"Could not resolve bpe.model for {model_id_or_path!r}")
    return path


@lru_cache(maxsize=16)
def _get_text_tokenizer(model_id_or_path: str) -> IndexTTS2Tokenizer:
    return IndexTTS2Tokenizer(_resolve_bpe_model_path(model_id_or_path), model_dir=model_id_or_path)


@lru_cache(maxsize=_TEXT_TOKEN_COUNT_CACHE_SIZE)
def _count_indextts2_text_tokens_cached(
    model_id_or_path: str,
    text: str,
    model_type: str,
    lang: str,
    text_normalization: bool,
    tokenizer_file: str,
) -> int:
    """Cache-keyed worker. Takes every argument positionally on purpose.

    ``lru_cache`` keys on the literal call signature, so an omitted keyword and
    an explicitly-passed default are two different entries. Callers here do
    both — ``validate()`` omits the defaults while
    ``estimate_indextts2_prefill_prompt_len`` passes all four — which would
    make every lookup a miss. Funnelling through one fully-positional
    signature is what actually makes the cache hit.
    """
    if model_type == "indextts2_5":
        text_ids, _ = prepare_indextts25_text(
            text,
            lang=lang,
            model_dir=model_id_or_path,
            text_normalization=text_normalization,
            tokenizer_file=tokenizer_file,
        )
        return sum(token_id not in _V25_TEXT_WRAPPER_TOKEN_IDS for token_id in text_ids) + _V25_TEXT_WRAPPER_TOKENS
    tokenizer = _get_text_tokenizer(model_id_or_path)
    return len(tokenizer.encode(text, add_special_tokens=False)) + _V2_TEXT_WRAPPER_TOKENS


def count_indextts2_text_tokens(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
) -> int:
    """Return the text-token sequence length the talker feeds to the GPT.

    This mirrors ``IndexTTS2Talker._tokenize_text`` exactly: the tokenizer
    output with the ``start_text``/``stop_text`` wrapper ids filtered out, then
    re-wrapped in exactly one of each. Callers use it to bound the sequence
    before it reaches ``text_pos_embedding``.

    Results are cached because a single request counts twice: the serving
    adapter's ``validate()`` bounds the sequence, then ``build()`` recomputes
    the same value through ``estimate_indextts2_prefill_prompt_len``. Both run
    synchronously on the event loop, and for IndexTTS 2.5 each call re-runs
    text normalization — measured at ~0.4-1.0 ms for plain Chinese but ~13 ms
    once the text contains numbers, which wetext expands digit by digit.
    """
    return _count_indextts2_text_tokens_cached(
        model_id_or_path,
        text,
        model_type,
        lang,
        text_normalization,
        tokenizer_file,
    )


# Keep the cache controls reachable through the public name so tests and any
# future cache-busting do not have to know about the private worker.
count_indextts2_text_tokens.cache_clear = _count_indextts2_text_tokens_cached.cache_clear
count_indextts2_text_tokens.cache_info = _count_indextts2_text_tokens_cached.cache_info


def resolve_indextts2_text_token_limit(hf_config: Any) -> int:
    """Return the largest text-token sequence the talker can index safely.

    ``IndexTTS2Talker`` builds ``text_pos_embedding`` as
    ``LearnedPositionEmbeddings(max_text_tokens + 2, ...)``, whose ``forward``
    indexes the table with ``arange(seq_len)``. A longer sequence is an
    out-of-bounds embedding lookup, i.e. a CUDA device-side assert that
    poisons the context and takes down the whole engine process instead of
    failing the one request. Callers must reject anything above this.
    """
    max_text_tokens = _DEFAULT_MAX_TEXT_TOKENS
    gpt_cfg = getattr(hf_config, "gpt", None)
    if isinstance(gpt_cfg, Mapping):
        value = gpt_cfg.get("max_text_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            max_text_tokens = value
    return max_text_tokens + 2


def estimate_indextts2_prefill_prompt_len(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
) -> int:
    """Return the placeholder prompt length expected by the IndexTTS2 talker.

    IndexTTS 2 uses 34 conditioning tokens. IndexTTS 2.5 uses one projected
    CAMPPlus speaker token followed by two zero tokens.
    """
    text_token_count = count_indextts2_text_tokens(
        model_id_or_path,
        text,
        model_type=model_type,
        lang=lang,
        text_normalization=text_normalization,
        tokenizer_file=tokenizer_file,
    )
    conditioning_prefix_tokens = (
        _V25_CONDITIONING_PREFIX_TOKENS if model_type == "indextts2_5" else _V2_CONDITIONING_PREFIX_TOKENS
    )
    return conditioning_prefix_tokens + text_token_count + _START_MEL_TOKENS


def build_indextts2_prefill_prompt_ids(
    model_id_or_path: str,
    text: str,
    *,
    model_type: str = "indextts2",
    lang: str = "zh",
    text_normalization: bool = True,
    tokenizer_file: str = INDEXTTS25_TOKENIZER_FILE,
    placeholder_token_id: int = 1,
) -> list[int]:
    prompt_len = estimate_indextts2_prefill_prompt_len(
        model_id_or_path,
        text,
        model_type=model_type,
        lang=lang,
        text_normalization=text_normalization,
        tokenizer_file=tokenizer_file,
    )
    return [placeholder_token_id] * prompt_len
