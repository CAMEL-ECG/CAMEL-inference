"""
Assertions and summaries for ECG language-model wrappers and their LoRA adapters.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Set
import functools
import os
import torch

from ecg_attention_masks import ECGBlockLayout

_ASSERTIONS_ENABLED = os.getenv("ASSERTIONS") == "1"

def _skip_if_assertions_disabled(func):
    """Decorator that no-ops assertion helpers when ASSERTIONS env var is not set."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _ASSERTIONS_ENABLED:
            return None
        return func(*args, **kwargs)

    return wrapper

@_skip_if_assertions_disabled
def assert_tensor_dtype(tensor: torch.Tensor, *, expected: torch.dtype, context: str) -> None:
    """Verify tensor dtype matches expectation."""
    if tensor.dtype != expected:
        raise AssertionError(f"{context}: expected dtype {expected}, got {tensor.dtype}")

@_skip_if_assertions_disabled
def assert_ecg_blocks_consistent(
    *,
    turn_parts: Iterable[Iterable[Dict[str, Any]]],
    ecg_blocks: Iterable[Dict[str, Any]],
) -> None:
    """
    Validate that structured turn parts contain the expected ECG markers per block.
    """
    blocks_list = list(ecg_blocks)
    expected_counts: Dict[int, Dict[str, int]] = {}
    per_lead_special_counts: Dict[int, Dict[str, int]] = {}
    per_lead_secs: Dict[int, Dict[str, Set[int]]] = {}
    global_special_count: Dict[int, int] = {}

    for idx, blk in enumerate(blocks_list):
        leads = [str(ld) for ld in blk.get("lead_names", [])]
        segs = [int(n) for n in blk.get("segments_per_lead", [])]
        expected_counts[idx] = {ld: int(n) for ld, n in zip(leads, segs)}
        per_lead_special_counts[idx] = {ld: 0 for ld in leads}
        per_lead_secs[idx] = {ld: set() for ld in leads}
        global_special_count[idx] = 0

    for turn in turn_parts:
        for part in turn:
            kind = part.get("kind")
            if kind == "text":
                continue
            block_idx = part.get("block_index")
            if block_idx is None or int(block_idx) not in expected_counts:
                raise AssertionError("ECG part references unknown block_index.")
            block_idx = int(block_idx)
            allowed_leads = set(expected_counts[block_idx].keys())
            if kind == "special":
                lead = part.get("lead")
                if lead:
                    if lead not in allowed_leads:
                        raise AssertionError(f"Special token references unknown lead '{lead}'.")
                    per_lead_special_counts[block_idx][lead] = per_lead_special_counts[block_idx].get(lead, 0) + 1
                else:
                    global_special_count[block_idx] = global_special_count.get(block_idx, 0) + 1
                token_literal = part.get("token")
                if not isinstance(token_literal, str) or len(token_literal) == 0:
                    raise AssertionError("Special turn part lacks a string token literal.")
                continue
            if kind == "ecg":
                lead = part.get("lead")
                if lead not in allowed_leads:
                    raise AssertionError(f"ECG segment references unknown lead '{lead}'.")
                sec_val = part.get("sec")
                expected_total = expected_counts[block_idx].get(lead, 0)
                if expected_total <= 0:
                    raise AssertionError(
                        f"Lead '{lead}' has non-positive declared segments ({expected_total}) but ECG markers are present."
                    )
                try:
                    sec = int(sec_val)
                except Exception as exc:  # noqa: BLE001
                    raise AssertionError(f"ECG segment for lead '{lead}' has non-integer sec {sec_val!r}.") from exc
                if sec < 1 or sec > expected_total:
                    raise AssertionError(
                        f"ECG segment for lead '{lead}' has second={sec}, expected within [1,{expected_total}]."
                    )
                if sec in per_lead_secs[block_idx][lead]:
                    raise AssertionError(f"Duplicate ECG segment marker for lead '{lead}' second {sec}.")
                per_lead_secs[block_idx][lead].add(sec)
                continue
            raise AssertionError(f"Unknown turn_parts kind '{kind}'.")

    for block_idx, expected in expected_counts.items():
        if global_special_count.get(block_idx, 0) != 2:
            raise AssertionError(
                f"Expected exactly two global ECG markers for block {block_idx}; "
                f"found {global_special_count.get(block_idx, 0)}."
            )
        for lead, expected_total in expected.items():
            expected_specials = per_lead_special_counts[block_idx].get(lead, 0)
            if expected_specials != 2:
                raise AssertionError(
                    f"Lead '{lead}' has {expected_specials} special markers; expected start and end (2 total)."
                )
            seen_secs = per_lead_secs[block_idx].get(lead, set())
            if expected_total != len(seen_secs):
                missing = sorted(set(range(1, expected_total + 1)) - seen_secs)
                raise AssertionError(
                    f"Lead '{lead}' missing ECG segment markers for seconds {missing} (expected {expected_total})."
                )


# ---------------- Trainer batch validation helpers -----------------------------------------------

@_skip_if_assertions_disabled
def assert_prefix_split_complete(*, offset: int, total_prefix_rows: int) -> None:
    """Validate that prefix splitting consumed all rows."""
    if offset != total_prefix_rows:
        raise RuntimeError(
            f"Prefix split mismatch: consumed {offset} rows but have {total_prefix_rows}"
        )


@_skip_if_assertions_disabled
def assert_prefix_matches_segments(
    *,
    prefix_rows: int,
    segments_per_lead: Iterable[int],
    lead_names: Iterable[str],
    sample_index: int,
    block_index: int,
) -> None:
    """Validate that prefix row count matches sum of segments_per_lead."""
    total_segments = sum(int(n) for n in segments_per_lead)
    if prefix_rows != total_segments:
        raise RuntimeError(
            f"Sample {sample_index} block {block_index}: Prefix rows ({prefix_rows}) "
            f"!= sum(segments_per_lead) ({total_segments}). "
            f"lead_names={list(lead_names)} segments_per_lead={list(segments_per_lead)}"
        )

@_skip_if_assertions_disabled
def assert_ecg_part_bounds(
    *,
    lead: str,
    sec: int,
    lead_to_offset: Mapping[str, int],
    declared_segments: Mapping[str, int],
    total_prefix_rows: int,
    sample_index: int,
    block_index: int,
) -> None:
    """Validate ECG part (lead, sec) falls within expected bounds."""
    if lead not in declared_segments:
        raise RuntimeError(f"Unknown lead {lead} in parts for sample {sample_index} block {block_index}")

    nseg = int(declared_segments[lead])
    if not (1 <= sec <= nseg):
        raise RuntimeError(
            f"sec out of range for lead {lead}: got {sec}, expected 1..{nseg}"
        )

    base = lead_to_offset[lead]
    start = base
    end = base + nseg  # exclusive
    row_idx = start + (sec - 1)

    # Check both global and per-lead bounds
    if not (0 <= row_idx < total_prefix_rows):
        raise RuntimeError(
            f"Bad (lead,sec)=({lead},{sec}) for sample {sample_index} block {block_index}: "
            f"row_idx {row_idx} not in [0,{total_prefix_rows})"
        )
    if not (start <= row_idx < end):
        raise RuntimeError(
            f"(lead,sec)=({lead},{sec}) maps outside this lead block "
            f"[{start},{end}) (row_idx={row_idx}) for sample {sample_index}"
        )


@_skip_if_assertions_disabled
def assert_layout_specials_complete(
    *,
    block_layout: ECGBlockLayout,
    lead_names: Iterable[str],
) -> None:
    """Validate that layout has complete and ordered special token markers.

    For each declared lead:
      - Both start and end must be present (or both absent)
      - If present, start < end

    For global markers:
      - Both start and end must be present (or both absent)
      - If present, start < end
    """
    # Check per-lead specials
    for ld in lead_names:
        s = block_layout.lead_start_idx.get(ld)
        e = block_layout.lead_end_idx.get(ld)
        if (s is None) != (e is None):
            raise RuntimeError(f"Lead {ld} missing start/end special (s={s}, e={e})")
        if s is not None and not (s < e):
            raise RuntimeError(f"Lead {ld} specials out of order: start={s}, end={e}")

    # Check global specials
    if (block_layout.global_start_idx is None) != (block_layout.global_end_idx is None):
        raise RuntimeError("Global start/end special mismatch")
    if block_layout.global_start_idx is not None and block_layout.global_end_idx is not None:
        if not (block_layout.global_start_idx < block_layout.global_end_idx):
            raise RuntimeError(
                f"Global specials out of order: start={block_layout.global_start_idx} "
                f"end={block_layout.global_end_idx}"
            )

# ---------------- Wrapper embedding validations --------------------------------------------------

@_skip_if_assertions_disabled
def assert_wrapper_embed_length(
    *,
    embeddings: torch.Tensor,
    ids: List[int],
    context: str,
) -> None:
    """Ensure embedding sequence length matches token count exactly.

    This is a critical invariant check that ensures the 1:1 mapping between
    input token IDs and output embeddings is preserved.

    Args:
        embeddings: Output embedding tensor (must be at least 1-D)
        ids: Input token ID list
        context: Description of where this check is being performed

    Raises:
        RuntimeError: if ids is not a list, embeddings are not at least 1-D,
                     or if the embedding length doesn't match the token count
    """
    if not isinstance(ids, list):
        raise RuntimeError(f"{context}: ids must be a python list of ints")
    if embeddings.dim() < 1:
        raise RuntimeError(f"{context}: embeddings must be at least 1-D, got shape {tuple(embeddings.shape)}")
    if embeddings.size(0) != len(ids):
        raise RuntimeError(f"{context}: embed length {embeddings.size(0)} != token count {len(ids)}")

@_skip_if_assertions_disabled
def assert_rest_length_nonnegative(*, rest_length: int) -> None:
    """Validate that rest token length is non-negative.

    This should never happen in correct code, but catching it early helps
    identify bugs in label construction logic.

    Args:
        rest_length: Length of ids_rest list

    Raises:
        ValueError: if rest_length is negative
    """
    if rest_length < 0:
        raise ValueError("ids_rest length is negative (internal error).")


# ---------------- Utility assertion helpers ------------------------------------------------------

@_skip_if_assertions_disabled
def assert_sorted_non_overlapping_spans(spans: List[tuple[int, int]], length: int, ctx: str) -> None:
    """Validate that spans are sorted, non-overlapping, and within bounds."""
    prev_end = -1
    for i, (s, e) in enumerate(spans):
        if not (0 <= s <= e <= length):
            raise AssertionError(f"{ctx}: span {i}={(s,e)} out of bounds for length {length}")
        if s < prev_end:
            raise AssertionError(f"{ctx}: spans overlap or not sorted at {i-1},{i}: prev_end={prev_end}, curr_start={s}")
        prev_end = e


@_skip_if_assertions_disabled
def assert_equal_int(a: int, b: int, msg: str) -> None:
    """Assert two integers are equal with a descriptive message."""
    if int(a) != int(b):
        raise AssertionError(f"{msg}: {a} != {b}")


@_skip_if_assertions_disabled
def assert_positive_int(n: int, msg: str) -> None:
    """Assert an integer is positive (> 0)."""
    if int(n) <= 0:
        raise AssertionError(f"{msg}: expected > 0, got {n}")


# ---------------- Schema and catalog validations -------------------------------------------------

@_skip_if_assertions_disabled
def assert_ecg_catalog_valid(catalog: Any, schema: Any) -> None:
    """Validate ECG special token catalog for uniqueness and mapping consistency.

    Checks:
      - All tokens are unique
      - Every canonical lead has entries in lead_to_indices and lead_to_tokens
      - Token-to-index mappings are consistent across all structures
      - Global markers (start/end) are present in the catalog
    """
    # Uniqueness
    if len(set(catalog.tokens)) != len(catalog.tokens):
        raise AssertionError("ECG special tokens contain duplicates")

    # Per-lead mappings
    for lead in schema.ecg.canonical_leads:
        if lead not in catalog.lead_to_indices or lead not in catalog.lead_to_tokens:
            raise AssertionError(f"Missing lead in catalog: {lead}")
        for kind in ("start", "end"):
            tok = catalog.lead_to_tokens[lead][kind]
            idx = catalog.lead_to_indices[lead][kind]
            if catalog.tokens[idx] != tok:
                raise AssertionError(f"Catalog mismatch for {lead}:{kind}: tokens[idx] != tok")
            if catalog.token_to_index.get(tok, None) != idx:
                raise AssertionError(f"token_to_index mismatch for {lead}:{kind}")

    # Global markers
    for tok in (schema.ecg.global_start, schema.ecg.global_end):
        if tok not in catalog.token_to_index:
            raise AssertionError(f"Global ECG token missing from catalog: {tok}")


# ---------------- Conversation and role validations ----------------------------------------------

@_skip_if_assertions_disabled
def assert_normalized_role_canonical(role: str, schema: Any) -> None:
    """Ensure normalized role matches one of the canonical prompt roles."""
    if role not in (schema.prompt.user_role, schema.prompt.model_role):
        raise AssertionError(f"Normalized role '{role}' did not resolve to a canonical prompt role")

# ---------------- Tokenization and span validations ----------------------------------------------

@_skip_if_assertions_disabled
def assert_tokenization_cursor_matches(cursor: int, ids_length: int) -> None:
    """Ensure cursor tracking matches actual text_ids length."""
    if cursor != ids_length:
        raise AssertionError(f"cursor ({cursor}) != len(text_ids) ({ids_length})")


@_skip_if_assertions_disabled
def assert_model_spans_valid(model_spans: List[tuple[int, int]], ids_length: int) -> None:
    """Validate model spans are sorted, non-overlapping, and at least one exists."""
    assert_sorted_non_overlapping_spans(model_spans, ids_length, ctx="model_spans_in_text")
    if len(model_spans) == 0:
        raise AssertionError("No model spans found in text ids")


@_skip_if_assertions_disabled
def assert_eos_appended(ids: List[int], tokenizer: Any, require_eos: bool) -> None:
    """Validate EOS token was appended if required."""
    if require_eos and tokenizer.eos_token_id is not None:
        if not ids or ids[-1] != tokenizer.eos_token_id:
            raise AssertionError("Required EOS was not appended at the end of text_ids")


# ---------------- u0 parts structure validations -------------------------------------------------

@_skip_if_assertions_disabled
def assert_turn_parts_structure_valid(
    parts: List[Dict[str, Any]],
    ecg_blocks: List[Dict[str, Any]],
    schema: Any,
    catalog: Any,
) -> None:
    """Validate the complete structure of turn parts for all blocks present."""
    block_indices = sorted({int(p.get("block_index")) for p in parts if p.get("block_index") is not None})
    for block_idx in block_indices:
        if block_idx < 0 or block_idx >= len(ecg_blocks):
            raise AssertionError(f"Unknown block_index {block_idx} in turn parts")
        blk = ecg_blocks[block_idx]
        leads_present = [str(ld) for ld in blk.get("lead_names", [])]
        segments_per_lead = [int(n) for n in blk.get("segments_per_lead", [])]

        special_tokens = [p.get("token") for p in parts if p.get("kind") == "special" and p.get("block_index") == block_idx]
        if schema.ecg.global_start not in special_tokens:
            raise AssertionError(f"Missing global_start special in block {block_idx}")
        if schema.ecg.global_end not in special_tokens:
            raise AssertionError(f"Missing global_end special in block {block_idx}")

        idx_global_start = next(
            (i for i, p in enumerate(parts)
             if p.get("block_index") == block_idx and p.get("kind") == "special"
             and p.get("token") == schema.ecg.global_start),
            None
        )
        idx_global_end = next(
            (i for i, p in enumerate(parts)
             if p.get("block_index") == block_idx and p.get("kind") == "special"
             and p.get("token") == schema.ecg.global_end),
            None
        )
        if idx_global_start is None or idx_global_end is None or not (idx_global_start < idx_global_end):
            raise AssertionError(f"Block {block_idx}: missing or misordered global start/end specials")

        for lead, nseg in zip(leads_present, segments_per_lead):
            assert_positive_int(nseg, f"segments_per_lead for {lead}")
            idx_start = next(
                (i for i, p in enumerate(parts)
                 if p.get("block_index") == block_idx and p.get("kind") == "special" and p.get("lead") == lead
                 and p.get("token") == catalog.lead_to_tokens[lead]["start"]),
                None
            )
            idx_end = next(
                (i for i, p in enumerate(parts)
                 if p.get("block_index") == block_idx and p.get("kind") == "special" and p.get("lead") == lead
                 and p.get("token") == catalog.lead_to_tokens[lead]["end"]),
                None
            )
            if idx_start is None or idx_end is None or not (idx_start < idx_end):
                raise AssertionError(f"Lead {lead}: missing or misordered start/end specials in block {block_idx}")
            secs = [p["sec"] for p in parts[idx_start+1:idx_end] if p.get("kind") == "ecg" and p.get("lead") == lead]
            if secs != list(range(1, int(nseg) + 1)):
                raise AssertionError(f"Lead {lead}: ECG seconds sequence invalid: {secs} vs 1..{nseg}")
            if parts[idx_start]["token_index"] != catalog.lead_to_indices[lead]["start"]:
                raise AssertionError(f"Lead {lead}: start token_index mismatch")
            if parts[idx_end]["token_index"] != catalog.lead_to_indices[lead]["end"]:
                raise AssertionError(f"Lead {lead}: end token_index mismatch")


@_skip_if_assertions_disabled
def assert_turn_content_ends_with_eot(text_block: str, end_of_turn: str) -> None:
    """Ensure turn content ends with the provided end-of-turn suffix."""
    if not text_block.endswith(end_of_turn):
        raise AssertionError("Turn content must end with end_of_turn suffix")


# ---------------- Per-sample packing validations -------------------------------------------------

@_skip_if_assertions_disabled
def assert_leads_canonical_and_ordered(leads_present: List[str], canonical_leads: tuple) -> None:
    """Validate all leads are canonical (order is explicit and not enforced here)."""
    lead_list = list(leads_present)
    if any(ld not in canonical_leads for ld in lead_list):
        raise AssertionError(f"Non-canonical lead found in leads_present: {lead_list}")
    if len(set(lead_list)) != len(lead_list):
        raise AssertionError(f"Duplicate lead detected in leads_present: {lead_list}")


@_skip_if_assertions_disabled
def assert_waveform_shapes_valid(
    leads_present: List[str],
    segments_per_lead: List[int],
    waveform_segments: Dict[str, Any],
) -> None:
    """Validate waveform tensor shapes and segment counts.

    Each waveform must be [T, 256] where T matches segments_per_lead.
    """
    for ld, nseg in zip(leads_present, segments_per_lead):
        assert_positive_int(nseg, f"segments_per_lead[{ld}]")
        wf = waveform_segments[ld]
        if wf.ndim != 2 or wf.shape[1] != 256:
            raise AssertionError(f"Waveform for {ld} must be [T,256], got {tuple(wf.shape)}")
        assert_equal_int(wf.shape[0], nseg, f"Waveform seconds vs segments_per_lead for {ld}")


__all__ = [
    "assertions_active",
    "capture_adapter_snapshot",
    "assert_wrapper_adapter_requires_grad",
    "assert_wrapper_optimizer_coverage",
    "assert_adapter_gradients",
    "assert_adapter_updates",
    "assert_trainable_param_sync",
    "assert_tensor_dtype",
    "assert_only_llava_proj_trainable",
    "summarize_trainables_llava_lora",
    "assert_language_lora_only",
    "assert_single_bos_eos",
    "assert_ecg_layout_valid",
    "assert_ecg_mask_against_layout",
    "assert_single_block_mask_matches_reference",
    "assert_additive_mask_padding",
    "assert_nonempty_waveform_segments",
    "assert_prefix_split_complete",
    "assert_prefix_matches_segments",
    "assert_ids_are_lists",
    "assert_embedding_length_matches_tokens",
    "assert_ecg_part_bounds",
    "assert_layout_specials_complete",
    "assert_labels_match_spans",
    "assert_wrapper_embed_length",
    "assert_rest_length_nonnegative",
    "assert_sorted_non_overlapping_spans",
    "assert_equal_int",
    "assert_positive_int",
    "assert_ecg_catalog_valid",
    "assert_normalized_role_canonical",
    "assert_rest_blocks_valid",
    "assert_tokenization_cursor_matches",
    "assert_model_spans_valid",
    "assert_turn_parts_consistent",
    "assert_ecg_blocks_consistent",
    "assert_eos_appended",
    "assert_turn_parts_structure_valid",
    "assert_turn_content_ends_with_eot",
    "assert_leads_canonical_and_ordered",
    "assert_waveform_shapes_valid",
    "assert_collate_item_valid",
]