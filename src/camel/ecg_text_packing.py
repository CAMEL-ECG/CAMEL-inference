import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from transformers import PreTrainedTokenizer

from assertions import (
    assert_ecg_catalog_valid,
    assert_normalized_role_canonical,
    assert_turn_parts_structure_valid,
    assert_turn_content_ends_with_eot,
    assert_leads_canonical_and_ordered,
    assert_waveform_shapes_valid,
)
from prompt_renderers import turn_wrappers

# NOTE: Local BOS/EOS assertions are implemented at the bottom of this file.
@dataclass(frozen=True)
class PromptTokens:
    start_of_turn: str
    end_of_turn: str
    user_role: str
    model_role: str
    require_bos: bool = True
    require_eos: bool = True
    allow_multiple_eos: bool = False


@dataclass(frozen=True)
class ConversationRules:
    format_id: str
    user_role_aliases: Tuple[str, ...] = ("human", "user")
    model_role_aliases: Tuple[str, ...] = ("gpt", "assistant")
    strip_image_from_roles: Tuple[str, ...] = ("human",)
    merge_system_with_first_user: bool = True


@dataclass(frozen=True)
class ECGTokenSchema:
    global_start: str
    global_end: str
    lead_start_template: str
    lead_end_template: str
    canonical_leads: Tuple[str, ...]


@dataclass(frozen=True)
class PackingSchema:
    prompt: PromptTokens
    conversation: ConversationRules
    ecg: ECGTokenSchema


@dataclass(frozen=True)
class ECGSpecialTokenCatalog:
    tokens: Tuple[str, ...]
    lead_to_indices: Dict[str, Dict[str, int]]
    lead_to_tokens: Dict[str, Dict[str, str]]
    token_to_index: Dict[str, int]


def _render_lead_template(template: str, lead: str) -> str:
    return template.format(
        lead=lead,
        lead_lower=lead.lower(),
        lead_upper=lead.upper(),
    )

_ECG_TOKEN_CACHE: Dict[PackingSchema, ECGSpecialTokenCatalog] = {}

def get_ecg_special_token_catalog(schema: PackingSchema) -> ECGSpecialTokenCatalog:
    cached = _ECG_TOKEN_CACHE.get(schema)
    if cached is not None:
        return cached

    tokens: List[str] = []
    lead_to_indices: Dict[str, Dict[str, int]] = {}
    lead_to_tokens: Dict[str, Dict[str, str]] = {}

    tokens.append(schema.ecg.global_start)
    tokens.append(schema.ecg.global_end)

    for lead in schema.ecg.canonical_leads:
        start_token = _render_lead_template(schema.ecg.lead_start_template, lead)
        end_token = _render_lead_template(schema.ecg.lead_end_template, lead)
        start_idx = len(tokens)
        tokens.append(start_token)
        end_idx = len(tokens)
        tokens.append(end_token)
        lead_to_indices[lead] = {"start": start_idx, "end": end_idx}
        lead_to_tokens[lead] = {"start": start_token, "end": end_token}

    catalog = ECGSpecialTokenCatalog(
        tokens=tuple(tokens),
        lead_to_indices=lead_to_indices,
        lead_to_tokens=lead_to_tokens,
        token_to_index={tok: idx for idx, tok in enumerate(tokens)},
    )

    # Validate catalog consistency
    assert_ecg_catalog_valid(catalog, schema)

    _ECG_TOKEN_CACHE[schema] = catalog
    return catalog


# ---- Conversation normalization + validation ----------------------------------------------------

def canonical_leads(schema: PackingSchema) -> List[str]:
    return list(schema.ecg.canonical_leads)


def _strip_image_tag(text: str) -> str:
    """
    Remove any <image> placeholder without gluing surrounding words.
    Replace the token and surrounding whitespace with a single space and
    normalize repeated spaces.
    """
    cleaned = re.sub(r"\s*<image>\s*", " ", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _normalize_role(role_value: Any, schema: PackingSchema) -> str:
    role = (role_value or "").strip()
    if not role:
        raise ValueError("Conversation turn is missing a role identifier.")
    if role.lower() == "system":
        return "system"
    if role.lower() == "developer":
        return "developer"
    needles = [schema.prompt.user_role] + list(schema.conversation.user_role_aliases)
    if role.lower() in (needle.lower() for needle in needles):
        out = schema.prompt.user_role
        assert_normalized_role_canonical(out, schema)
        return out
    needles = [schema.prompt.model_role] + list(schema.conversation.model_role_aliases)
    if role.lower() in (needle.lower() for needle in needles):
        out = schema.prompt.model_role
        assert_normalized_role_canonical(out, schema)
        return out
    raise ValueError(f"Unknown conversation role '{role_value}' for schema '{schema.conversation.format_id}'.")


def _maybe_strip_content(text: str, canonical_role: str, schema: PackingSchema) -> str:
    target_roles: set = set()
    for r in schema.conversation.strip_image_from_roles:
        try:
            target_roles.add(_normalize_role(r, schema).lower())
        except ValueError:
            target_roles.add(str(r).lower())
    if canonical_role.lower() in target_roles:
        return _strip_image_tag(text)
    return text

# ---- Tokenize & mark assistant spans (exclude control tokens from loss) --------------------------

# ---- Build structured turn parts ---------------------------------------------------------------

def build_structured_turn_parts(
    *,
    content: List[Dict[str, Any]],
    canonical_role: str,
    schema: PackingSchema,
    ecg_blocks: List[Dict[str, Any]],
    sampling_rate: Optional[float],
    turn_suffix: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    prompt_tokens = schema.prompt
    catalog = get_ecg_special_token_catalog(schema)
    parts: List[Dict[str, Any]] = []
    text_segments: List[str] = []

    def _append_text(txt: str) -> None:
        if not txt:
            return
        if parts and parts[-1].get("kind") == "text":
            parts[-1]["text"] += txt
        else:
            parts.append({"kind": "text", "text": txt})
        text_segments.append(txt)

    for item in content:
        if not isinstance(item, dict):
            raise ValueError("Conversation content items must be dicts.")
        item_type = item.get("type")
        if item_type == "text":
            text_val = item.get("text", "")
            if not isinstance(text_val, str):
                raise ValueError("Text content item must have a string 'text' field.")
            cleaned = _maybe_strip_content(text_val, canonical_role, schema)
            _append_text(cleaned)
            continue
        if item_type == "ecg":
            waveform_segments = item.get("waveform_segments")
            if not isinstance(waveform_segments, dict):
                raise ValueError("ECG content item missing waveform_segments mapping.")
            item_rate = item.get("sampling_rate")
            if item_rate is not None and sampling_rate is not None and float(item_rate) != float(sampling_rate):
                raise ValueError(
                    f"ECG item sampling_rate {item_rate} does not match sample sampling_rate {sampling_rate}."
                )
            lead_names = [str(ld) for ld in waveform_segments.keys()]
            if not lead_names:
                raise ValueError("ECG content item has no leads.")
            segments_per_lead = [int(waveform_segments[ld].shape[0]) for ld in lead_names]
            assert_leads_canonical_and_ordered(lead_names, schema.ecg.canonical_leads)
            assert_waveform_shapes_valid(lead_names, segments_per_lead, waveform_segments)
            block_index = len(ecg_blocks)
            ecg_blocks.append({
                "lead_names": lead_names,
                "segments_per_lead": segments_per_lead,
                "waveform_segments": OrderedDict((ld, waveform_segments[ld]) for ld in lead_names),
            })

            parts.append({
                "kind": "special",
                "token": schema.ecg.global_start,
                "token_index": catalog.token_to_index[schema.ecg.global_start],
                "block_index": block_index,
            })
            text_segments.append(schema.ecg.global_start)

            for lead, nseg in zip(lead_names, segments_per_lead):
                lead_tokens = catalog.lead_to_indices[lead]
                parts.append({
                    "kind": "special",
                    "token": catalog.lead_to_tokens[lead]["start"],
                    "token_index": lead_tokens["start"],
                    "lead": lead,
                    "block_index": block_index,
                })
                text_segments.append(catalog.lead_to_tokens[lead]["start"])
                for sec in range(1, int(nseg) + 1):
                    parts.append({
                        "kind": "ecg",
                        "lead": lead,
                        "sec": sec,
                        "block_index": block_index,
                    })
                parts.append({
                    "kind": "special",
                    "token": catalog.lead_to_tokens[lead]["end"],
                    "token_index": lead_tokens["end"],
                    "lead": lead,
                    "block_index": block_index,
                })
                text_segments.append(catalog.lead_to_tokens[lead]["end"])

            parts.append({
                "kind": "special",
                "token": schema.ecg.global_end,
                "token_index": catalog.token_to_index[schema.ecg.global_end],
                "block_index": block_index,
            })
            text_segments.append(schema.ecg.global_end)
            continue
        raise ValueError(f"Unknown content item type '{item_type}'.")

    turn_content = "".join(text_segments)
    if turn_suffix is None:
        _, suffix = turn_wrappers(schema, canonical_role)
    else:
        suffix = turn_suffix
    turn_text_block = turn_content + suffix

    assert_turn_parts_structure_valid(parts, ecg_blocks, schema, catalog)
    assert_turn_content_ends_with_eot(turn_text_block, suffix)

    return turn_text_block, parts


def build_text_only_turn_parts(
    *,
    content: List[Dict[str, Any]],
    canonical_role: str,
    schema: PackingSchema,
    turn_suffix: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    prompt_tokens = schema.prompt
    parts: List[Dict[str, Any]] = []
    text_segments: List[str] = []
    needs_channel_header = (
        schema.conversation.format_id == "harmony_chat_v1"
        and canonical_role == schema.prompt.model_role
    )
    # Track raw content for diagnostics
    raw_content_debug: List[Dict[str, Any]] = []

    def _append_text(txt: str) -> None:
        if not txt:
            return
        if parts and parts[-1].get("kind") == "text":
            parts[-1]["text"] += txt
        else:
            parts.append({"kind": "text", "text": txt})
        text_segments.append(txt)

    for item in content:
        if not isinstance(item, dict):
            raise ValueError("Conversation content items must be dicts.")
        item_type = item.get("type")
        if item_type != "text":
            raise ValueError("Model turns cannot contain ECG content items.")
        text_val = item.get("text", "")
        if not isinstance(text_val, str):
            raise ValueError("Text content item must have a string 'text' field.")
        if needs_channel_header and text_val.lstrip().startswith("<|channel|>"):
            raise ValueError("Assistant content must not include harmony channel headers.")
        cleaned = _maybe_strip_content(text_val, canonical_role, schema)
        raw_content_debug.append({
            "raw_text": repr(text_val[:200]) + ("..." if len(text_val) > 200 else ""),
            "raw_len": len(text_val),
            "cleaned_text": repr(cleaned[:200]) + ("..." if len(cleaned) > 200 else ""),
            "cleaned_len": len(cleaned),
        })
        _append_text(cleaned)

    turn_content = "".join(text_segments)
    if not turn_content:
        # Build detailed diagnostic message
        diag_lines = [
            "Model turn is empty after preprocessing.",
            f"  role: {canonical_role}",
            f"  num_content_items: {len(content)}",
        ]
        for i, dbg in enumerate(raw_content_debug):
            diag_lines.append(f"  item[{i}]: raw_len={dbg['raw_len']}, cleaned_len={dbg['cleaned_len']}")
            diag_lines.append(f"    raw: {dbg['raw_text']}")
            diag_lines.append(f"    cleaned: {dbg['cleaned_text']}")
        raise ValueError("\n".join(diag_lines))
    if needs_channel_header:
        channel_header = "<|channel|>final<|message|>"
        parts.insert(0, {"kind": "text", "text": channel_header})
        text_segments.insert(0, channel_header)
        turn_content = "".join(text_segments)
    if turn_suffix is None:
        _, suffix = turn_wrappers(schema, canonical_role)
    else:
        suffix = turn_suffix
    turn_text_block = turn_content + suffix
    assert_turn_content_ends_with_eot(turn_text_block, suffix)
    return turn_text_block, parts


def annotate_turn_parts_with_ids(
    turn_parts: List[List[Dict[str, Any]]],
    tokenizer: PreTrainedTokenizer,
) -> List[List[Dict[str, Any]]]:
    """Attach token ids to text parts so the trainer can skip per-step tokenization."""
    for parts in turn_parts:
        for part in parts:
            if part.get("kind") == "text":
                txt = part.get("text", "")
                part["ids"] = tokenizer.encode(txt, add_special_tokens=False) if txt else []
    return turn_parts

def assert_single_bos_eos(
    text_ids: List[int],
    tok: PreTrainedTokenizer,
    *,
    require_bos_at_start: bool,
    require_single_terminal_eos: bool,
    allow_multiple_eos: bool = False,
) -> None:
    """Validate BOS/EOS placement under various schema-specific policies."""
    if require_bos_at_start:
        if tok.bos_token_id is None:
            raise AssertionError("BOS token required but tokenizer has none")
        bos_pos = [i for i, t in enumerate(text_ids) if t == tok.bos_token_id]
        if len(bos_pos) != 1 or bos_pos[0] != 0:
            raise AssertionError(f"BOS placement invalid: positions={bos_pos}")
    else:
        if tok.bos_token_id is not None:
            bos_pos = [i for i, t in enumerate(text_ids) if t == tok.bos_token_id]
            if len(bos_pos) > 1 or (bos_pos and bos_pos[0] != 0):
                raise AssertionError(f"Unexpected BOS placement: positions={bos_pos}")

    # EOS checks
    if tok.eos_token_id is not None:
        eos_pos = [i for i, t in enumerate(text_ids) if t == tok.eos_token_id]
        if allow_multiple_eos:
            if require_single_terminal_eos:
                # Allow multiple EOS (e.g., ChatML-style per turn), but require the last EOS at sequence end.
                if len(eos_pos) == 0 or eos_pos[-1] != (len(text_ids) - 1):
                    raise AssertionError(f"EOS bad: positions={eos_pos}")
            else:
                # Allow any count anywhere; no terminal EOS required (matches Qwen3 training practice).
                pass
        else:
            if require_single_terminal_eos:
                # Require exactly one EOS, and it must be terminal.
                if len(eos_pos) != 1 or eos_pos[0] != (len(text_ids) - 1):
                    raise AssertionError(f"EOS bad: positions={eos_pos}")
            else:
                # Allow 0 or 1, but if present it must be terminal.
                if len(eos_pos) > 1 or (eos_pos and eos_pos[0] != (len(text_ids) - 1)):
                    raise AssertionError(f"EOS bad: positions={eos_pos}")


def assert_struct_bos_eos(
    token_struct: Dict[str, List[int]],
    tok: PreTrainedTokenizer,
    *,
    require_bos_at_start: bool,
    require_single_terminal_eos: bool,
    allow_multiple_eos: bool = False,
) -> None:
    """Wrapper that validates concatenated ids per schema policy."""
    text_ids = token_struct["text_ids"]
    assert_single_bos_eos(
        text_ids,
        tok,
        require_bos_at_start=require_bos_at_start,
        require_single_terminal_eos=require_single_terminal_eos,
        allow_multiple_eos=allow_multiple_eos,
    )
