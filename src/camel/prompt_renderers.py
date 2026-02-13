"""Prompt rendering and span construction helpers."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple
from transformers import PreTrainedTokenizer

from camel.assertions import (
    assert_tokenization_cursor_matches,
    assert_model_spans_valid,
    assert_eos_appended,
)

def _ensure_trailing_newline(s: str) -> str:
    if s.endswith("\n"):
        return s
    return s + "\n"

def _chat_v1_wrappers(schema, role: str) -> Tuple[str, str]:
    prefix = f"{schema.prompt.start_of_turn}{role}\n"
    suffix = _ensure_trailing_newline(schema.prompt.end_of_turn)
    return prefix, suffix

def _harmony_v1_wrappers(schema, role: str, *, use_return: bool = False) -> Tuple[str, str]:
    if role == schema.prompt.model_role:
        prefix = f"{schema.prompt.start_of_turn}{role}"
        suffix = "<|return|>" if use_return else str(schema.prompt.end_of_turn)
        return prefix, suffix
    prefix = f"{schema.prompt.start_of_turn}{role}<|message|>"
    suffix = str(schema.prompt.end_of_turn)
    return prefix, suffix

def _render_with_wrappers(
    tokenizer: PreTrainedTokenizer,
    turns: List[Dict[str, str]],
    *,
    schema,
    wrapper_fn,
) -> Dict[str, Any]:
    tok = tokenizer
    prompt_tokens = schema.prompt

    text_ids: List[int] = []
    if prompt_tokens.require_bos and tok.bos_token_id is not None:
        text_ids.append(tok.bos_token_id)

    model_spans_in_text: List[Tuple[int, int]] = []
    cursor = len(text_ids)
    text_preview_parts: List[str] = []

    for turn in turns:
        role = turn["role"]
        text_block = turn["text_block"]
        prefix, suffix = wrapper_fn(schema, role)
        content = text_block
        if suffix and content.endswith(suffix):
            content = content[: -len(suffix)]
        ids_prefix = tok.encode(prefix, add_special_tokens=False)
        ids_content = tok.encode(content, add_special_tokens=False)
        ids_suffix = tok.encode(suffix, add_special_tokens=False)

        text_ids.extend(ids_prefix)
        text_ids.extend(ids_content)
        text_ids.extend(ids_suffix)

        if role == prompt_tokens.model_role:
            s = cursor + len(ids_prefix)
            e = s + len(ids_content) + len(ids_suffix)
            if e > s:
                model_spans_in_text.append((s, e))
        cursor += len(ids_prefix) + len(ids_content) + len(ids_suffix)
        text_preview_parts.append(prefix + content + suffix)

    assert_tokenization_cursor_matches(cursor, len(text_ids))

    if prompt_tokens.require_eos and tok.eos_token_id is not None:
        text_ids.append(tok.eos_token_id)
        if model_spans_in_text and turns[-1]["role"] == prompt_tokens.model_role:
            model_spans_in_text[-1] = (model_spans_in_text[-1][0], len(text_ids))

    assert_eos_appended(text_ids, tok, prompt_tokens.require_eos)
    assert_model_spans_valid(model_spans_in_text, len(text_ids))

    return {
        "text_ids": text_ids,
        "model_spans_in_text": model_spans_in_text,
        "text_preview": "".join(text_preview_parts),
    }

def _render_chat_v1(
    tokenizer: PreTrainedTokenizer,
    turns: List[Dict[str, str]],
    *,
    schema,
) -> Dict[str, Any]:
    return _render_with_wrappers(
        tokenizer,
        turns,
        schema=schema,
        wrapper_fn=_chat_v1_wrappers,
    )

def _render_harmony_v1(
    tokenizer: PreTrainedTokenizer,
    turns: List[Dict[str, str]],
    *,
    schema,
    use_return_for_last_assistant: bool = False,
) -> Dict[str, Any]:
    tok = tokenizer
    prompt_tokens = schema.prompt

    text_ids: List[int] = []
    if prompt_tokens.require_bos and tok.bos_token_id is not None:
        text_ids.append(tok.bos_token_id)

    model_spans_in_text: List[Tuple[int, int]] = []
    cursor = len(text_ids)
    text_preview_parts: List[str] = []

    last_assistant_idx = None
    if use_return_for_last_assistant:
        for idx in range(len(turns) - 1, -1, -1):
            if turns[idx]["role"] == prompt_tokens.model_role:
                last_assistant_idx = idx
                break

    for idx, turn in enumerate(turns):
        role = turn["role"]
        text_block = turn["text_block"]
        use_return = use_return_for_last_assistant and last_assistant_idx is not None and idx == last_assistant_idx
        prefix, suffix = _harmony_v1_wrappers(schema, role, use_return=use_return)
        content = text_block
        if suffix and content.endswith(suffix):
            content = content[: -len(suffix)]
        ids_prefix = tok.encode(prefix, add_special_tokens=False)
        ids_content = tok.encode(content, add_special_tokens=False)
        ids_suffix = tok.encode(suffix, add_special_tokens=False)

        text_ids.extend(ids_prefix)
        text_ids.extend(ids_content)
        text_ids.extend(ids_suffix)

        if role == prompt_tokens.model_role:
            s = cursor + len(ids_prefix)
            e = s + len(ids_content) + len(ids_suffix)
            if e > s:
                model_spans_in_text.append((s, e))
        cursor += len(ids_prefix) + len(ids_content) + len(ids_suffix)
        text_preview_parts.append(prefix + content + suffix)

    assert_tokenization_cursor_matches(cursor, len(text_ids))

    if prompt_tokens.require_eos and tok.eos_token_id is not None:
        text_ids.append(tok.eos_token_id)
        if model_spans_in_text and turns[-1]["role"] == prompt_tokens.model_role:
            model_spans_in_text[-1] = (model_spans_in_text[-1][0], len(text_ids))

    assert_eos_appended(text_ids, tok, prompt_tokens.require_eos)
    assert_model_spans_valid(model_spans_in_text, len(text_ids))

    return {
        "text_ids": text_ids,
        "model_spans_in_text": model_spans_in_text,
        "text_preview": "".join(text_preview_parts),
    }

_PROMPT_RENDERERS: Dict[str, Callable[[PreTrainedTokenizer, List[Dict[str, str]], Any], Dict[str, Any]]] = {
    "gemma_chat_v1": _render_chat_v1,
    "qwen_chat_v1": _render_chat_v1,
}

def render_prompt_and_spans(
    tokenizer: PreTrainedTokenizer,
    turns: List[Dict[str, str]],
    *,
    schema,
    use_return_for_last_assistant: bool = False,
) -> Dict[str, Any]:
    format_id = str(schema.conversation.format_id)
    if format_id == "harmony_chat_v1":
        return _render_harmony_v1(
            tokenizer,
            turns,
            schema=schema,
            use_return_for_last_assistant=use_return_for_last_assistant,
        )
    renderer = _PROMPT_RENDERERS.get(format_id)
    if renderer is None:
        raise ValueError(f"Unknown prompt format '{format_id}'.")
    return renderer(tokenizer, turns, schema=schema)

def turn_wrappers(schema, role: str, *, use_return: bool = False) -> Tuple[str, str]:
    format_id = str(schema.conversation.format_id)
    if format_id in ("gemma_chat_v1", "qwen_chat_v1"):
        return _chat_v1_wrappers(schema, role)
    if format_id == "harmony_chat_v1":
        return _harmony_v1_wrappers(schema, role, use_return=use_return)
    raise ValueError(f"Unknown prompt format '{format_id}'.")

def assistant_generation_prefix(schema) -> str:
    format_id = str(schema.conversation.format_id)
    if format_id in ("gemma_chat_v1", "qwen_chat_v1"):
        return f"{schema.prompt.start_of_turn}{schema.prompt.model_role}\n"
    if format_id == "harmony_chat_v1":
        return f"{schema.prompt.start_of_turn}{schema.prompt.model_role}"
    raise ValueError(f"Unknown prompt format '{format_id}'.")


__all__ = ["render_prompt_and_spans", "turn_wrappers", "assistant_generation_prefix"]
