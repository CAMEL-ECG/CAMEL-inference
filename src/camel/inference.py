#!/usr/bin/env python3
# inference.py — ECGText inference (model loading + generation helpers)
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig

# Local imports
from model_introspect import resolve_hidden_size as _resolve_hidden_size
from model_registry import load_registry
from training_setup import initialize_tokenizer, build_packing_schema, register_ecg_special_tokens
from model_init import build_wrapper, attach_lora, build_conv_encoder
from ecg_text_packing import (
    _normalize_conversation,
    annotate_turn_parts_with_ids,
    build_structured_turn_parts,
    build_text_only_turn_parts,
    get_ecg_special_token_catalog,
)
from prompt_renderers import render_prompt_and_spans, turn_wrappers, assistant_generation_prefix
from ecg_attention_masks import (
    ECGBlockLayout,
    ECGSequenceLayout,
    MaskBuildResult,
    ECGMaskStrategy,
    get_mask_strategy,
)
from assertions import (
    assert_ecg_blocks_consistent,
    assert_ecg_part_bounds,
    assert_layout_specials_complete,
    assert_prefix_matches_segments,
    assert_prefix_split_complete,
)
from checkpoint_utils import (
    load_llava_and_lora,
    update_wrapper_language_model,
    extract_lora_config_from_checkpoints,
    peek_projector_name,
)

# ------------------------------
# Device & conv builder
# ------------------------------

def _device(device=None) -> torch.device:
    if device:
        return device
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


_HARMONY_CHANNEL_RE = re.compile(r"<\|channel\|>(.*?)<\|message\|>", re.DOTALL)
_HARMONY_DELIM_RE = re.compile(r"<\|end\|>|<\|return\|>|<\|call\|>|<\|start\|>")


def _extract_harmony_messages(text: str) -> List[Tuple[str, str]]:
    matches = list(_HARMONY_CHANNEL_RE.finditer(text))
    if not matches:
        raise ValueError("No harmony channel headers found in model output.")
    out: List[Tuple[str, str]] = []
    for match in matches:
        channel_raw = match.group(1).strip()
        channel = channel_raw.split()[0] if channel_raw else ""
        if not channel:
            raise ValueError("Harmony channel header is empty.")
        start = match.end()
        end_match = _HARMONY_DELIM_RE.search(text, start)
        end = end_match.start() if end_match else len(text)
        out.append((channel, text[start:end]))
    return out


def _checkpoint_has_conv(ckpt_path: Optional[str]) -> bool:
    if not ckpt_path:
        return False
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint {ckpt_path} must be a dict to inspect conv metadata.")
    return isinstance(payload.get("conv"), dict)


# ------------------------------
# Prompt building & stopping
# ------------------------------

@dataclass
class PromptContext:
    """Container describing the prepared prompt state for autoregressive generation."""

    inputs_embeds: torch.Tensor
    layout: ECGSequenceLayout
    prompt_preview: str
    stop_ids: List[int]
    input_embedder: nn.Embedding
    mask_strategy: ECGMaskStrategy
    mask_result: MaskBuildResult


def _sanitize_segments(tensor: torch.Tensor) -> torch.Tensor:
    """Detach → float32 → replace NaN/Inf so downstream encoders stay numerically stable."""
    out = tensor.detach().cpu().to(dtype=torch.float32)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    min_p: float,
) -> torch.Tensor:
    """
    Draw the next token id given final-step logits and sampling parameters.

    Uses greedy decoding when temperature <= 0; otherwise applies temperature
    scaling, optional nucleus sampling, and multinomial sampling.
    """
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")

    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = logits / max(temperature, 1e-5)
    probs = torch.softmax(scaled, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    keep_mask = torch.ones_like(sorted_probs, dtype=torch.bool)

    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), sorted_probs.numel())
        top_k_mask = torch.zeros_like(sorted_probs, dtype=torch.bool)
        top_k_mask[:top_k] = True
        keep_mask &= top_k_mask

    if 0.0 < top_p < 1.0:
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = (cumulative - sorted_probs) < top_p
        cutoff_mask[0] = True  # always keep the highest-prob token
        keep_mask &= cutoff_mask

    if min_p is not None and min_p > 0.0:
        keep_mask &= sorted_probs >= float(min_p)

    filtered_probs = sorted_probs[keep_mask]
    filtered_indices = sorted_indices[keep_mask]
    if filtered_probs.numel() == 0:
        filtered_probs = sorted_probs[:1]
        filtered_indices = sorted_indices[:1]

    prob_sum = filtered_probs.sum()
    if not torch.isfinite(prob_sum) or prob_sum <= 0:
        return sorted_indices[0]
    normalized = filtered_probs / prob_sum
    next_idx = torch.multinomial(normalized, num_samples=1, replacement=False)
    return filtered_indices[next_idx].squeeze(0)


class KardiaLM:
    """High-level chat interface around an ECG language model."""

    def __init__(
        self,
        *,
        model_registry_path: Optional[str],
        model_config_name: str,
        hf_model_id_override: Optional[str],
        adapter_ckpt: str,
        conv_ckpt: Optional[str] = None,
        no_lora: bool = False,
        use_dora: bool = False,
        default_max_new_tokens: int = 1000,
        default_temperature: float = 1.0,
        default_top_k: Optional[int] = 64,
        default_top_p: float = 0.95,
        default_min_p: float = 0.0,
        mask_strategy: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        registry = load_registry(registry_path=model_registry_path)
        model_cfg = registry.get(model_config_name)

        self.model_cfg = model_cfg
        self.hf_model_id = hf_model_id_override or model_cfg.hf_id
        self.packing_schema = build_packing_schema(self.hf_model_id)
        self.tokenizer_cfg = model_cfg.tokenizer_config()
        self.arch_cfg = model_cfg.architecture_config()
        self.system_text = None
        self.developer_text = None
        if self.packing_schema.conversation.format_id == "harmony_chat_v1":
            self.system_text = model_cfg.required_prompt_text("system_prompt")
            self.developer_text = model_cfg.required_prompt_text("developer_prompt")
            if not self.system_text.strip():
                raise RuntimeError("System prompt text for harmony format must be non-empty.")
            if not self.developer_text.strip():
                raise RuntimeError("Developer prompt text for harmony format must be non-empty.")

        self.device = _device(device)
        self.dtype = torch.bfloat16
        self.mask_strategy: ECGMaskStrategy = get_mask_strategy(mask_strategy)
        self.expect_dora = bool(use_dora)

        tok = initialize_tokenizer(
            self.hf_model_id,
            trust_remote_code=True,
            use_fast=self.tokenizer_cfg.use_fast,
            add_prefix_space=self.tokenizer_cfg.add_prefix_space,
        )
        self.tokenizer = tok

        catalog = get_ecg_special_token_catalog(self.packing_schema)
        self.ecg_special_token_id_map = register_ecg_special_tokens(tok, catalog)

        pad_strategy = self.tokenizer_cfg.pad_token_strategy.lower()
        if pad_strategy == "eos":
            if tok.eos_token is None:
                raise RuntimeError(
                    f"Tokenizer for model '{model_cfg.name}' lacks an EOS token required for pad_token_strategy='eos'."
                )
            tok.pad_token = tok.eos_token
        elif pad_strategy not in ("existing", "keep"):
            raise RuntimeError(f"Unsupported pad_token_strategy '{self.tokenizer_cfg.pad_token_strategy}'.")

        if self.tokenizer_cfg.require_bos and tok.bos_token is None:
            raise RuntimeError(f"Tokenizer for model '{model_cfg.name}' is missing a BOS token.")
        if self.tokenizer_cfg.require_eos and tok.eos_token is None:
            raise RuntimeError(f"Tokenizer for model '{model_cfg.name}' is missing an EOS token.")

        attn_impl = self.arch_cfg.attn_implementation or "flash_attention_2"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                device_map=None,
            ).to(self.device)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_id,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                attn_implementation="eager",
                device_map=None,
            ).to(self.device)
        if model.get_input_embeddings().weight.shape[0] != len(tok):
            model.resize_token_embeddings(len(tok))
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        self.model = model

        adapter_ckpt_path = os.path.expanduser(adapter_ckpt)
        adapter_has_conv = _checkpoint_has_conv(adapter_ckpt_path)
        if not adapter_has_conv and not conv_ckpt:
            raise RuntimeError(
                "Adapter checkpoint lacks conv weights; supply --conv_ckpt to match training."
            )
        lora_cfg_dict = extract_lora_config_from_checkpoints(adapter_ckpt_path, None)
        active_lora_cfg: Optional[LoraConfig] = None
        if self.expect_dora and no_lora:
            raise RuntimeError("--use-dora cannot be combined with --no-lora since no adapters would be loaded.")
        if lora_cfg_dict and not no_lora:
            cfg_use_dora = bool(lora_cfg_dict.get("use_dora", False))
            if cfg_use_dora and not self.expect_dora:
                raise RuntimeError(
                    "Checkpoint adapters were trained with DoRA; re-run inference with --use-dora to load them."
                )
            if self.expect_dora and not cfg_use_dora:
                raise RuntimeError(
                    "Checkpoint adapters were trained without DoRA; omit --use-dora or use a checkpoint with DoRA."
                )
            model, active_lora_cfg = attach_lora(model, lora_cfg_dict, self.device)
            model.eval()
        elif no_lora and lora_cfg_dict:
            print("[LoRA] --no-lora set; skipping LoRA adapters from checkpoint.", flush=True)
        elif self.expect_dora:
            raise RuntimeError("--use-dora was provided, but no LoRA/DoRA adapters were found in the checkpoint.")

        conv = build_conv_encoder(
            conv_ckpt_path=None if adapter_has_conv else conv_ckpt,
            device=self.device,
            unfreeze=False,
        )
        conv.eval()
        for p in conv.parameters():
            p.requires_grad = False
        self.conv_encoder = conv

        hidden_size = _resolve_hidden_size(model, self.arch_cfg.hidden_size_attrs)
        wrapper_cls = model_cfg.resolve_wrapper_class()
        enc_out_dim = self.arch_cfg.conv_out_dim if getattr(self.arch_cfg, "conv_out_dim", None) is not None else 64
        projector_name = peek_projector_name(adapter_ckpt_path) or "linear"
        wrapper = build_wrapper(
            wrapper_cls=wrapper_cls,
            language_model=model,
            conv_encoder=conv,
            hidden_size=hidden_size,
            num_ecg_special_tokens=len(catalog.tokens),
            dtype=self.dtype,
            enc_out_dim=int(enc_out_dim),
            freeze_encoder=True,
            inference=True,
            projector_name=projector_name,
        )
        self.projector_name = projector_name
        self.wrapper = wrapper

        _extra_payload, model, inferred_lora_cfg = load_llava_and_lora(
            wrapper,
            model,
            adapter_ckpt_path,
            expect_lora=(active_lora_cfg is not None),
            load_lora=not no_lora,
        )
        update_wrapper_language_model(wrapper, model)
        if active_lora_cfg is None and inferred_lora_cfg is not None and not no_lora:
            active_lora_cfg = inferred_lora_cfg
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        inp_emb = model.get_input_embeddings().weight
        inp_dev = inp_emb.device
        target_dtype = inp_emb.dtype
        wrapper.llava_proj.to(device=inp_dev, dtype=torch.float32)
        wrapper.enc.to(device=inp_dev, dtype=torch.float32)
        wrapper.ecg_special_embed.to(device=inp_dev, dtype=target_dtype)
        llava_param = next(wrapper.llava_proj.parameters(), None)
        if llava_param is None:
            raise AssertionError("llava_proj unexpectedly has no parameters.")
        if llava_param.device != inp_dev:
            raise AssertionError(f"llava_proj on {llava_param.device}, expected {inp_dev}")
        if llava_param.dtype != torch.float32:
            raise AssertionError(f"llava_proj dtype {llava_param.dtype}, expected torch.float32")
        conv_param = next(wrapper.enc.parameters(), None)
        if conv_param is None:
            raise AssertionError("Convolutional encoder unexpectedly has no parameters.")
        if conv_param.device != inp_dev:
            raise AssertionError(f"Conv encoder on {conv_param.device}, expected {inp_dev}")
        if conv_param.dtype != torch.float32:
            raise AssertionError(f"Conv encoder dtype {conv_param.dtype}, expected torch.float32")
        assert next(wrapper.ecg_special_embed.parameters()).device == inp_dev, (
            f"ecg_special_embed on {next(wrapper.ecg_special_embed.parameters()).device}, expected {inp_dev}"
        )
        try:
            wrapper.language_model.eval()
        except Exception:
            pass

        self.default_max_new_tokens = int(default_max_new_tokens)
        self.default_temperature = float(default_temperature)
        self.default_top_k = int(default_top_k) if default_top_k is not None else None
        self.default_top_p = float(default_top_p)
        self.default_min_p = float(default_min_p)

    def chat(
        self,
        *,
        conversation: List[Dict[str, Any]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        harmony_output: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate a response for a structured multi-turn conversation."""
        context = self._prepare_prompt_context(conversation=conversation)

        max_new_tokens = int(max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens)
        temperature = float(temperature if temperature is not None else self.default_temperature)
        resolved_top_k = int(top_k) if top_k is not None else (self.default_top_k if self.default_top_k is not None else None)
        top_p = float(top_p if top_p is not None else self.default_top_p)
        min_p = float(min_p if min_p is not None else self.default_min_p)

        token_ids = self._autoregressive_generate(
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=resolved_top_k,
            top_p=top_p,
            min_p=min_p,
        )
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        if self.packing_schema.conversation.format_id == "harmony_chat_v1":
            mode = harmony_output if harmony_output is not None else "all"
            if mode != "raw":
                messages = _extract_harmony_messages(text)
                if mode == "all":
                    text = "\n".join(msg for _, msg in messages)
                elif mode == "final":
                    finals = [msg for channel, msg in messages if channel == "final"]
                    if not finals:
                        raise ValueError("No final channel output found in harmony response.")
                    text = finals[-1]
                else:
                    raise ValueError(f"Unknown harmony_output '{mode}'.")
        return text, context.prompt_preview

    def _to_waveform_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(np.asarray(value))
        else:
            tensor = torch.tensor(value, dtype=torch.float32)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.ndim == 1:
            if tensor.numel() != 256:
                raise ValueError("Expected a 256-sample vector for a single lead second.")
            tensor = tensor.view(1, 256)
        elif tensor.ndim == 2:
            if tensor.size(-1) != 256:
                raise ValueError("Waveform segments must have length 256 along the last dimension.")
        else:
            raise ValueError("Waveform tensor must be rank 1 or 2 with 256-sample segments.")
        return tensor.contiguous()

    def _prepare_prompt_context(
        self,
        *,
        conversation: List[Dict[str, Any]],
    ) -> PromptContext:
        tok = self.tokenizer
        wrapper = self.wrapper
        packing_schema = self.packing_schema
        device = self.device

        prompt_tokens = packing_schema.prompt
        if not isinstance(conversation, list) or not conversation:
            raise ValueError("conversation must be a non-empty list of turns.")

        conv_input: List[Dict[str, Any]] = []
        for turn in conversation:
            if not isinstance(turn, dict):
                raise ValueError("Conversation turns must be dicts.")
            if "from" not in turn and "role" in turn:
                turn = dict(turn)
                turn["from"] = turn.get("role")
            conv_input.append(turn)

        turns = _normalize_conversation(conv_input, packing_schema, self.system_text, self.developer_text)
        if turns[-1]["role"] != prompt_tokens.user_role:
            raise ValueError("Conversation must end with a user turn to generate.")

        def _sanitize_content(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            sanitized: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    raise ValueError("Conversation content items must be dicts.")
                item_type = item.get("type")
                if item_type == "ecg":
                    waveform = item.get("waveform_segments")
                    if not isinstance(waveform, dict):
                        raise ValueError("ECG content item missing waveform_segments mapping.")
                    wf_out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
                    for ld, value in waveform.items():
                        wf_out[str(ld)] = _sanitize_segments(self._to_waveform_tensor(value))
                    new_item = dict(item)
                    new_item["waveform_segments"] = wf_out
                    sanitized.append(new_item)
                    continue
                if item_type == "text":
                    text_val = item.get("text")
                    if not isinstance(text_val, str):
                        raise ValueError("Text content item must have a string 'text' field.")
                    if "<image>" in text_val:
                        raise ValueError("Conversation text must not contain <image> in inference mode.")
                sanitized.append(item)
            return sanitized

        ecg_blocks: List[Dict[str, Any]] = []
        turn_parts: List[List[Dict[str, Any]]] = []
        token_turns: List[Dict[str, str]] = []

        for turn in turns:
            role = turn["role"]
            content = _sanitize_content(turn["content"])
            if role == prompt_tokens.model_role:
                turn_text_block, content_parts = build_text_only_turn_parts(
                    content=content,
                    canonical_role=role,
                    schema=packing_schema,
                )
            else:
                turn_text_block, content_parts = build_structured_turn_parts(
                    content=content,
                    canonical_role=role,
                    schema=packing_schema,
                    ecg_blocks=ecg_blocks,
                    sampling_rate=None,
                )
            prefix, suffix = turn_wrappers(packing_schema, role)
            parts = [{"kind": "text", "text": prefix}]
            parts.extend(content_parts)
            parts.append({"kind": "text", "text": suffix})
            turn_parts.append(parts)
            token_turns.append({"role": role, "text_block": turn_text_block})

        if not ecg_blocks:
            raise ValueError("No ECG blocks found in conversation.")

        turn_parts = annotate_turn_parts_with_ids(turn_parts, tok)
        assert_ecg_blocks_consistent(turn_parts=turn_parts, ecg_blocks=ecg_blocks)

        token_struct = render_prompt_and_spans(tok, token_turns, schema=packing_schema)
        text_ids = list(token_struct["text_ids"])
        if (
            prompt_tokens.require_eos
            and tok.eos_token_id is not None
            and text_ids
            and text_ids[-1] == tok.eos_token_id
        ):
            text_ids = text_ids[:-1]
        text_preview = token_struct.get("text_preview", "")

        model_prefix = assistant_generation_prefix(packing_schema)
        ids_model_prefix = tok.encode(model_prefix, add_special_tokens=False)
        text_ids.extend(ids_model_prefix)
        prompt_preview = text_preview + model_prefix

        model_prefix_parts = [{"kind": "text", "text": model_prefix, "ids": ids_model_prefix}]
        all_parts = list(turn_parts) + [model_prefix_parts]

        flat_blocks = [blk["waveform_segments"] for blk in ecg_blocks]
        lead_orders = [blk["lead_names"] for blk in ecg_blocks]
        prefix_all, prefix_lens = wrapper.ecg_prefix_batch(
            flat_blocks,
            device=device,
            lead_orders=lead_orders,
        )
        prefixes: List[torch.Tensor] = []
        offset = 0
        for n in prefix_lens:
            prefixes.append(prefix_all[offset:offset + int(n)])
            offset += int(n)
        assert_prefix_split_complete(offset=offset, total_prefix_rows=int(prefix_all.size(0)))

        block_layouts: List[ECGBlockLayout] = []
        lead_offsets: List[Dict[str, int]] = []
        lead_special_counts: List[Dict[str, int]] = []
        for blk_idx, blk in enumerate(ecg_blocks):
            lead_names = [str(ld) for ld in blk.get("lead_names", [])]
            segs_per_lead = [int(n) for n in blk.get("segments_per_lead", [])]
            prefix_rows = prefixes[blk_idx].size(0) if blk_idx < len(prefixes) else 0
            assert_prefix_matches_segments(
                prefix_rows=prefix_rows,
                segments_per_lead=segs_per_lead,
                lead_names=lead_names,
                sample_index=0,
                block_index=blk_idx,
            )
            lead_to_offset: Dict[str, int] = {}
            c = 0
            for ld, nseg in zip(lead_names, segs_per_lead):
                lead_to_offset[ld] = c
                c += int(nseg)
            lead_offsets.append(lead_to_offset)
            block_layouts.append(ECGBlockLayout(
                start_idx=None,
                end_idx_exclusive=None,
                global_start_idx=None,
                global_end_idx=None,
                lead_start_idx={},
                lead_end_idx={},
                signal_pos_by_lead={ld: [None] * int(nseg) for ld, nseg in zip(lead_names, segs_per_lead)},
                time_to_signal_idxs={},
                declared_segments_per_lead={ld: int(nseg) for ld, nseg in zip(lead_names, segs_per_lead)},
            ))
            lead_special_counts.append({})

        special_indices: List[int] = [
            int(part["token_index"])
            for turn in all_parts
            for part in turn
            if part.get("kind") == "special"
        ]
        if special_indices:
            special_idx_tensor = torch.tensor(special_indices, dtype=torch.long, device=device)
            special_embeds = wrapper.ecg_special_tokens_to_embeds(special_idx_tensor, device=device)
        else:
            special_embeds = torch.empty((0, wrapper.hidden_size), dtype=wrapper.dtype, device=device)

        input_embedder = wrapper.language_model.get_input_embeddings()
        if text_ids:
            E_text_all = wrapper.tokens_to_embeds(input_embedder, text_ids, device=device)
        else:
            E_text_all = torch.empty((0, wrapper.hidden_size), dtype=wrapper.dtype, device=device)

        text_cursor = 0
        empty_text = E_text_all[:0]
        chunks: List[torch.Tensor] = []
        layout = ECGSequenceLayout(seq_len=0, text_idxs=[], blocks=block_layouts)

        def _take_text(count: int) -> torch.Tensor:
            nonlocal text_cursor
            if count <= 0:
                return empty_text
            end = text_cursor + count
            if end > E_text_all.size(0):
                raise RuntimeError("Text embedding cursor exceeded available embeddings")
            out = E_text_all[text_cursor:end]
            text_cursor = end
            return out

        def _record_text(count: int, cursor: int) -> None:
            for i in range(count):
                layout.text_idxs.append(cursor + i)

        cursor = 0
        special_cursor = 0

        if (
            text_ids
            and prompt_tokens.require_bos
            and tok.bos_token_id is not None
            and text_ids[0] == tok.bos_token_id
        ):
            E_bos = _take_text(1)
            chunks.append(E_bos)
            _record_text(1, cursor)
            cursor += 1

        for turn in all_parts:
            for part in turn:
                kind = part.get("kind")
                if kind == "text":
                    ids_chunk = part.get("ids")
                    if ids_chunk is None:
                        txt = part.get("text", "")
                        ids_chunk = tok.encode(txt, add_special_tokens=False) if txt else []
                    if ids_chunk:
                        if ids_chunk != text_ids[text_cursor:text_cursor + len(ids_chunk)]:
                            raise RuntimeError("Special token id does not match text_ids cursor.")
                        E_chunk = _take_text(len(ids_chunk))
                        chunks.append(E_chunk)
                        _record_text(len(ids_chunk), cursor)
                        cursor += len(ids_chunk)
                    continue
                if kind == "special":
                    if special_cursor >= special_embeds.size(0):
                        raise RuntimeError("Special-token cursor exceeded embeddings.")
                    tok_idx = int(part.get("token_index", -1))
                    expected_id = self.ecg_special_token_id_map.get(tok_idx)
                    if expected_id is None:
                        raise RuntimeError(f"Unknown ECG special token index {tok_idx}.")
                    if text_cursor >= len(text_ids):
                        raise RuntimeError("Text cursor exceeded available text ids.")
                    if text_ids[text_cursor] != expected_id:
                        raise RuntimeError("Special token id does not match text_ids cursor.")
                    _take_text(1)
                    chunks.append(special_embeds[special_cursor:special_cursor + 1])
                    _record_text(1, cursor)

                    block_index = int(part.get("block_index", -1))
                    if block_index < 0 or block_index >= len(block_layouts):
                        raise RuntimeError("ECG part references unknown block_index.")
                    block_layout = block_layouts[block_index]
                    lead_name = part.get("lead")
                    if lead_name:
                        cnt = lead_special_counts[block_index].get(lead_name, 0)
                        if cnt == 0:
                            block_layout.lead_start_idx[lead_name] = cursor
                        else:
                            block_layout.lead_end_idx[lead_name] = cursor
                        lead_special_counts[block_index][lead_name] = cnt + 1
                    else:
                        if block_layout.global_start_idx is None:
                            block_layout.global_start_idx = cursor
                            block_layout.start_idx = cursor
                        else:
                            block_layout.global_end_idx = cursor
                            block_layout.end_idx_exclusive = cursor + 1

                    cursor += 1
                    special_cursor += 1
                    continue
                if kind == "ecg":
                    block_index = int(part.get("block_index", -1))
                    if block_index < 0 or block_index >= len(block_layouts):
                        raise RuntimeError("ECG part references unknown block_index.")
                    ld = part["lead"]
                    sec = int(part["sec"])
                    lead_to_offset = lead_offsets[block_index]
                    block_layout = block_layouts[block_index]
                    prefix_all = prefixes[block_index]
                    assert_ecg_part_bounds(
                        lead=ld,
                        sec=sec,
                        lead_to_offset=lead_to_offset,
                        declared_segments=block_layout.declared_segments_per_lead,
                        total_prefix_rows=prefix_all.size(0),
                        sample_index=0,
                        block_index=block_index,
                    )
                    base = lead_to_offset[ld]
                    row_idx = base + (sec - 1)
                    chunks.append(prefix_all[row_idx:row_idx + 1])
                    sig_list = block_layout.signal_pos_by_lead[ld]
                    if sec - 1 >= len(sig_list):
                        raise RuntimeError("ECG segment index exceeds declared segments_per_lead")
                    sig_list[sec - 1] = cursor
                    block_layout.time_to_signal_idxs.setdefault(sec, []).append(cursor)
                    cursor += 1
                    continue
                raise RuntimeError(f"Unknown turn part kind '{kind}'.")

        remaining = len(text_ids) - text_cursor
        if remaining > 0:
            E_tail = _take_text(remaining)
            chunks.append(E_tail)
            _record_text(remaining, cursor)
            cursor += remaining

        if special_cursor != special_embeds.size(0):
            raise RuntimeError("Did not consume all special-token embeddings for prompt")
        if text_cursor != E_text_all.size(0):
            raise RuntimeError("Text embedding cursor did not consume all embeddings")

        inputs_embeds = torch.cat(chunks, dim=0)
        layout.seq_len = inputs_embeds.size(0)

        for blk_idx, blk_layout in enumerate(block_layouts):
            for ld, expected in blk_layout.declared_segments_per_lead.items():
                slots = blk_layout.signal_pos_by_lead[ld]
                if any(pos is None for pos in slots):
                    raise RuntimeError(f"Lead {ld} missing ECG slots; expected {expected}.")
                blk_layout.signal_pos_by_lead[ld] = [int(pos) for pos in slots]
            if blk_layout.global_start_idx is None or blk_layout.global_end_idx is None:
                raise RuntimeError("ECG block missing global start/end specials.")
            if blk_layout.end_idx_exclusive is None:
                blk_layout.end_idx_exclusive = int(blk_layout.global_end_idx) + 1
            if blk_layout.start_idx is None:
                blk_layout.start_idx = int(blk_layout.global_start_idx)
            assert_layout_specials_complete(
                block_layout=blk_layout,
                lead_names=ecg_blocks[blk_idx]["lead_names"],
            )
            all_specials = []
            if blk_layout.global_start_idx is not None:
                all_specials.append(blk_layout.global_start_idx)
            all_specials.extend(list(blk_layout.lead_start_idx.values()))
            all_specials.extend(list(blk_layout.lead_end_idx.values()))
            if blk_layout.global_end_idx is not None:
                all_specials.append(blk_layout.global_end_idx)
            blk_layout.special_idxs_sorted = sorted(all_specials)
            blk_layout.signal_pos_list = sorted(
                [p for lst in blk_layout.signal_pos_by_lead.values() for p in lst]
            )

        mask_result = self.mask_strategy.build(
            layout,
            device=device,
            dtype=inputs_embeds.dtype,
        )
        use_return = self.packing_schema.conversation.format_id == "harmony_chat_v1"
        _, stop_text = turn_wrappers(self.packing_schema, prompt_tokens.model_role, use_return=use_return)
        stop_ids = tok.encode(stop_text, add_special_tokens=False)
        return PromptContext(
            inputs_embeds=inputs_embeds,
            layout=layout,
            prompt_preview=prompt_preview,
            stop_ids=stop_ids,
            input_embedder=input_embedder,
            mask_strategy=self.mask_strategy,
            mask_result=mask_result,
        )

    def _autoregressive_generate(
        self,
        *,
        context: PromptContext,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
        min_p: float,
    ) -> List[int]:
        tok = self.tokenizer
        wrapper = self.wrapper
        device = context.inputs_embeds.device
        embeds = context.inputs_embeds.clone()
        layout = context.layout
        input_embedder = context.input_embedder
        mask_result = context.mask_result

        generated: List[int] = []
        stop_ids = context.stop_ids
        stop_len = len(stop_ids)
        eos_id = tok.eos_token_id

        for _ in range(max_new_tokens):
            additive = mask_result.additive.unsqueeze(0).unsqueeze(0)
            outputs = wrapper.forward_language_model(
                inputs_embeds=embeds.unsqueeze(0),
                attention_mask=additive,
                labels=None,
            )
            logits = outputs.logits[0, -1, :].float()
            next_token = _sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
            )
            token_id = int(next_token.item())
            generated.append(token_id)

            if stop_len and generated[-stop_len:] == stop_ids:
                generated = generated[:-stop_len]
                break
            if eos_id is not None and token_id == eos_id:
                break

            with torch.no_grad():
                new_embed = wrapper.tokens_to_embeds(input_embedder, [token_id], device=device)
            embeds = torch.cat([embeds, new_embed], dim=0)

            layout.seq_len = embeds.size(0)
            new_idx = layout.seq_len - 1
            layout.text_idxs.append(new_idx)
            mask_result = context.mask_strategy.update_for_generated_token(
                layout,
                device=device,
                dtype=embeds.dtype,
                previous=mask_result,
            )
            context.mask_result = mask_result

        return generated
