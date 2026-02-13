"""
Runtime and configuration helpers extracted from train_ecg_text.py.
These utilities keep the training entrypoint concise while preserving the
original behaviour when preparing distributed state, tokenizer metadata, and
packing configuration.
"""
from __future__ import annotations

from typing import Dict, Optional, List
import torch.distributed as dist
from transformers import AutoTokenizer

from camel.ecg_text_packing import (
    ECGSpecialTokenCatalog,
    PackingSchema,
    PromptTokens,
)
from camel.model_registry import ModelConfig, ModelRegistryError, load_registry

def is_main_process() -> bool:
    """Return True for rank 0 (or standalone execution)."""
    return (not dist.is_initialized()) or dist.get_rank() == 0

def build_packing_schema(pretrained_model_id: str) -> PackingSchema:
    """
    Construct the packing schema (prompt + conversation rules + ECG tokens)
    for the given backbone using the shared registry.
    """
    registry = load_registry()
    cfg: Optional[ModelConfig]
    try:
        cfg = registry.get(pretrained_model_id)
    except ModelRegistryError:
        cfg = None
        for name in registry.names():
            candidate = registry.get(name)
            if candidate.hf_id == pretrained_model_id:
                cfg = candidate
                break
    if cfg is None:
        raise ModelRegistryError(
            f"Pretrained model '{pretrained_model_id}' not found in registry at {registry.source_path}"
        )
    prompt_cfg = cfg.prompt_config()
    roles_dict = dict(prompt_cfg.roles or {})
    try:
        user_role = str(roles_dict["user"])
        model_role = str(roles_dict["model"])
    except KeyError as exc:
        missing = exc.args[0]
        raise ModelRegistryError(
            f"Prompt configuration for registry entry '{cfg.name}' is missing the '{missing}' role."
        ) from exc
    prompt_tokens = PromptTokens(
        start_of_turn=prompt_cfg.start_of_turn,
        end_of_turn=prompt_cfg.end_of_turn,
        user_role=user_role,
        model_role=model_role,
        require_bos=prompt_cfg.enforce_bos,
        require_eos=prompt_cfg.enforce_eos,
        allow_multiple_eos=prompt_cfg.allow_multiple_eos,
    )
    packing_cfg = cfg.packing_config()
    conversation_rules = packing_cfg.conversation
    ecg_tokens = packing_cfg.ecg_tokens
    return PackingSchema(
        prompt=prompt_tokens,
        conversation=conversation_rules,
        ecg=ecg_tokens,
    )

def initialize_tokenizer(
    model_id: str,
    *,
    trust_remote_code: bool = True,
    use_fast: Optional[bool] = None,
    add_prefix_space: Optional[bool] = None,
) -> AutoTokenizer:
    """
    Instantiate the HF tokenizer, allowing policy to be driven by the registry
    (use_fast/add_prefix_space). If not provided, defaults are use_fast=True,
    add_prefix_space=False.
    """
    # Honor registry defaults when the caller doesn't override them.
    default_use_fast = True
    default_add_prefix_space = False
    try:
        registry = load_registry()
        cfg: Optional[ModelConfig]
        try:
            cfg = registry.get(model_id)
        except ModelRegistryError:
            cfg = None
            for name in registry.names():
                candidate = registry.get(name)
                if candidate.hf_id == model_id:
                    cfg = candidate
                    break
        if cfg is not None:
            tcfg = cfg.tokenizer_config()
            default_use_fast = bool(tcfg.use_fast)
            default_add_prefix_space = bool(tcfg.add_prefix_space)
    except Exception:
        # Fall back to built-in defaults if registry is unavailable.
        pass

    return AutoTokenizer.from_pretrained(
        model_id,
        use_fast=default_use_fast if use_fast is None else bool(use_fast),
        add_prefix_space=default_add_prefix_space if add_prefix_space is None else bool(add_prefix_space),
        trust_remote_code=trust_remote_code,
    )

def register_ecg_special_tokens(
    tokenizer: AutoTokenizer,
    catalog: ECGSpecialTokenCatalog,
) -> Dict[int, int]:
    """
    Ensure the tokenizer includes the ECG special tokens from the provided catalog.
    Returns a mapping from catalog index to token ID.
    """
    # Add only tokens that are currently unknown to the tokenizer (not present
    # as core specials or regular vocab entries).
    tokens_to_add: List[str] = []
    for token in catalog.tokens:
        tok_id = tokenizer.convert_tokens_to_ids(token)
        if tok_id is None or tok_id == tokenizer.unk_token_id:
            tokens_to_add.append(token)
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    ecg_special_token_id_map: Dict[int, int] = {}
    for token, catalog_index in catalog.token_to_index.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer failed to register ECG special token: {token}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] != token_id:
            raise RuntimeError(f"ECG special token does not map to a single id: {token}")
        ecg_special_token_id_map[catalog_index] = int(token_id)
    return ecg_special_token_id_map
