"""
Utilities for loading per-model configuration metadata used across training and inference.

The registry is defined in YAML (see model_registry.yaml in this directory) and exposes
immutable ModelConfig objects for downstream consumers. The intent is to centralize
model-specific defaults (prompt format, tokenizer quirks, wrapper class path, LoRA constraints, etc.)
so that adding support for a new backbone primarily involves updating the registry.
"""
from __future__ import annotations

import copy
import importlib
import dataclasses
import os
from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
import yaml

class ModelRegistryError(RuntimeError):
    """Raised when the registry file is missing or malformed."""

@dataclasses.dataclass(frozen=True)
class PromptConfig:
    start_of_turn: str
    end_of_turn: str
    roles: Mapping[str, str]
    enforce_bos: bool
    enforce_eos: bool
    allow_multiple_eos: bool


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    pad_token_strategy: str
    require_bos: bool
    require_eos: bool
    use_fast: bool = True
    add_prefix_space: bool = False


@dataclasses.dataclass(frozen=True)
class ArchitectureConfig:
    wrapper_class: str
    hidden_size_attrs: Tuple[str, ...]
    language_model_path_hints: Tuple[str, ...]
    attn_implementation: str
    conv_out_dim: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class LoRAPolicyConfig:
    expect_language_only: bool
    allowed_markers: Tuple[str, ...]
    blocked_markers: Tuple[str, ...]
    freeze_vision: bool


@dataclasses.dataclass(frozen=True)
class PackingConversationConfig:
    format_id: str
    user_role_aliases: Tuple[str, ...]
    model_role_aliases: Tuple[str, ...]
    strip_image_from_roles: Tuple[str, ...]
    merge_system_with_first_user: bool


@dataclasses.dataclass(frozen=True)
class PackingECGTokensConfig:
    global_start: str
    global_end: str
    lead_start_template: str
    lead_end_template: str
    canonical_leads: Tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PackingConfig:
    prompt_format: str
    conversation: PackingConversationConfig
    ecg_tokens: PackingECGTokensConfig


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Typed wrapper over a single model entry in the registry."""

    name: str
    data: Mapping[str, Any]

    @property
    def hf_id(self) -> str:
        return _require_str(self.data, "hf_id", self.name)

    @property
    def prompt(self) -> Mapping[str, Any]:
        return _require_mapping(self.data, "prompt", self.name)

    @property
    def tokenizer(self) -> Mapping[str, Any]:
        return _require_mapping(self.data, "tokenizer", self.name)

    @property
    def architecture(self) -> Mapping[str, Any]:
        return _require_mapping(self.data, "architecture", self.name)

    @property
    def lora_policy(self) -> Mapping[str, Any]:
        return _require_mapping(self.data, "lora_policy", self.name)

    @property
    def packing(self) -> Mapping[str, Any]:
        return _require_mapping(self.data, "packing", self.name)

    def prompt_config(self) -> PromptConfig:
        prompt = self.prompt
        roles = _require_mapping(prompt, "roles", self.name, section="prompt")
        return PromptConfig(
            start_of_turn=_require_str(prompt, "start_of_turn", self.name, section="prompt"),
            end_of_turn=_require_str(prompt, "end_of_turn", self.name, section="prompt"),
            roles={k: str(v) for k, v in roles.items()},
            enforce_bos=_require_bool(prompt, "enforce_bos", self.name, section="prompt"),
            enforce_eos=_require_bool(prompt, "enforce_eos", self.name, section="prompt"),
            allow_multiple_eos=_optional_bool(prompt, "allow_multiple_eos", False, self.name, section="prompt"),
        )

    def required_prompt_text(self, key: str) -> str:
        return _require_str(self.prompt, key, self.name, section="prompt")

    def tokenizer_config(self) -> TokenizerConfig:
        tokenizer = self.tokenizer
        pad_strategy = _require_str(tokenizer, "pad_token_strategy", self.name, section="tokenizer")
        require_bos = _require_bool(tokenizer, "require_bos", self.name, section="tokenizer")
        require_eos = _require_bool(tokenizer, "require_eos", self.name, section="tokenizer")
        # Optional fields with defaults for backward compatibility
        use_fast = _optional_bool(tokenizer, "use_fast", True, self.name, section="tokenizer")
        add_prefix_space = _optional_bool(tokenizer, "add_prefix_space", False, self.name, section="tokenizer")
        return TokenizerConfig(
            pad_token_strategy=pad_strategy,
            require_bos=require_bos,
            require_eos=require_eos,
            use_fast=use_fast,
            add_prefix_space=add_prefix_space,
        )

    def architecture_config(self) -> ArchitectureConfig:
        arch = self.architecture
        conv_out = arch.get("conv_out_dim") if isinstance(arch, ABCMapping) else None
        try:
            conv_out_int = int(conv_out) if conv_out is not None else None
        except Exception:
            conv_out_int = None
        return ArchitectureConfig(
            wrapper_class=_require_str(arch, "wrapper_class", self.name, section="architecture"),
            hidden_size_attrs=tuple(
                _require_sequence_of_str(arch, "hidden_size_attrs", self.name, section="architecture")
            ),
            language_model_path_hints=tuple(
                _require_sequence_of_str(arch, "language_model_path_hints", self.name, section="architecture")
            ),
            attn_implementation=_require_str(arch, "attn_implementation", self.name, section="architecture"),
            conv_out_dim=conv_out_int,
        )

    def lora_policy_config(self) -> LoRAPolicyConfig:
        lora = self.lora_policy
        return LoRAPolicyConfig(
            expect_language_only=_require_bool(lora, "expect_language_only", self.name, section="lora_policy"),
            allowed_markers=tuple(
                _require_sequence_of_str(lora, "allowed_markers", self.name, section="lora_policy")
            ),
            blocked_markers=tuple(
                _require_sequence_of_str(lora, "blocked_markers", self.name, section="lora_policy")
            ),
            freeze_vision=_require_bool(lora, "freeze_vision", self.name, section="lora_policy"),
        )

    def packing_config(self) -> PackingConfig:
        packing = self.packing
        format_id = _require_str(packing, "prompt_format", self.name, section="packing")

        conversation = _require_mapping(packing, "conversation", self.name, section="packing")
        user_aliases = tuple(
            _require_sequence_of_str(conversation, "user_role_aliases", self.name, section="packing.conversation")
        )
        model_aliases = tuple(
            _require_sequence_of_str(conversation, "model_role_aliases", self.name, section="packing.conversation")
        )
        strip_roles = tuple(
            _require_sequence_of_str(conversation, "strip_image_from_roles", self.name, section="packing.conversation")
        )
        merge_system = _require_bool(
            conversation, "merge_system_with_first_user", self.name, section="packing.conversation"
        )
        conv_cfg = PackingConversationConfig(
            format_id=format_id,
            user_role_aliases=user_aliases,
            model_role_aliases=model_aliases,
            strip_image_from_roles=strip_roles,
            merge_system_with_first_user=merge_system,
        )

        ecg_tokens = _require_mapping(packing, "ecg_tokens", self.name, section="packing")
        global_start = _require_str(ecg_tokens, "global_start", self.name, section="packing.ecg_tokens")
        global_end = _require_str(ecg_tokens, "global_end", self.name, section="packing.ecg_tokens")
        lead_start_template = _require_str(ecg_tokens, "lead_start_template", self.name, section="packing.ecg_tokens")
        lead_end_template = _require_str(ecg_tokens, "lead_end_template", self.name, section="packing.ecg_tokens")
        canonical_leads = tuple(
            _require_sequence_of_str(ecg_tokens, "canonical_leads", self.name, section="packing.ecg_tokens")
        )
        ecg_cfg = PackingECGTokensConfig(
            global_start=global_start,
            global_end=global_end,
            lead_start_template=lead_start_template,
            lead_end_template=lead_end_template,
            canonical_leads=canonical_leads,
        )

        return PackingConfig(
            prompt_format=format_id,
            conversation=conv_cfg,
            ecg_tokens=ecg_cfg,
        )

    def resolve_wrapper_class(self):
        arch = self.architecture_config()
        if "." not in arch.wrapper_class:
            raise ModelRegistryError(
                f"Wrapper class path '{arch.wrapper_class}' for model '{self.name}' must be in 'module.ClassName' form."
            )
        module_name, class_name = arch.wrapper_class.rsplit(".", 1)
        try:
            module = importlib.import_module("." + module_name, package=__package__)
        except ImportError as exc:
            raise ModelRegistryError(
                f"Failed to import wrapper module '{module_name}' for model '{self.name}': {exc}"
            ) from exc
        try:
            wrapper_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ModelRegistryError(
                f"Wrapper class '{class_name}' not found in module '{module_name}' for model '{self.name}'."
            ) from exc
        return wrapper_cls


def _default_registry_path() -> Path:
    return Path(__file__).resolve().with_name("model_registry.yaml")


def load_registry(
    *,
    registry_path: Optional[os.PathLike[str] | str] = None,
    model_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> "ModelRegistry":
    """
    Load the model registry from YAML.

    Args:
        registry_path: Optional path to the YAML file. Defaults to `model_registry.yaml` alongside this module.
        model_overrides: Optional mapping of model name -> override dict that will be deep-merged
                         onto the YAML entry (useful for ad-hoc experimentation).
    """
    path = Path(registry_path) if registry_path is not None else _default_registry_path()
    if not path.exists():
        raise ModelRegistryError(f"Model registry file not found at {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ModelRegistryError(f"Failed to parse model registry YAML: {exc}") from exc

    if not isinstance(raw, ABCMapping):
        raise ModelRegistryError("Model registry root must be a mapping.")

    models = raw.get("models")
    if not isinstance(models, ABCMapping) or len(models) == 0:
        raise ModelRegistryError("Model registry must define a non-empty 'models' mapping.")

    entries: Dict[str, Mapping[str, Any]] = {}
    for name, cfg in models.items():
        if not isinstance(name, str):
            raise ModelRegistryError("Model names must be strings.")
        if not isinstance(cfg, ABCMapping):
            raise ModelRegistryError(f"Model '{name}' entry must be a mapping.")
        merged = copy.deepcopy(cfg)
        if model_overrides and name in model_overrides:
            _deep_update(merged, model_overrides[name])
        _validate_model_entry(name, merged)
        entries[name] = merged

    return ModelRegistry(entries, source_path=path)


class ModelRegistry:
    """In-memory view over the registry."""

    def __init__(self, models: Mapping[str, Mapping[str, Any]], *, source_path: Path):
        self._models = dict(models)
        self._source_path = Path(source_path)

    @property
    def source_path(self) -> Path:
        return self._source_path

    def names(self) -> Iterable[str]:
        return tuple(self._models.keys())

    def get(self, name: str) -> ModelConfig:
        if name not in self._models:
            raise ModelRegistryError(f"Unknown model '{name}'. Known models: {sorted(self._models)}")
        return ModelConfig(name=name, data=_deep_freeze(self._models[name]))


def _validate_model_entry(name: str, entry: Mapping[str, Any]) -> None:
    # Required string
    _require_str(entry, "hf_id", name)

    # Prompt section
    prompt = _require_mapping(entry, "prompt", name)
    _require_str(prompt, "start_of_turn", name, section="prompt")
    _require_str(prompt, "end_of_turn", name, section="prompt")
    roles = _require_mapping(prompt, "roles", name, section="prompt")
    for role_key in ("user", "model"):
        _require_str(roles, role_key, name, section="prompt.roles")
    for flag in ("enforce_bos", "enforce_eos"):
        _require_bool(prompt, flag, name, section="prompt")

    # Tokenizer section
    tokenizer = _require_mapping(entry, "tokenizer", name)
    _require_str(tokenizer, "pad_token_strategy", name, section="tokenizer")
    for flag in ("require_bos", "require_eos"):
        _require_bool(tokenizer, flag, name, section="tokenizer")

    # Architecture
    architecture = _require_mapping(entry, "architecture", name)
    _require_str(architecture, "wrapper_class", name, section="architecture")
    hidden_attrs = _require_sequence_of_str(architecture, "hidden_size_attrs", name, section="architecture")
    if len(hidden_attrs) == 0:
        raise ModelRegistryError(
            f"Model '{name}' architecture.hidden_size_attrs must contain at least one attribute path."
        )
    _require_sequence_of_str(architecture, "language_model_path_hints", name, section="architecture")
    _require_str(architecture, "attn_implementation", name, section="architecture")
    # conv_out_dim is optional; if present, ensure it is an int-like
    if "conv_out_dim" in architecture:
        val = architecture.get("conv_out_dim")
        try:
            int(val)  # type: ignore[arg-type]
        except Exception:
            raise ModelRegistryError(
                f"Model '{name}' field 'architecture.conv_out_dim' must be an integer when provided."
            )

    # LoRA policy
    lora_policy = _require_mapping(entry, "lora_policy", name)
    _require_bool(lora_policy, "expect_language_only", name, section="lora_policy")
    allowed = _require_sequence_of_str(lora_policy, "allowed_markers", name, section="lora_policy")
    blocked = _require_sequence_of_str(lora_policy, "blocked_markers", name, section="lora_policy")
    overlap = set(allowed).intersection(blocked)
    if overlap:
        raise ModelRegistryError(
            f"Model '{name}' lora_policy.allowed_markers and lora_policy.blocked_markers overlap: {sorted(overlap)}"
        )
    _require_bool(lora_policy, "freeze_vision", name, section="lora_policy")

    # Packing
    packing = _require_mapping(entry, "packing", name)
    _require_str(packing, "prompt_format", name, section="packing")
    conversation = _require_mapping(packing, "conversation", name, section="packing")
    _require_sequence_of_str(conversation, "user_role_aliases", name, section="packing.conversation")
    _require_sequence_of_str(conversation, "model_role_aliases", name, section="packing.conversation")
    _require_sequence_of_str(conversation, "strip_image_from_roles", name, section="packing.conversation")
    _require_bool(conversation, "merge_system_with_first_user", name, section="packing.conversation")
    ecg_tokens = _require_mapping(packing, "ecg_tokens", name, section="packing")
    _require_str(ecg_tokens, "global_start", name, section="packing.ecg_tokens")
    _require_str(ecg_tokens, "global_end", name, section="packing.ecg_tokens")
    _require_str(ecg_tokens, "lead_start_template", name, section="packing.ecg_tokens")
    _require_str(ecg_tokens, "lead_end_template", name, section="packing.ecg_tokens")
    if len(_require_sequence_of_str(ecg_tokens, "canonical_leads", name, section="packing.ecg_tokens")) == 0:
        raise ModelRegistryError(
            f"Model '{name}' packing.ecg_tokens.canonical_leads must contain at least one lead."
        )


def _require_mapping(
    parent: Mapping[str, Any],
    key: str,
    model_name: str,
    *,
    section: Optional[str] = None,
) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, ABCMapping):
        loc = f"{section}.{key}" if section else key
        raise ModelRegistryError(f"Model '{model_name}' is missing mapping '{loc}'.")
    return value


def _require_str(
    parent: Mapping[str, Any],
    key: str,
    model_name: str,
    *,
    section: Optional[str] = None,
) -> str:
    value = parent.get(key)
    if not isinstance(value, str):
        loc = f"{section}.{key}" if section else key
        raise ModelRegistryError(f"Model '{model_name}' field '{loc}' must be a string.")
    return value


def _require_bool(
    parent: Mapping[str, Any],
    key: str,
    model_name: str,
    *,
    section: Optional[str] = None,
) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        loc = f"{section}.{key}" if section else key
        raise ModelRegistryError(f"Model '{model_name}' field '{loc}' must be a boolean.")
    return value


def _require_sequence_of_str(
    parent: Mapping[str, Any],
    key: str,
    model_name: str,
    *,
    section: Optional[str] = None,
) -> Tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, ABCSequence) or isinstance(value, (str, bytes)):
        loc = f"{section}.{key}" if section else key
        raise ModelRegistryError(f"Model '{model_name}' field '{loc}' must be a sequence of strings.")
    items = []
    for item in value:
        if not isinstance(item, str):
            loc = f"{section}.{key}" if section else key
            raise ModelRegistryError(f"Model '{model_name}' field '{loc}' must contain only strings.")
        items.append(item)
    return tuple(items)


def _deep_update(target: Dict[str, Any], updates: Mapping[str, Any]) -> None:
    """
    Recursively merge `updates` into `target` in-place.
    """
    for key, value in updates.items():
        if isinstance(value, ABCMapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)  # type: ignore[arg-type]
        else:
            target[key] = copy.deepcopy(value)


def _optional_bool(
    parent: Mapping[str, Any],
    key: str,
    default: bool,
    model_name: str,
    *,
    section: Optional[str] = None,
) -> bool:
    if key not in parent:
        return default
    value = parent.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
    loc = f"{section}.{key}" if section else key
    raise ModelRegistryError(f"Model '{model_name}' field '{loc}' must be a boolean when provided.")


def _deep_freeze(obj: Any) -> Any:
    """
    Recursively convert mutable containers to immutable/read-only equivalents.
    """
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(v) for v in obj)
    if isinstance(obj, tuple):
        return tuple(_deep_freeze(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(_deep_freeze(v) for v in obj)
    return obj
