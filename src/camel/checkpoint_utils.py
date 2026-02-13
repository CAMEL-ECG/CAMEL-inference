"""
Checkpoint- and state-management utilities shared by ECG training scripts.
These helpers were extracted from train_ecg_text.py to keep the training entrypoint
focused on orchestration while preserving original behaviour.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DTensor
from peft import (
    LoraConfig,
    TaskType,
    set_peft_model_state_dict,
)

from training_setup import is_main_process

def _module_has_dtensor_params(mod: nn.Module) -> bool:
    """
    Return True if any parameter tensor underlying the module is a DTensor.
    Under FSDP2 it is typically parameter.data that carries the DTensor type.
    """
    for param in mod.parameters(recurse=True):
        if isinstance(getattr(param, "data", None), DTensor):
            return True
    return False

def _extract_projector_name(payload: Dict[str, Any]) -> Optional[str]:
    """Return the stored projector name if present."""
    name = payload.get("projector_name")
    if isinstance(name, str) and name:
        return name
    extra = payload.get("extra")
    if isinstance(extra, dict):
        extra_name = extra.get("projector_name")
        if isinstance(extra_name, str) and extra_name:
            return extra_name
    return None

def peek_projector_name(path: str) -> Optional[str]:
    """Load a checkpoint just far enough to read the projector name metadata."""
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint {path} must be a dict to inspect projector metadata.")
    return _extract_projector_name(payload)

def extract_lora_config_from_checkpoints(
    resume_ckpt_path: Optional[str],
    load_llava_from: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Extract the LoRA configuration embedded in checkpoints (resume path preferred)."""
    def _load_config(path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not path:
            return None
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load checkpoint '{path}' while extracting LoRA config"
            ) from exc
        lora_payload = payload.get("lora")
        if isinstance(lora_payload, dict) and isinstance(lora_payload.get("config"), dict):
            cfg = dict(lora_payload["config"])
            if "use_dora" in cfg:
                cfg["use_dora"] = bool(cfg["use_dora"])
            return cfg
        return None
    cfg = _load_config(resume_ckpt_path)
    if cfg is not None:
        return cfg
    return _load_config(load_llava_from)

def load_llava_and_lora(
    wrapper: nn.Module,
    model: nn.Module,
    ckpt_path: str,
    *,
    expect_lora: bool,
    load_lora: bool = True,
    missing_lora_ok: bool = False,
) -> Tuple[Dict[str, Any], nn.Module, Optional[LoraConfig]]:
    """
    Load llava_proj (mandatory), optional conv encoder weights, ECG special-token
    embeddings, and LoRA adapters from a checkpoint.
    """
    payload = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint {ckpt_path} must be a dict, got {type(payload).__name__}")
    extra_payload = payload.get("extra") or {}
    if not isinstance(extra_payload, dict):
        raise RuntimeError(
            f"Checkpoint {ckpt_path} has non-dict extra payload of type {type(extra_payload).__name__}"
        )
    ckpt_projector = _extract_projector_name(payload)
    wrapper_projector = getattr(wrapper, "projector_name", None)
    if ckpt_projector is not None and wrapper_projector is not None and wrapper_projector != ckpt_projector:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} projector '{ckpt_projector}' does not match wrapper projector '{wrapper_projector}'."
        )
    llava_sd = payload.get("llava_proj")
    if not isinstance(llava_sd, dict):
        raise RuntimeError(f"Checkpoint {ckpt_path} missing llava_proj state_dict.")
    if is_main_process():
        print(f"[load-llava] Loading llava_proj weights from {ckpt_path}", flush=True)
    any_llava_dt = _module_has_dtensor_params(wrapper.llava_proj)
    if any_llava_dt:
        set_model_state_dict(
            model=wrapper.llava_proj,
            model_state_dict=llava_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
    else:
        wrapper.llava_proj.load_state_dict(llava_sd, strict=True)
    conv_sd = payload.get("conv")
    conv_expected = bool(extra_payload.get("conv_trainable"))
    if conv_sd is None:
        if conv_expected:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} indicates conv_trainable=True but conv weights are missing."
            )
    else:
        if not isinstance(conv_sd, dict):
            raise RuntimeError(
                f"Checkpoint {ckpt_path} conv payload must be a state_dict, got {type(conv_sd).__name__}"
            )
        if is_main_process():
            print(f"[load-llava] Loading conv encoder weights from {ckpt_path}", flush=True)
        any_conv_dt = _module_has_dtensor_params(wrapper.enc)
        if any_conv_dt:
            set_model_state_dict(
                model=wrapper.enc,
                model_state_dict=conv_sd,
                options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
            )
        else:
            wrapper.enc.load_state_dict(conv_sd, strict=True)
    ecg_special_sd = payload.get("ecg_special")
    if not isinstance(ecg_special_sd, dict):
        raise RuntimeError(f"Checkpoint {ckpt_path} missing ECG special-token embedding state.")
    if is_main_process():
        print(f"[load-llava] Loading ECG special-token embedding from {ckpt_path}", flush=True)
    any_special_dt = _module_has_dtensor_params(wrapper.ecg_special_embed)
    if any_special_dt:
        set_model_state_dict(
            model=wrapper.ecg_special_embed,
            model_state_dict=ecg_special_sd,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
    else:
        wrapper.ecg_special_embed.load_state_dict(ecg_special_sd, strict=True)
    lora_payload = payload.get("lora")
    loaded_lora = False
    created_cfg: Optional[LoraConfig] = None
    if load_lora and lora_payload is not None:
        if not isinstance(lora_payload, dict):
            raise RuntimeError(
                f"Checkpoint {ckpt_path} has non-dict LoRA payload of type {type(lora_payload).__name__}"
            )
        lora_state = lora_payload.get("state_dict")
        if not isinstance(lora_state, dict):
            raise RuntimeError(f"Checkpoint {ckpt_path} LoRA payload missing state_dict.")
        if is_main_process():
            print(f"[load-llava] Loading LoRA adapters from {ckpt_path}", flush=True)
        set_peft_model_state_dict(model, lora_state)
        loaded_lora = True
        cfg_dict = lora_payload.get("config")
        if isinstance(cfg_dict, dict):
            cfg_args = dict(cfg_dict)
            task_type_raw = cfg_args.get("task_type", TaskType.CAUSAL_LM)
            if not isinstance(task_type_raw, TaskType):
                try:
                    task_type_raw = TaskType(task_type_raw)
                except Exception:
                    task_type_raw = TaskType.CAUSAL_LM
            cfg_args["task_type"] = task_type_raw
            if "lora_dropout" in cfg_args:
                try:
                    cfg_args["lora_dropout"] = float(cfg_args["lora_dropout"])
                except Exception:
                    cfg_args["lora_dropout"] = 0.0
            if "r" in cfg_args:
                try:
                    cfg_args["r"] = int(cfg_args["r"])
                except Exception:
                    cfg_args["r"] = 0
            if "lora_alpha" in cfg_args:
                try:
                    cfg_args["lora_alpha"] = int(cfg_args["lora_alpha"])
                except Exception:
                    cfg_args["lora_alpha"] = 0
            if "target_modules" in cfg_args and cfg_args["target_modules"] is not None:
                cfg_args["target_modules"] = list(cfg_args["target_modules"])
            cfg_args.setdefault("bias", "none")
            cfg_args.setdefault("inference_mode", False)
            cfg_args["use_dora"] = bool(cfg_args.get("use_dora", False))
            try:
                created_cfg = LoraConfig(**cfg_args)
            except Exception:
                created_cfg = None
    if expect_lora and load_lora and not loaded_lora:
        if missing_lora_ok:
            if is_main_process():
                print(
                    f"[load-llava] Warning: checkpoint {ckpt_path} contains no LoRA adapters; "
                    "continuing with the currently-initialized adapters.",
                    flush=True,
                )
        else:
            raise RuntimeError(
                f"[load-llava] Warning: expected LoRA adapters in {ckpt_path} but none were loaded."
            )
    return extra_payload, model, created_cfg

def update_wrapper_language_model(wrapper: nn.Module, model: nn.Module) -> None:
    """Ensure the wrapper references the latest language-model instance."""
    wrapper.language_model = model

__all__ = [
    "find_latest_step_checkpoint",
    "extract_lora_config_from_checkpoints",
    "dump_lora_state_fsdp_safe",
    "prepare_optimizer_state_payload",
    "load_llava_and_lora",
    "update_wrapper_language_model",
    "ensure_no_dtensor",
    "peek_projector_name",
]
