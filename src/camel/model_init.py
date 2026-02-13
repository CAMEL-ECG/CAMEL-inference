from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from ecg_gemma_model import ECGGemmaPrefix as ECGModelPrefix

def attach_lora(
    model: nn.Module,
    lora_cfg_dict: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, LoraConfig]:
    """Attach LoRA adapters to the frozen model, leaving only LoRA trainable."""
    cfg = LoraConfig(
        r=int(lora_cfg_dict["r"]),
        lora_alpha=int(lora_cfg_dict.get("lora_alpha", int(lora_cfg_dict["r"]) * 2)),
        lora_dropout=float(lora_cfg_dict.get("lora_dropout", 0.0)),
        target_modules=list(lora_cfg_dict.get("target_modules", [])),
        task_type=TaskType(lora_cfg_dict.get("task_type", "CAUSAL_LM")),
        bias=lora_cfg_dict.get("bias", "none"),
        inference_mode=False,
        use_dora=bool(lora_cfg_dict.get("use_dora", False)),
    )
    model = get_peft_model(model, cfg)
    model.to(device)
    return model, cfg

def build_conv_encoder(
    *,
    conv_ckpt_path: Optional[str],
    device: torch.device,
    unfreeze: bool = False,
) -> nn.Module:
    """
    Build the 1D conv stack and load weights from the provided checkpoint, including
    key normalization and optional unfreezing. Identical to train_ecg_text.py.
    """
    enc = nn.Sequential(
        nn.Conv1d(1,   32, kernel_size=4, stride=2, padding=1),  # L:256->128
        nn.ReLU(inplace=True),
        nn.Conv1d(32,  64, kernel_size=4, stride=2, padding=1),  # L:128->64
        nn.ReLU(inplace=True),
        nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # L:64->32
        nn.ReLU(inplace=True),
        nn.Conv1d(128, 4,  kernel_size=4, stride=2, padding=1),  # L:32->16, C:4
        nn.ReLU(inplace=True),
    ).to(device=device, dtype=torch.float32)

    if conv_ckpt_path:
        ckpt = torch.load(conv_ckpt_path, map_location="cpu")
        raw_sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        norm_sd: Dict[str, torch.Tensor] = {}
        for k, v in raw_sd.items():
            kk = k
            if kk.startswith("module."):
                kk = kk[len("module."):]
            if kk.startswith("_orig_mod."):
                kk = kk[len("_orig_mod."):]
            if kk.startswith("enc."):
                kk = kk[len("enc."):]
            norm_sd[kk] = v
        wanted = {f"{i}.{w}" for i in (0, 2, 4, 6) for w in ("weight", "bias")}
        conv_sd = {k: v for k, v in norm_sd.items() if k in wanted}
        missing, unexpected = enc.load_state_dict(conv_sd, strict=True)
        if missing or unexpected:
            print(f"[conv load] missing={list(missing)} unexpected={list(unexpected)}")
    enc.eval()
    return enc

def build_wrapper(
    *,
    wrapper_cls: Type[nn.Module] = ECGModelPrefix,
    language_model: nn.Module,
    conv_encoder: nn.Module,
    hidden_size: int,
    num_ecg_special_tokens: int,
    dtype: torch.dtype,
    enc_out_dim: int = 64,
    freeze_encoder: bool = True,
    inference: bool = False,
    projector_name: str = "linear",
) -> ECGModelPrefix:
    """Construct the ECG-language wrapper (keeps default wrapper class)."""
    wrapper = wrapper_cls(
        language_model,
        enc=conv_encoder,
        hidden_size=hidden_size,
        num_ecg_special_tokens=num_ecg_special_tokens,
        dtype=dtype,
        enc_out_dim=enc_out_dim,
        freeze_encoder=freeze_encoder,
        inference=inference,
        projector_name=projector_name,
    )
    return wrapper


__all__ = [
    "attach_lora",
    "build_conv_encoder",
    "build_wrapper",
]
