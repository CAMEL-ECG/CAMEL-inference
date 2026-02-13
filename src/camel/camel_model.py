#!/usr/bin/env python3
"""Compatibility wrapper around inference.KardiaLM."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional
import numpy as np
import torch

from inference import KardiaLM
from process_ecg import get_waveform

class CAMEL:
    def __init__(
        self,
        device,
        ckpt: str,
        model_config_name: str = 'medgemma-4b',
        conv_ckpt: Optional[str] = None,
        no_lora: bool = False,
        mask_strategy: str = 'semantic',
        **model_args,
    ) -> None:
        default_top_k = model_args.pop("default_top_k", 64)
        default_top_p = float(model_args.pop("default_top_p", 0.95))
        default_min_p = float(model_args.pop("default_min_p", 0.0))

        self.session = KardiaLM(
            hf_model_id_override=None,
            model_config_name=model_config_name,
            adapter_ckpt=ckpt,
            conv_ckpt=conv_ckpt,
            no_lora=no_lora,
            default_max_new_tokens=int(model_args.pop("default_max_new_tokens", 1000)),
            default_temperature=float(model_args.pop("default_temperature", 1.0)),
            default_top_k=None if default_top_k is None else int(default_top_k),
            default_top_p=default_top_p,
            default_min_p=default_min_p,
            mask_strategy=mask_strategy,
            device=device
        )
        self.prompt_tokens = self.session.packing_schema.prompt
        self.device = self.session.device

    def run(self, data, input_text, args):
        generate_kwargs = dict(
            data=data,
            input_text=input_text,
            max_length=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            device=self.device,
        )
        if self.top_p is not None:
            generate_kwargs["top_p"] = self.top_p
        if self.model.__class__.__name__ == "OurModel":
            if self.top_k is not None:
                generate_kwargs["top_k"] = self.top_k
            if self.min_p is not None:
                generate_kwargs["min_p"] = self.min_p
        generated_text = self.generate( **generate_kwargs)
        return generated_text

    def generate(
        self,
        data: Any,
        input_text: Optional[str],
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: Optional[int] = 64,
        top_p: float = 0.95,
        min_p: float = 0.0,
    ) -> str:
        device = device or self.session.device
        content = self._build_content(data=data, input_text=input_text, device=device)
        conversation = [{"from": self.prompt_tokens.user_role, "content": content}]
        
        text, _ = self.session.chat(
            conversation=conversation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
        return text

    def _build_content(self, *, data: Any, input_text: Optional[str], device) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if input_text:
            content.append({"type": "text", "text": input_text})
        for d in data:
            d = get_waveform(d)
            waveform = self._prepare_waveform(d, device=device)
            content.append({"type": "ecg", "waveform_segments": waveform})
        return content

    def _prepare_waveform(self, data: Any, *, device) -> Any:
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=device, dtype=torch.float32)
            default_lead = self.session.packing_schema.ecg.canonical_leads[0] if self.session.packing_schema.ecg.canonical_leads else "lead"
            return OrderedDict({default_lead: tensor})
        if isinstance(data, np.ndarray):
            tensor = torch.as_tensor(data, dtype=torch.float32, device=device)
            default_lead = self.session.packing_schema.ecg.canonical_leads[0] if self.session.packing_schema.ecg.canonical_leads else "lead"
            return OrderedDict({default_lead: tensor})
        if isinstance(data, dict) or hasattr(data, "items"):
            cleaned = OrderedDict()
            for ld, tensor in data.items():
                t = torch.as_tensor(tensor, dtype=torch.float32, device=device)
                cleaned[ld] = t
            return cleaned
        raise TypeError(f"Unsupported waveform input type: {type(data)}")

__all__ = ["CAMEL"]
