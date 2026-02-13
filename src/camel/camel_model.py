#!/usr/bin/env python3
"""Compatibility wrapper around inference.KardiaLM."""

from __future__ import annotations

from typing import Any, Optional
import json
import torch

from camel.inference import KardiaLM
from camel.process_ecg import get_waveform

class CAMEL:
    def __init__(
        self,
        device: torch.device,
        mode: str,
        model_config_name: str = 'medgemma-4b-it',
        conv_ckpt: Optional[str] = None,
        no_lora: bool = False,
        mask_strategy: str = 'semantic',
        **model_args,
    ) -> None:
        default_top_k = model_args.pop("default_top_k", 64)
        default_top_p = float(model_args.pop("default_top_p", 0.95))
        default_min_p = float(model_args.pop("default_min_p", 0.0))

        # Initialize model
        if mode == 'base':
            ckpt = 'checkpoints/camel_base.pt'
        elif mode == 'ecgbench':
            ckpt = 'checkpoints/camel_ecginstruct.pt'
        elif mode == 'forecast':
            ckpt = 'checkpoints/camel_forecast.pt'

        self.session = KardiaLM(
            model_registry_path=None,
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
        self.device = device

    def run(self, args):
        if args.json is None and (args.ecgs is None):
            raise ValueError("Either one of --json or --ecgs should be non-empty.")
        
        if args.json is None:
            text = args.text or ''
            raw_context = [{'type': 'text', 'text': text}]
            for ecg in args.ecgs:
                raw_context.append({'type': 'ecg', 'ecg': ecg})
        else:
            try:
                with open(args.json, "r") as f:
                    raw_context = json.load(f)
            except:
                raise ValueError(f'Failed during reading json: {args.json}')

        content = self._build_content(raw_content=raw_context, ecg_configs=args.ecg_configs)
        generate_kwargs = dict(
            content=content,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
        )
        generated_text = self.generate(**generate_kwargs)
        return generated_text

    def generate(
        self,
        content: list[dict[str, Any]],
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: Optional[int] = 64,
        top_p: float = 0.95,
        min_p: float = 0.0,
    ) -> str:
        text, prompt_preview = self.session.chat(
            conversation=content,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
        return text, prompt_preview

    def _parse_ecg_config(self, ecg_configs: Optional[list[str]], n_ecgs:int) -> tuple:
        def _parse_single_config(config):
            start_ind, end_ind, leads = None, None, None
            for field in config.split(";"):
                field = field.strip()
                if not field:
                    continue
                if ":" not in field:
                    raise ValueError(f"Invalid field: {field}. Expected key:value.")

                key, value = field.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "start":
                    start_ind = int(value)
                elif key == "end":
                    end_ind = int(value)
                elif key in ("use_leads", "leads"):
                    leads = [x.strip() for x in value.split(",") if x.strip()]
                else:
                    print(f"Ignoring the unknown key: {key}")
            return start_ind, end_ind, leads
        
        if ecg_configs is None:
            output = [None] * n_ecgs
            return output, output, output
        
        n_configs = len(ecg_configs)
        if  n_configs!= 1 and n_configs != n_ecgs:
            raise ValueError(f'Found {n_configs} ECG configs for {n_ecgs} ECG inputs. The number of config should be 1 or match the number of ECGs.')
        
        start_inds, end_inds, leads = [], [], []
        for config in ecg_configs:
            start_ind, end_ind, lead = _parse_single_config(config)
            print(f'ECG Config: {start_ind}, {end_ind}, {lead}')
            start_inds.append(start_ind)
            end_inds.append(end_ind)
            leads.append(lead)
        
        if n_configs == 1 and n_ecgs > 1:
            start_inds = start_inds * n_ecgs
            end_inds = end_inds * n_ecgs
            leads = leads * n_ecgs

        return start_inds, end_inds, leads


    def _build_content(self, *, raw_content: list[dict[str, str]], ecg_configs: Optional[list[str]]) -> list[dict[str, Any]]:
        n_ecgs = sum([True for c in raw_content if c['type'] == 'ecg'])
        starts, ends, leads = self._parse_ecg_config(ecg_configs, n_ecgs)
        ecg_ind = 0

        content: list[dict[str, Any]] = []
        for c in raw_content:
            if c['type'] == 'text':
                content.append({"type": "text", "text": c['text']})
            elif c['type'] == 'ecg':
                waveform = get_waveform(ecg_path=c['ecg'], start_sec=starts[ecg_ind], end_sec=ends[ecg_ind], leads=leads[ecg_ind], device=self.device)
                ecg_ind += 1
                content.append({"type": "ecg", "waveform_segments": waveform})
 
        conversation = [{"from": self.prompt_tokens.user_role, "content": content}]
        return conversation

__all__ = ["CAMEL"]
