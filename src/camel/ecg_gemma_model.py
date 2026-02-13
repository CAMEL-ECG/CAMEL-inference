# ecg_gemma_model.py
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM

from ecg_model_wrapper import ECGLanguageModelWrapper

class ECGGemmaPrefix(ECGLanguageModelWrapper):
    """
    Frozen: Gemma-IT (language model), 1D conv signal encoder (loaded from disk)
    Trainable: llava_proj (Linear 64 -> Gemma hidden size)
    Optionally trainable: conv encoder when explicitly requested

    This wrapper turns per-second ECG windows into single "pseudo-token" rows
    that are interleaved into user turns at embedding time.
    """

    def __init__(
        self,
        gemma: AutoModelForCausalLM,
        enc: nn.Module,
        hidden_size: int,
        num_ecg_special_tokens: int,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        enc_out_dim: int = 64,  # from the specified conv stack: 4 channels * 16 length = 64
        freeze_encoder: bool = True,
        inference: bool = False,
        projector_name: str = "linear",
    ):
        super().__init__(
            language_model=gemma,
            conv_encoder=enc,
            hidden_size=hidden_size,
            num_ecg_special_tokens=num_ecg_special_tokens,
            dtype=dtype,
            enc_out_dim=enc_out_dim,
            freeze_encoder=freeze_encoder,
            inference=inference,
            projector_name=projector_name,
        )
        # Use language_model consistently; no Gemma alias is set.

    def forward_language_model(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor],
        output_hidden_states = False,
    ):
        # Ensure inputs live on the device of the LM input embeddings when sharded
        embedder_fn = getattr(self.language_model, "get_input_embeddings", None)
        if callable(embedder_fn):
            try:
                embed_module = embedder_fn()
            except Exception as exc:
                raise RuntimeError("Failed to obtain language-model input embeddings for device inference") from exc
            if not hasattr(embed_module, "weight"):
                raise RuntimeError("Input embedding module lacks a weight parameter; cannot infer device")
            dev0 = embed_module.weight.device
        else:
            params_iter = self.language_model.parameters()
            try:
                first_param = next(params_iter)
            except StopIteration as exc:
                raise RuntimeError("Language model exposes no parameters to infer device placement") from exc
            dev0 = first_param.device

        if inputs_embeds.device != dev0:
            inputs_embeds = inputs_embeds.to(dev0)
        if attention_mask.device != dev0:
            attention_mask = attention_mask.to(dev0)
        if labels is not None and labels.device != dev0:
            labels = labels.to(dev0)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )

    
    # FSDP2 helpers removed (not used)


__all__ = ["ECGGemmaPrefix"]
