"""
Abstract ECG-language model adapter with shared conv→projection logic.

Subclasses can override only the pieces that differ (e.g., prompt format or
language-model forwarding) while inheriting the common ECG prefix handling.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from camel.assertions import assert_wrapper_embed_length, assert_rest_length_nonnegative, assert_tensor_dtype
from camel.projectors import build_projector

class ECGNonFiniteInputError(RuntimeError):
    """Raised when the conv encoder input contains NaN or Inf values."""

    def __init__(self, sample_idx: int, lead: Optional[str] = None) -> None:
        self.sample_idx = int(sample_idx)
        self.lead = lead
        lead_part = f", lead={lead}" if lead is not None else ""
        super().__init__(f"Non-finite waveform detected (sample_idx={self.sample_idx}{lead_part})")


class ECGLanguageModelWrapper(nn.Module):
    """
    Turns per-lead ECG waveforms into prefix embeddings consumable by a language model.
    Stores the ECG encoder, trainable adapter, and the target language model.
    """

    def __init__(
        self,
        *,
        language_model: nn.Module,
        conv_encoder: nn.Module,
        hidden_size: int,
        num_ecg_special_tokens: int,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        enc_out_dim: int = 64,
        freeze_encoder: bool = True,
        inference: bool = False,
        projector_name: str = "linear",
    ) -> None:
        super().__init__()

        if int(num_ecg_special_tokens) <= 0:
            raise ValueError("num_ecg_special_tokens must be positive")

        # Keep LM in train mode (actual freezing handled by caller).
        self.language_model = language_model.train()

        # Conv encoder may be frozen or trainable depending on configuration.
        self.enc = conv_encoder
        if freeze_encoder:
            self.enc = self.enc.eval()
            for p in self.enc.parameters():
                p.requires_grad = False
        else:
            self.enc = self.enc.train()
            for p in self.enc.parameters():
                p.requires_grad = True

        self.hidden_size = int(hidden_size)
        self.dtype = dtype or torch.bfloat16
        self.num_ecg_special_tokens = int(num_ecg_special_tokens)
        self.inference = bool(inference)
        self.projector_name = str(projector_name or "linear")

        # Trainable adapter: conv output → LM hidden size (kept in fp32 for stability).
        self.llava_proj = build_projector(self.projector_name, int(enc_out_dim), self.hidden_size)
        if self.inference:
            self.llava_proj.to(dtype=torch.float32)

        self.ecg_special_embed = nn.Embedding(self.num_ecg_special_tokens, self.hidden_size)
        self.ecg_special_embed.to(dtype=self.dtype)
        if self.inference:
            self.enc.to(dtype=torch.float32)
        self._grad_ckpt_enabled = self._detect_grad_ckpt_state()

    def _projector_param_dtype(self) -> torch.dtype:
        """Return dtype of the first projector parameter (defaults to fp32)."""
        first_param = next(self.llava_proj.parameters(), None)
        if first_param is None:
            return torch.float32
        return first_param.dtype

    # ---- Gradient checkpointing --------------------------------------------------------------

    def _detect_grad_ckpt_state(self) -> bool:
        if hasattr(self.language_model, "is_gradient_checkpointing"):
            try:
                return bool(self.language_model.is_gradient_checkpointing)
            except Exception:
                return False
        if hasattr(self.language_model, "gradient_checkpointing"):
            try:
                return bool(self.language_model.gradient_checkpointing)
            except Exception:
                return False
        return False

    def set_gradient_checkpointing(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        current = self._detect_grad_ckpt_state()
        if current == enabled:
            self._grad_ckpt_enabled = enabled
            return False
        if enabled:
            if hasattr(self.language_model, "gradient_checkpointing_enable"):
                self.language_model.gradient_checkpointing_enable()
            if hasattr(getattr(self.language_model, "config", None), "use_cache"):
                self.language_model.config.use_cache = False
        else:
            if hasattr(self.language_model, "gradient_checkpointing_disable"):
                self.language_model.gradient_checkpointing_disable()
            elif hasattr(self.language_model, "gradient_checkpointing"):
                try:
                    self.language_model.gradient_checkpointing = False
                except Exception:
                    pass
            if hasattr(getattr(self.language_model, "config", None), "use_cache"):
                self.language_model.config.use_cache = True
        self._grad_ckpt_enabled = enabled
        return True

    def enable_gradient_checkpointing(self) -> None:
        self.set_gradient_checkpointing(True)

    def disable_gradient_checkpointing(self) -> None:
        self.set_gradient_checkpointing(False)

    def is_gradient_checkpointing_enabled(self) -> bool:
        return bool(self._grad_ckpt_enabled)

    # ---- Token helpers -----------------------------------------------------------------------

    def tokens_to_embeds(self, input_embedder: nn.Embedding, ids: List[int], device: torch.device) -> Tensor:
        ids_t = torch.tensor(ids, dtype=torch.long, device=device)
        embeddings = input_embedder(ids_t)
        embeddings = embeddings.to(dtype=self.dtype)
        # Defensive: enforce 1:1 mapping between ids and embeddings
        assert_wrapper_embed_length(embeddings=embeddings, ids=ids, context="tokens_to_embeds")
        return embeddings

    def ecg_special_tokens_to_embeds(self, indices: torch.Tensor | List[int], device: torch.device) -> Tensor:
        if torch.is_tensor(indices):
            idx = indices.to(device=device, dtype=torch.long)
        else:
            idx = torch.tensor(indices, dtype=torch.long, device=device)
        embeds = self.ecg_special_embed(idx)
        return embeds.to(dtype=self.dtype)

    # ---- ECG prefix encoding -----------------------------------------------------------------

    def ecg_prefix(
        self,
        waveform_segments: Dict[str, Tensor],
        device: torch.device,
        lead_order: Optional[List[str]] = None,
    ) -> Tensor:
        """Encode a single sample's ECG prefix in a deterministic lead order.

        Args:
            waveform_segments: Mapping lead -> [T,256] windows.
            device: Target device for encoder input.
            lead_order: Optional explicit order of leads to iterate. If provided,
                segments are concatenated in this exact order; otherwise relies on
                insertion order of the mapping.
        """
        seqs: List[Tensor] = []
        if lead_order is None:
            items = list(waveform_segments.items())
        else:
            items = [(ld, waveform_segments[ld]) for ld in lead_order if ld in waveform_segments]
            # Validate presence when an explicit order is provided
            missing = [ld for ld in lead_order if ld not in waveform_segments]
            if missing:
                raise ValueError(f"Missing leads in waveform_segments for requested order: {missing}")
        for lead, seg in items:
            seg = torch.as_tensor(seg)
            if seg.ndim != 2 or seg.shape[1] != 256:
                raise ValueError(f"Waveform for lead {lead} must be [T,256], got {seg.shape}")
            seqs.append(seg)
        x = torch.cat(seqs, dim=0)
        x = x.to(device=device, dtype=torch.float32)
        x = x.unsqueeze(1)  # [P, 1, 256]

        enc_trainable = any(p.requires_grad for p in self.enc.parameters())
        ctx = torch.enable_grad() if enc_trainable else torch.no_grad()
        with ctx:
            z = self.enc(x)  # [P, C, L]
        if self.inference:
            z = z.to(dtype=torch.float32)
        assert_tensor_dtype(z, expected=torch.float32, context="conv encoder output (single)")
        self.ensure_finite(z, "conv encoder output")
        z = z.flatten(1)  # [P, 64] for conv stack

        proj_dtype = self._projector_param_dtype()
        if z.dtype != proj_dtype:
            z = z.to(dtype=proj_dtype)
        y = self.llava_proj(z)
        if self.inference:
            y = y.to(dtype=torch.float32)
        assert_tensor_dtype(y, expected=torch.float32, context="llava_proj output (single)")
        return y.to(dtype=self.dtype)

    def ecg_prefix_batch(
        self,
        waveform_segments_batch: List[Dict[str, Tensor]],
        device: torch.device,
        lead_orders: Optional[List[List[str]]] = None,
    ) -> Tuple[Tensor, List[int]]:
        """Encode a batch of ECG prefixes with explicit per-sample lead orders.

        Args:
            waveform_segments_batch: List of dicts; each maps lead -> [T,256].
            device: Target device for encoder input.
            lead_orders: Optional list of lead-order lists, one per sample. If
                provided, each sample's segments are concatenated in that exact
                order; otherwise relies on insertion order of each mapping.
        """
        all_seqs: List[Tensor] = []
        prefix_lengths: List[int] = []

        for i, wv_dict in enumerate(waveform_segments_batch):
            seqs: List[Tensor] = []
            leads_for_sample: List[str] = []
            if lead_orders is not None and i < len(lead_orders) and lead_orders[i] is not None:
                order = lead_orders[i]
                missing = [ld for ld in order if ld not in wv_dict]
                if missing:
                    raise ValueError(f"Missing leads for sample {i}: {missing}")
                items = [(ld, wv_dict[ld]) for ld in order]
            else:
                items = list(wv_dict.items())
            for lead, seg in items:
                seg = torch.as_tensor(seg)
                if seg.ndim != 2 or seg.shape[1] != 256:
                    raise ValueError(f"Waveform for lead {lead} must be [T,256], got {seg.shape}")
                seqs.append(seg)
                leads_for_sample.append(str(lead))
            sample_segments = torch.cat(seqs, dim=0)
            if not torch.isfinite(sample_segments).all().item():
                bad_lead = None
                for lead_name, seg_tensor in zip(leads_for_sample, seqs):
                    if not torch.isfinite(seg_tensor).all().item():
                        bad_lead = lead_name
                        break
                raise ECGNonFiniteInputError(sample_idx=i, lead=bad_lead)
            all_seqs.append(sample_segments)
            prefix_lengths.append(sample_segments.size(0))

        x = torch.cat(all_seqs, dim=0)
        x = x.to(device=device, dtype=torch.float32).unsqueeze(1)

        enc_trainable = any(p.requires_grad for p in self.enc.parameters())
        ctx = torch.enable_grad() if enc_trainable else torch.no_grad()
        with ctx:
            z = self.enc(x)
        if self.inference:
            z = z.to(dtype=torch.float32)
        assert_tensor_dtype(z, expected=torch.float32, context="conv encoder output (batch)")
        self.ensure_finite(z, "conv encoder output (batch)")
        z = z.flatten(1)

        proj_dtype = self._projector_param_dtype()
        if z.dtype != proj_dtype:
            z = z.to(dtype=proj_dtype)
        y = self.llava_proj(z)
        if self.inference:
            y = y.to(dtype=torch.float32)
        assert_tensor_dtype(y, expected=torch.float32, context="llava_proj output (batch)")
        return y.to(dtype=self.dtype), prefix_lengths

    # ---- Language-model forward --------------------------------------------------------------

    def forward_language_model(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor],
    ):
        """
        Default HF-style forward call; subclasses can override for custom behavior.
        """
        embedder_fn = getattr(self.language_model, "get_input_embeddings", None)
        if callable(embedder_fn):
            embed_module = embedder_fn()
            if not hasattr(embed_module, "weight"):
                raise RuntimeError("Language model embeddings missing weight parameter; cannot infer device.")
            target_device = embed_module.weight.device
        else:
            try:
                first_param = next(self.language_model.parameters())
            except StopIteration as exc:
                raise RuntimeError("Language model exposes no parameters to infer device placement.") from exc
            target_device = first_param.device

        if inputs_embeds.device != target_device:
            inputs_embeds = inputs_embeds.to(target_device)
        if attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)
        if labels is not None and labels.device != target_device:
            labels = labels.to(target_device)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

    # ---- Label helpers ----------------------------------------------------------------------

    @staticmethod
    def build_labels_from_lengths(
        *,
        ids_rest: List[int],
        model_spans_in_rest: List[Tuple[int, int]],
        total_len: int,
        offset_rest: int,
    ) -> Tensor:
        """Build label tensor using explicit offsets to satisfy any packing schema.

        Args:
            ids_rest: Token ids for the supervised span of the prompt.
            model_spans_in_rest: List of (start, end) spans (relative to ids_rest) to supervise.
            total_len: Total sequence length of the assembled prompt.
            offset_rest: Absolute position in the sequence where `ids_rest` begins.

        Returns:
            Tensor of shape (total_len,) with supervised ids placed according to spans; all other
            positions are filled with -100.

        Raises:
            ValueError: if `offset_rest` is invalid or spans fall outside the provided bounds.
        """
        labels = torch.full((total_len,), fill_value=-100, dtype=torch.long)

        offset_rest = int(offset_rest)
        if offset_rest < 0 or offset_rest > total_len:
            raise ValueError(f"Invalid rest offset {offset_rest} for sequence length {total_len}.")
        rest_len = len(ids_rest)
        if offset_rest + rest_len > total_len:
            raise ValueError(
                f"Rest tokens (len={rest_len}) exceed total_len {total_len} with offset {offset_rest}."
            )
        # Defensive: no negative rest length (should not happen, but keep invariant tight)
        assert_rest_length_nonnegative(rest_length=rest_len)

        for (s, e) in model_spans_in_rest:
            if not (0 <= s <= e <= rest_len):
                raise ValueError(
                    f"Model span {(s, e)} is out of bounds for ids_rest length {rest_len}."
                )
            s_abs = offset_rest + s
            e_abs = offset_rest + e
            if not (0 <= s_abs <= e_abs <= total_len):
                raise ValueError(
                    f"Model span {(s, e)} with rest offset {offset_rest} is out of bounds for length {total_len}."
                )
            labels[s_abs:e_abs] = torch.tensor(ids_rest[s:e], dtype=torch.long)
        return labels

    # ---- Trainable summaries -----------------------------------------------------------------

    def summarize_trainables(self) -> Mapping[str, int]:
        def _count(params: Iterable[nn.Parameter]) -> int:
            return sum(int(p.numel()) for p in params if p.requires_grad)

        llava_train = _count(self.llava_proj.parameters())
        ecg_special_train = _count(self.ecg_special_embed.parameters())
        enc_train = _count(self.enc.parameters())
        lm_train = _count(self.language_model.parameters())

        total = llava_train + ecg_special_train + enc_train + lm_train
        return {
            "llava_proj_trainable": llava_train,
            "ecg_special_trainable": ecg_special_train,
            "enc_trainable": enc_train,
            "lm_trainable": lm_train,
            "total_trainable": total,
        }

    # ---- Utility ---------------------------------------------------------------------------

    @staticmethod
    def ensure_finite(tensor: Tensor, context: str) -> None:
        if not torch.isfinite(tensor).all():
            xin_nan = torch.isnan(tensor).any().item()
            raise RuntimeError(f"Encountered non-finite values in {context} (input_has_nan={bool(xin_nan)}).")
