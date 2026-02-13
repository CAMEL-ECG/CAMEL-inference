"""ECG-aware attention mask utilities with pluggable strategy support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol
import torch

@dataclass
class ECGBlockLayout:
    """Layout describing a single ECG block inside the assembled sequence."""

    start_idx: Optional[int]
    end_idx_exclusive: Optional[int]
    global_start_idx: Optional[int] = None
    global_end_idx: Optional[int] = None
    lead_start_idx: Dict[str, int] = field(default_factory=dict)
    lead_end_idx: Dict[str, int] = field(default_factory=dict)
    signal_pos_by_lead: Dict[str, List[int]] = field(default_factory=dict)
    time_to_signal_idxs: Dict[int, List[int]] = field(default_factory=dict)
    special_idxs_sorted: List[int] = field(default_factory=list)
    signal_pos_list: List[int] = field(default_factory=list)
    declared_segments_per_lead: Dict[str, int] = field(default_factory=dict)
    conv_idxs: List[int] = field(default_factory=list)


@dataclass
class ECGSequenceLayout:
    """Compact description of the assembled token layout for one training sample."""

    seq_len: int
    text_idxs: List[int] = field(default_factory=list)
    blocks: List[ECGBlockLayout] = field(default_factory=list)


def _as_tensor(indices: List[int], device: torch.device) -> torch.Tensor:
    if not indices:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(sorted(indices), dtype=torch.long, device=device)


def apply_single_block_semantic_mask_(
    allowed: torch.Tensor,
    block_layout: ECGBlockLayout,
    *,
    visible_prefix_len: int,
    key_limit_exclusive: int,
    apply_header_causal: bool = True,
) -> None:
    """
    In-place semantic mask update for a single ECG block. Mirrors the historical
    single-block logic, but operates on a provided boolean mask.
    """
    L = int(allowed.size(0))
    device = allowed.device
    header = (
        torch.arange(int(visible_prefix_len), dtype=torch.long, device=device)
        if int(visible_prefix_len) > 0
        else torch.empty(0, dtype=torch.long, device=device)
    )

    specials_list = block_layout.special_idxs_sorted or []
    if not specials_list:
        specials_list = sorted(
            ([block_layout.global_start_idx] if block_layout.global_start_idx is not None else [])
            + list(block_layout.lead_start_idx.values())
            + list(block_layout.lead_end_idx.values())
            + ([block_layout.global_end_idx] if block_layout.global_end_idx is not None else [])
        )
        block_layout.special_idxs_sorted = specials_list
    if key_limit_exclusive is not None:
        specials_list = [i for i in specials_list if int(i) < int(key_limit_exclusive)]
    specials = _as_tensor([int(i) for i in specials_list], device)

    signals_list = block_layout.signal_pos_list or []
    if not signals_list:
        signal_all: List[int] = []
        for lst in block_layout.signal_pos_by_lead.values():
            signal_all.extend(lst)
        signals_list = sorted(signal_all)
        block_layout.signal_pos_list = signals_list
    if key_limit_exclusive is not None:
        signals_list = [i for i in signals_list if int(i) < int(key_limit_exclusive)]
    signals = _as_tensor([int(i) for i in signals_list], device)

    lead_starts = _as_tensor(list(block_layout.lead_start_idx.values()), device)
    lead_ends = _as_tensor(list(block_layout.lead_end_idx.values()), device)

    if apply_header_causal and header.numel():
        allowed[header[:, None], header[None, :]] = header[:, None] >= header[None, :]

    gs = block_layout.global_start_idx
    if gs is not None and header.numel():
        allowed[int(gs), header] = True

    rows_before = []
    if lead_starts.numel():
        rows_before.append(lead_starts)
    if signals.numel():
        rows_before.append(signals)
    if lead_ends.numel():
        rows_before.append(lead_ends)
    rows_before_t = (
        torch.cat(rows_before, dim=0) if rows_before else torch.empty(0, dtype=torch.long, device=device)
    )

    if rows_before_t.numel():
        if header.numel():
            allowed[rows_before_t[:, None], header[None, :]] = True
        if specials.numel():
            allowed[rows_before_t[:, None], specials[None, :]] = specials[None, :] < rows_before_t[:, None]

    ttsi: Dict[int, List[int]] = block_layout.time_to_signal_idxs
    if signals.numel() and ttsi:
        pos_min_time: Dict[int, int] = {}
        pos_to_time: Dict[int, int] = {}
        for t, idxs in ttsi.items():
            for p in idxs:
                pos_to_time[p] = t
                prev = pos_min_time.get(p)
                if prev is None or t < prev:
                    pos_min_time[p] = t

        u_pos_list = sorted(pos_min_time.keys())
        if u_pos_list:
            u_pos = torch.tensor(u_pos_list, dtype=torch.long, device=device)
            u_time = torch.tensor([pos_min_time[p] for p in u_pos_list], dtype=torch.long, device=device)
            q_time = torch.tensor([pos_to_time.get(p, 0) for p in signals_list], dtype=torch.long, device=device)
            allowed[signals[:, None], u_pos[None, :]] = (u_time[None, :] <= q_time[:, None])

    for lead, eidx in block_layout.lead_end_idx.items():
        lead_sigs = block_layout.signal_pos_by_lead.get(lead, [])
        if lead_sigs:
            allowed[int(eidx), torch.tensor(lead_sigs, dtype=torch.long, device=device)] = True

    ge = block_layout.global_end_idx
    if ge is not None:
        gei = int(ge)
        if header.numel():
            allowed[gei, header] = True
        if specials.numel():
            allowed[gei, specials] = True
        if signals.numel():
            allowed[gei, signals] = True

    conv = _as_tensor(block_layout.conv_idxs, device)
    if conv.numel():
        allowed[conv[:, None], conv[None, :]] = conv[:, None] >= conv[None, :]
        if header.numel():
            allowed[conv[:, None], header[None, :]] = True
        if specials.numel():
            allowed[conv[:, None], specials[None, :]] = True
        if signals.numel():
            allowed[conv[:, None], signals[None, :]] = True
        cols = torch.arange(L, device=device)
        conv_rows = allowed[conv, :]
        conv_rows &= (cols.unsqueeze(0) <= conv.unsqueeze(1))
        allowed[conv, :] = conv_rows

    if specials.numel():
        allowed[specials, specials] = True

    if key_limit_exclusive is not None and int(key_limit_exclusive) < L:
        block_rows_list = list(specials_list) + list(signals_list)
        if block_rows_list:
            block_rows = _as_tensor(block_rows_list, device)
            allowed[block_rows, int(key_limit_exclusive):] = False

@dataclass
class MaskBuildResult:
    """Container for per-sample mask artifacts produced by a strategy."""

    additive: torch.Tensor
    boolean: Optional[torch.Tensor] = None


class ECGMaskStrategy(Protocol):
    """Protocol for strategies that build and update per-sample attention masks."""

    name: str

    def build(
        self,
        layout: ECGSequenceLayout,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MaskBuildResult:
        ...

    def update_for_generated_token(
        self,
        layout: ECGSequenceLayout,
        *,
        device: torch.device,
        dtype: torch.dtype,
        previous: MaskBuildResult,
    ) -> MaskBuildResult:
        ...


class SemanticMaskStrategy:
    """Default strategy reproducing the historical ECG-aware attention mask."""

    name = "semantic"

    def build(
        self,
        layout: ECGSequenceLayout,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> MaskBuildResult:
        L = int(layout.seq_len)
        if L <= 0:
            raise ValueError("Semantic mask requires a positive sequence length")
        allowed = torch.tril(torch.ones((L, L), dtype=torch.bool, device=device))
        multi_block = len(layout.blocks) > 1
        for block in layout.blocks:
            if block.start_idx is None or block.end_idx_exclusive is None:
                continue
            block_rows: List[int] = []
            specials_list = block.special_idxs_sorted or []
            if not specials_list:
                specials_list = sorted(
                    ([block.global_start_idx] if block.global_start_idx is not None else [])
                    + list(block.lead_start_idx.values())
                    + list(block.lead_end_idx.values())
                    + ([block.global_end_idx] if block.global_end_idx is not None else [])
                )
                block.special_idxs_sorted = specials_list
            signals_list = block.signal_pos_list or []
            if not signals_list:
                signal_all: List[int] = []
                for lst in block.signal_pos_by_lead.values():
                    signal_all.extend(lst)
                signals_list = sorted(signal_all)
                block.signal_pos_list = signals_list
            if specials_list:
                block_rows.extend(int(i) for i in specials_list)
            if signals_list:
                block_rows.extend(int(i) for i in signals_list)
            if block_rows:
                rows = torch.tensor(sorted(set(block_rows)), dtype=torch.long, device=device)
                allowed[rows, :] = False
            apply_single_block_semantic_mask_(
                allowed,
                block,
                visible_prefix_len=int(block.start_idx),
                key_limit_exclusive=int(block.end_idx_exclusive),
                apply_header_causal=not multi_block,
            )
        additive = self._boolean_to_additive(allowed, device=device, dtype=dtype)
        return MaskBuildResult(additive=additive, boolean=allowed)

    def update_for_generated_token(
        self,
        layout: ECGSequenceLayout,
        *,
        device: torch.device,
        dtype: torch.dtype,
        previous: MaskBuildResult,
    ) -> MaskBuildResult:
        allowed = previous.boolean
        if allowed is None:
            return self.build(layout, device=device, dtype=dtype)
        prev_len = int(allowed.size(0))
        new_allowed = torch.zeros((prev_len + 1, prev_len + 1), dtype=torch.bool, device=device)
        new_allowed[:prev_len, :prev_len] = allowed
        new_allowed[prev_len, : prev_len + 1] = True
        additive = self._boolean_to_additive(new_allowed, device=device, dtype=dtype)
        return MaskBuildResult(additive=additive, boolean=new_allowed)


    @classmethod
    def _build_boolean_mask(cls, layout: "ECGSequenceLayout", device: torch.device) -> torch.Tensor:
        if len(layout.blocks) != 1:
            raise ValueError("Single-block mask builder requires exactly one ECG block")
        L = int(layout.seq_len)
        allowed = torch.zeros((L, L), dtype=torch.bool, device=device)
        block = layout.blocks[0]
        prefix_len = int(block.start_idx or 0)
        end_idx = int(block.end_idx_exclusive or L)
        conv_idxs = [idx for idx in layout.text_idxs if int(idx) >= end_idx]
        block_ref = ECGBlockLayout(
            start_idx=block.start_idx,
            end_idx_exclusive=block.end_idx_exclusive,
            global_start_idx=block.global_start_idx,
            global_end_idx=block.global_end_idx,
            lead_start_idx=dict(block.lead_start_idx),
            lead_end_idx=dict(block.lead_end_idx),
            signal_pos_by_lead=dict(block.signal_pos_by_lead),
            time_to_signal_idxs=dict(block.time_to_signal_idxs),
            special_idxs_sorted=list(block.special_idxs_sorted),
            signal_pos_list=list(block.signal_pos_list),
            declared_segments_per_lead=dict(block.declared_segments_per_lead),
            conv_idxs=conv_idxs,
        )
        apply_single_block_semantic_mask_(
            allowed,
            block_ref,
            visible_prefix_len=prefix_len,
            key_limit_exclusive=end_idx,
        )
        return allowed

    @staticmethod
    def _boolean_to_additive(allowed: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        additive = torch.zeros(allowed.shape, dtype=dtype, device=device)
        additive.masked_fill_(~allowed, float("-inf"))
        return additive


DEFAULT_MASK_STRATEGY = SemanticMaskStrategy()


MASK_STRATEGY_REGISTRY: Dict[str, ECGMaskStrategy] = {
    DEFAULT_MASK_STRATEGY.name: DEFAULT_MASK_STRATEGY,
}


def get_mask_strategy(name: Optional[str]) -> ECGMaskStrategy:
    """Resolve a registered mask strategy by name (case-insensitive)."""
    if name is None:
        return DEFAULT_MASK_STRATEGY
    key = str(name).lower()
    try:
        return MASK_STRATEGY_REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(MASK_STRATEGY_REGISTRY))
        raise ValueError(f"Unknown ECG mask strategy '{name}'. Known strategies: {known}") from exc


__all__ = [
    "ECGBlockLayout",
    "ECGSequenceLayout",
    "MaskBuildResult",
    "ECGMaskStrategy",
    "SemanticMaskStrategy",
    "DEFAULT_MASK_STRATEGY",
    "MASK_STRATEGY_REGISTRY",
    "get_mask_strategy",
]
