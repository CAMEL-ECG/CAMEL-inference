"""Model introspection utilities driven by registry hints.

Centralizes resolution of model-internal structures to avoid hardcoded
attribute names in call sites. Use the hint paths defined in the model
registry to locate transformer layers and config attributes.
"""
from __future__ import annotations

from typing import List, Optional, Sequence
import torch.nn as nn

def _walk_attr_path(root: object, dotted_path: str) -> Optional[object]:
    cur: object = root
    for part in dotted_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur

def resolve_layers(model: nn.Module, path_hints: Sequence[str]) -> List[nn.Module]:
    """
    Resolve the text transformer layer sequence using the first successful
    dotted path from `path_hints` relative to common roots.

    We try against several candidate roots to be robust to wrappers (e.g., PEFT):
      - the model itself
      - model.base_model (if present)
      - model.base_model.model (if present)

    We also try a small set of generic fallback hints ("model.language_model.layers",
    "language_model.layers", "model.layers", "layers") if the provided hints fail.
    """
    roots: List[object] = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        roots.append(base)
        base_model_attr = getattr(base, "model", None)
        if base_model_attr is not None:
            roots.append(base_model_attr)

    tried: List[str] = []
    def _try_hints(root: object, hints: Sequence[str]) -> Optional[List[nn.Module]]:
        for hint in hints:
            tried.append(hint)
            obj = _walk_attr_path(root, hint)
            if obj is None:
                continue
            if isinstance(obj, (list, tuple)) and all(isinstance(x, nn.Module) for x in obj):
                return list(obj)
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                try:
                    seq = list(obj)
                    if seq and all(isinstance(x, nn.Module) for x in seq):
                        return seq
                except Exception:
                    pass
        return None

    # Try provided hints against all roots
    for root in roots:
        found = _try_hints(root, path_hints)
        if found is not None:
            return found

    # Fallback generic hints against all roots
    generic_hints = (
        "model.language_model.layers",
        "language_model.layers",
        "model.layers",
        "layers",
    )
    for root in roots:
        found = _try_hints(root, generic_hints)
        if found is not None:
            return found

    raise RuntimeError(
        f"Could not resolve transformer layers via provided hints: {list(path_hints)}"
    )

def resolve_hidden_size(model: nn.Module, attr_paths: Sequence[str]) -> int:
    """Resolve hidden size via the first successful dotted config attribute path."""
    for path in attr_paths:
        val = _walk_attr_path(model, path)
        if isinstance(val, (int, float)):
            return int(val)
    raise AttributeError(
        f"Could not resolve hidden size from any of: {list(attr_paths)}"
    )


__all__ = [
    "resolve_layers",
    "resolve_hidden_size",
]
