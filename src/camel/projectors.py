"""Projector registry utilities.

Provides a lightweight mechanism to swap the adapter architecture that maps the
conv encoder output to the language-model hidden size. Mirrors the ergonomic
API used by loss.py: a registry, default implementations, and a simple factory.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable
import torch.nn as nn

ProjectorBuilder = Callable[[int, int], nn.Module]

_PROJECTOR_REGISTRY: Dict[str, ProjectorBuilder] = {}

def register_projector(name: str) -> Callable[[ProjectorBuilder], ProjectorBuilder]:
    """Decorator to register a projector builder under a unique name."""
    key = name.strip().lower()

    def _decorator(fn: ProjectorBuilder) -> ProjectorBuilder:
        if not callable(fn):
            raise TypeError("Projector builder must be callable.")
        if key in _PROJECTOR_REGISTRY:
            raise ValueError(f"Projector '{name}' is already registered.")
        _PROJECTOR_REGISTRY[key] = fn
        return fn

    return _decorator

@register_projector("linear")
def _linear_projector(in_dim: int, out_dim: int) -> nn.Module:
    """Single linear adapter (current default)."""
    return nn.Linear(in_dim, out_dim, bias=True)

def available_projectors() -> Iterable[str]:
    """Return sorted projector names."""
    return sorted(_PROJECTOR_REGISTRY.keys())

def build_projector(name: str, in_dim: int, out_dim: int) -> nn.Module:
    """Instantiate a registered projector."""
    if not _PROJECTOR_REGISTRY:
        raise RuntimeError("No projectors registered.")
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("Projector name must be a non-empty string.")
    builder = _PROJECTOR_REGISTRY.get(key)
    if builder is None:
        raise KeyError(
            f"Unknown projector '{name}'. Available: {', '.join(available_projectors())}"
        )
    return builder(int(in_dim), int(out_dim))

__all__ = [
    "ProjectorBuilder",
    "available_projectors",
    "build_projector",
]
