"""Safe checkpoint loading helpers for the active research pipeline."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def load_checkpoint(path, *, map_location="cpu"):
    """Load a state-dict checkpoint with PyTorch's restricted unpickler.

    PyTorch versions before this hardening stored ``torch.__version__`` as a
    ``TorchVersion`` instance inside the resolved config.  Allow-list only
    that harmless compatibility type; model objects and arbitrary globals
    remain disallowed.
    """
    try:
        from torch.torch_version import TorchVersion

        context = torch.serialization.safe_globals([TorchVersion])
    except (AttributeError, ImportError):
        context = nullcontext()

    with context:
        return torch.load(path, map_location=map_location, weights_only=True)
