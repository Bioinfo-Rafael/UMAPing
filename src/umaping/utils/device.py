"""Device resolution shared by CLI commands and training/inference code."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve a user-facing device string ("auto", "cpu", "cuda", "cuda:0", ...).

    "auto" picks CUDA when available, otherwise CPU. An explicit "cuda" request
    on a machine without CUDA raises rather than silently falling back, since
    that almost always indicates a misconfigured remote environment.
    """
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Resolved device 'auto' -> %s", device)
        return device

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device '{requested}' but torch.cuda.is_available() is False."
        )
    return device
