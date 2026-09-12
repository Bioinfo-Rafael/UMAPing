"""Central random-seed control for Python, NumPy, and Torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch (CPU + all CUDA devices) from one value.

    ``deterministic=True`` additionally asks cuDNN to use deterministic
    algorithms. This can slow some GPU kernels down but keeps reference
    trajectories and checkpoints reproducible across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """A standalone torch.Generator, for sampling that must not disturb the
    global RNG stream (e.g. negative sampling inside a training loop that
    should be resumable independent of unrelated RNG consumption elsewhere).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
