"""Dependency-light deterministic helpers for resumable NEPAL campaigns."""

from __future__ import annotations

import random

import numpy as np


def seed_nepal_runtime(seed: int) -> None:
    resolved = int(seed)
    random.seed(resolved)
    np.random.seed(resolved % (2**32 - 1))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
