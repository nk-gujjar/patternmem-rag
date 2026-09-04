"""
patternmem._utils
~~~~~~~~~~~~~~~~~~
Internal utility helpers shared across backends.

All functions here are *pure* (no I/O, no async, no side-effects).
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity in [−1, 1] between two L2-normalised vectors.

    Returns 0.0 if either vector has zero norm (avoids division-by-zero).
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
