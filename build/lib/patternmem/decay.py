"""
patternmem.decay
~~~~~~~~~~~~~~~~
Pure decay/eviction functions for PatternMem pattern memory.

These are *pure functions* — no I/O, no async, no side effects.  They are
the only place in the codebase that contains the numeric constants that govern
pattern reinforcement and eviction.  Tests may monkey-patch these constants.

Constants
---------
REINFORCE_DELTA : float
    Amount added to ``decay_weight`` when a hint-augmented run improves the
    score by more than ``IMPROVEMENT_GAP``.
DECAY_DELTA : float
    Amount subtracted from ``decay_weight`` when a hint-augmented run does *not*
    improve the score by the required gap.
EVICTION_FLOOR : float
    Patterns whose ``decay_weight`` falls below this value are deleted.
IMPROVEMENT_GAP : float
    Minimum score improvement required to be counted as a reinforcement event.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module-level constants (single source of truth for decay numerics)
# ---------------------------------------------------------------------------

REINFORCE_DELTA: float = 0.1
DECAY_DELTA: float = 0.2
EVICTION_FLOOR: float = 0.1
IMPROVEMENT_GAP: float = 0.15


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_new_decay_weight(
    stored: float,
    old_score: float,
    new_score: float,
) -> float:
    """Return the updated ``decay_weight`` after comparing *old_score* to *new_score*.

    Parameters
    ----------
    stored:
        Current ``decay_weight`` of the pattern in the backend.
    old_score:
        The evaluation score stored in the pattern at write time.
    new_score:
        The evaluation score from the most recent Phase 2 run (after hint injection).

    Returns
    -------
    float
        New decay weight.  May be below ``EVICTION_FLOOR`` — the caller is
        responsible for checking ``should_evict`` and triggering deletion.

    Notes
    -----
    The formula mirrors the spec exactly::

        if new_score > stored_pattern.score + 0.15:
            decay_weight += 0.1   # reinforce
        else:
            decay_weight -= 0.2   # decay
    """
    if new_score > old_score + IMPROVEMENT_GAP:
        return stored + REINFORCE_DELTA
    return stored - DECAY_DELTA


def should_evict(decay_weight: float) -> bool:
    """Return ``True`` if the pattern should be permanently evicted.

    Parameters
    ----------
    decay_weight:
        The *updated* decay weight (output of ``compute_new_decay_weight``).

    Returns
    -------
    bool
        ``True`` iff ``decay_weight < EVICTION_FLOOR``.
    """
    return decay_weight < EVICTION_FLOOR
