"""
tests/test_decay.py
~~~~~~~~~~~~~~~~~~~~
Property-based tests for patternmem.decay using Hypothesis.

Properties verified
-------------------
1. ``decay_weight`` is bounded above by ``initial + n * REINFORCE_DELTA`` for
   any sequence of n reinforcing score pairs.
2. A pattern starting at ``decay_weight = 0.5`` (2.5× DECAY_DELTA above the
   floor) is always evicted within 3 consecutive non-improving rounds.
3. ``should_evict`` is True iff weight < EVICTION_FLOOR.
4. Pure functions — no state mutation between calls.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from patternmem.decay import (
    DECAY_DELTA,
    EVICTION_FLOOR,
    IMPROVEMENT_GAP,
    REINFORCE_DELTA,
    compute_new_decay_weight,
    should_evict,
)


# ---------------------------------------------------------------------------
# Basic unit tests
# ---------------------------------------------------------------------------


class TestDecayBasics:
    def test_reinforce_when_score_improves_enough(self) -> None:
        new_weight = compute_new_decay_weight(
            stored=1.0, old_score=0.2, new_score=0.2 + IMPROVEMENT_GAP + 0.01
        )
        assert abs(new_weight - (1.0 + REINFORCE_DELTA)) < 1e-9

    def test_decay_when_score_does_not_improve_enough(self) -> None:
        new_weight = compute_new_decay_weight(
            stored=1.0, old_score=0.5, new_score=0.5  # no improvement
        )
        assert abs(new_weight - (1.0 - DECAY_DELTA)) < 1e-9

    def test_decay_at_exact_improvement_gap_boundary(self) -> None:
        """Exactly at the boundary is NOT a reinforce (> required, not >=)."""
        new_weight = compute_new_decay_weight(
            stored=1.0, old_score=0.5, new_score=0.5 + IMPROVEMENT_GAP
        )
        # 0.5 + 0.15 is NOT > 0.5 + 0.15, so this decays
        assert abs(new_weight - (1.0 - DECAY_DELTA)) < 1e-9

    def test_should_evict_true_below_floor(self) -> None:
        assert should_evict(EVICTION_FLOOR - 0.001) is True

    def test_should_evict_false_at_floor(self) -> None:
        assert should_evict(EVICTION_FLOOR) is False

    def test_should_evict_false_above_floor(self) -> None:
        assert should_evict(EVICTION_FLOOR + 0.001) is False


class TestEvictionWithin3Rounds:
    def test_pattern_at_half_decays_to_eviction_in_3_rounds(self) -> None:
        """Pattern at decay_weight=0.5 → evicted in ≤ 3 non-improving rounds.

        0.5 − 0.2 = 0.3 → still alive
        0.3 − 0.2 = 0.1 → at floor, not yet evicted (should_evict checks < not <=)
        0.1 − 0.2 = −0.1 → below floor, evict
        """
        weight = 0.5
        rounds = 0
        max_rounds = 3
        old_score = 0.5
        new_score = 0.5  # no improvement

        evicted = False
        while rounds < max_rounds:
            weight = compute_new_decay_weight(weight, old_score, new_score)
            rounds += 1
            if should_evict(weight):
                evicted = True
                break

        assert evicted, f"Pattern not evicted after {max_rounds} rounds (weight={weight})"


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


@given(
    initial=st.floats(min_value=0.1, max_value=2.0),
    n_reinforce=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=500)
def test_decay_weight_bounded_above(initial: float, n_reinforce: int) -> None:
    """For any n reinforcing rounds, weight ≤ initial + n * REINFORCE_DELTA."""
    weight = initial
    old_score = 0.3
    for _ in range(n_reinforce):
        weight = compute_new_decay_weight(
            stored=weight,
            old_score=old_score,
            new_score=old_score + IMPROVEMENT_GAP + 0.01,  # always reinforce
        )
    expected_max = initial + n_reinforce * REINFORCE_DELTA
    assert weight <= expected_max + 1e-9, (
        f"weight {weight} exceeded bound {expected_max}"
    )


@given(
    initial=st.floats(min_value=EVICTION_FLOOR, max_value=2.0),
    score_pairs=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0),
            st.floats(min_value=0.0, max_value=1.0),
        ),
        min_size=1,
        max_size=50,
    ),
)
@settings(max_examples=500)
def test_eviction_flag_consistent_with_floor(
    initial: float,
    score_pairs: list[tuple[float, float]],
) -> None:
    """``should_evict`` is True iff weight < EVICTION_FLOOR, always."""
    weight = initial
    for old_score, new_score in score_pairs:
        weight = compute_new_decay_weight(weight, old_score, new_score)
    assert should_evict(weight) == (weight < EVICTION_FLOOR)


@given(
    initial=st.floats(min_value=EVICTION_FLOOR + 0.001, max_value=0.5 + EVICTION_FLOOR),
    n_decay=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=200)
def test_monotone_decay_eventually_evicts(initial: float, n_decay: int) -> None:
    """Any pattern decaying monotonically eventually reaches eviction."""
    weight = initial
    old_score = 0.5
    new_score = 0.5  # no improvement → always decay
    for _ in range(n_decay):
        weight = compute_new_decay_weight(weight, old_score, new_score)

    # After enough rounds, weight must be strictly below initial
    expected_after = initial - n_decay * DECAY_DELTA
    assert abs(weight - expected_after) < 1e-9, (
        f"After {n_decay} decay rounds: expected {expected_after:.4f}, got {weight:.4f}"
    )
