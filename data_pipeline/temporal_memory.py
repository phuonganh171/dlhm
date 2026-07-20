"""
Temporal memory builder for hierarchy prediction.

Adapted from ORacle's Memory Scene Graph mechanism. Instead of scene-graph
triplets, the memory encodes L0/L1 hierarchy state changes within a single
L2 segment for one role.

Memory is scoped to the current L2 segment — it captures what happened
*earlier in this phase* for this role, not the full surgery history.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config import SHORT_TERM_WINDOW


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChangeEntry:
    """One state-change event in the temporal changelog."""
    tp_id: str
    level: str          # "L0" or "L1"
    action: str         # "start" or "remove"
    description: str    # the L0 description or L1 summary text

    def format(self) -> str:
        prefix = "+" if self.action == "start" else "-"
        return f"{prefix}{self.level}:{self.description}"


@dataclass
class MemoryState:
    """Accumulated memory at a specific frame within an L2 segment."""
    long_term: List[str] = field(default_factory=list)
    short_term: List[str] = field(default_factory=list)
    changelog: List[ChangeEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory builder
# ---------------------------------------------------------------------------

class TemporalMemoryBuilder:
    """
    Walk through frames in an L2 segment and build temporal memory.

    At each frame, compare the current (L0, L1) state against the
    previous frame's state. Record transitions into a changelog, then
    format long-term and short-term memory strings.

    During training: feed ground-truth (L0, L1) states.
    During inference: feed the model's own predicted states.
    """

    def __init__(self, short_term_window: int = SHORT_TERM_WINDOW):
        self.short_term_window = short_term_window
        self._changelog: List[ChangeEntry] = []
        self._long_term_set: set[str] = set()
        self._long_term_list: List[str] = []
        self._prev_l0: Optional[str] = None
        self._prev_l1: Optional[str] = None

    def reset(self) -> None:
        """Clear state for a new L2 segment."""
        self._changelog.clear()
        self._long_term_set.clear()
        self._long_term_list.clear()
        self._prev_l0 = None
        self._prev_l1 = None

    def step(
        self,
        tp_id: str,
        l0_description: str,
        l1_summary: str,
    ) -> MemoryState:
        """
        Process one frame and return the memory state *before* this frame.

        The returned memory represents what the model sees as context when
        predicting for tp_id. It does NOT include tp_id's own state.

        Returns a MemoryState with the long-term and short-term strings
        accumulated from all previous frames in this L2 segment.
        """
        # Snapshot memory BEFORE updating with current frame
        state = MemoryState(
            long_term=list(self._long_term_list),
            short_term=[e.format() for e in self._changelog[-self.short_term_window:]],
            changelog=list(self._changelog),
        )

        # Detect state changes
        if self._prev_l1 is not None and l1_summary != self._prev_l1:
            removal = ChangeEntry(tp_id, "L1", "remove", self._prev_l1)
            self._changelog.append(removal)
            addition = ChangeEntry(tp_id, "L1", "start", l1_summary)
            self._changelog.append(addition)
            key = f"L1:{l1_summary}"
            if key not in self._long_term_set:
                self._long_term_set.add(key)
                self._long_term_list.append(key)
        elif self._prev_l1 is None and l1_summary:
            addition = ChangeEntry(tp_id, "L1", "start", l1_summary)
            self._changelog.append(addition)
            key = f"L1:{l1_summary}"
            if key not in self._long_term_set:
                self._long_term_set.add(key)
                self._long_term_list.append(key)

        if self._prev_l0 is not None and l0_description != self._prev_l0:
            removal = ChangeEntry(tp_id, "L0", "remove", self._prev_l0)
            self._changelog.append(removal)
            addition = ChangeEntry(tp_id, "L0", "start", l0_description)
            self._changelog.append(addition)
            key = f"L0:{l0_description}"
            if key not in self._long_term_set:
                self._long_term_set.add(key)
                self._long_term_list.append(key)
        elif self._prev_l0 is None and l0_description:
            addition = ChangeEntry(tp_id, "L0", "start", l0_description)
            self._changelog.append(addition)
            key = f"L0:{l0_description}"
            if key not in self._long_term_set:
                self._long_term_set.add(key)
                self._long_term_list.append(key)

        self._prev_l0 = l0_description
        self._prev_l1 = l1_summary

        return state


# ---------------------------------------------------------------------------
# Memory formatting
# ---------------------------------------------------------------------------

def format_memory_string(
    state: MemoryState,
    include_long: bool = True,
    include_short: bool = True,
) -> str:
    """
    Format a MemoryState into the text string injected into the prompt.

    Example output::

        <memory_start>
        Long: L1:Patient positioning; L0:preparing patient
        Short: -L0:preparing patient; +L0:drilling patient
        <memory_end>
    """
    if not include_long and not include_short:
        return ""

    parts: List[str] = []

    if include_long and state.long_term:
        parts.append("Long: " + "; ".join(state.long_term))

    if include_short and state.short_term:
        parts.append("Short: " + "; ".join(state.short_term))

    if not parts:
        return ""

    return "<memory_start>\n" + "\n".join(parts) + "\n<memory_end>"


# ---------------------------------------------------------------------------
# Temporal augmentation
# ---------------------------------------------------------------------------

def augment_memory(
    state: MemoryState,
    rng: random.Random | None = None,
) -> str:
    """
    Apply temporal augmentation to a memory state and return formatted string.

    Augmentation strategy (from PLAN.md):
    - 50% chance: drop memory entirely (return "")
    - ~16.7%: short-term only
    - ~16.7%: long-term only
    - ~16.7%: both long + short

    Within any selected tier, each entry has 50% chance of being dropped
    (history dropout).
    """
    rng = rng or random.Random()

    roll = rng.random()
    if roll < 0.5:
        return ""
    elif roll < 0.667:
        # short-term only
        include_long, include_short = False, True
    elif roll < 0.833:
        # long-term only
        include_long, include_short = True, False
    else:
        # both
        include_long, include_short = True, True

    # History dropout: 50% chance to drop each entry
    aug_state = MemoryState(
        long_term=[e for e in state.long_term if rng.random() > 0.5],
        short_term=[e for e in state.short_term if rng.random() > 0.5],
    )

    return format_memory_string(aug_state, include_long, include_short)
