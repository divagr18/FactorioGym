"""What a client was given beyond the bare observation.

`profiles.assistance` was the literal string "none" in both manifest writers,
so a run whose goal geometry the environment chose for it read identically to
one that chose for itself. R2.3 of the development redirection names that
directly: do not replace an unresolved multi-fault task with an undocumented
automatic subgoal selector.
"""

from __future__ import annotations

from typing import Any

#: The goal vector points at whatever the task declared, unchanged.
STATIC = "none"


def describe_assistance(spec: Any, *, extra: tuple[str, ...] = ()) -> str:
    """A declared name for every assistance a run received.

    Composed rather than fixed so a second assistance cannot silently replace
    the first in the record.
    """
    parts: list[str] = []
    policy = getattr(spec, "focus_policy", "static")
    if policy != "static":
        parts.append(f"goal-focus:{policy}")
    parts.extend(extra)
    return "+".join(parts) if parts else STATIC
