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
    # Fault *localisation*, which was undeclared while fault *selection* was
    # named. Publishing the fault tile is the larger hint of the two: with it,
    # "repair a broken line" is "walk to a published coordinate and place one
    # item", and R4.3's `diagnosis` track is exactly the same scene without it.
    #
    # The two declarations are independent and merely happen to coincide in
    # both repair families today -- `fault_markers` says what the evaluator
    # treats as the fault, `public_markers` says what the agent can see -- so
    # this is computed from their intersection rather than assumed.
    fault = tuple(getattr(spec, "fault_markers", ()) or ())
    if fault:
        published = set(getattr(spec, "public_markers", ()) or ())
        if published.intersection(fault):
            parts.append("fault-location:published")
    parts.extend(extra)
    return "+".join(parts) if parts else STATIC
