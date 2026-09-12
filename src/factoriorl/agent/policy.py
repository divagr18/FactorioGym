"""One evaluator-safe policy interface for hosted and local language models.

The agent loop owns an environment and can render this interface directly.  A
learner running on another host cannot, so the rollout bridge serialises the
same object.  Keeping this seam here prevents a second, weaker prompt renderer
from becoming the de-facto training contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from factoriorl.agent.summary import (
    SUMMARY_ENCODING_VERSION,
    LegalAction,
    ObservationSummary,
    TaskBrief,
    action_vocabulary,
    argument_domains,
    argument_requirements,
    legal_actions,
    objective_block,
    static_reference,
    summarise,
    survey_block,
    targetable_actions,
    visible_handles,
)


@dataclass(frozen=True)
class PolicyTurn:
    """Everything a policy may see when choosing one action.

    The raw domains are retained for local validation.  ``summary`` is the
    model-facing rendering and has no path to task truth or verifier state.
    """

    system_prompt: str
    static_prefix: str
    summary: ObservationSummary
    vocabulary: tuple[tuple[str, str], ...]
    legal: tuple[LegalAction, ...]
    raw_domains: dict[str, Any]
    requires: dict[str, tuple[str, ...]]
    targetable: frozenset[str]
    handles: frozenset[str]

    @property
    def prompt_digest(self) -> str:
        payload = f"{self.system_prompt}\n\n{self.static_prefix}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def messages(self) -> list[dict[str, str]]:
        """The first request body; callers append later turns immutably."""
        head = self.system_prompt
        if self.static_prefix:
            head = f"{head}\n\n{self.static_prefix}"
        return [
            {"role": "system", "content": head},
            {"role": "user", "content": self.summary.render()},
        ]

    def to_public_dict(self) -> dict[str, Any]:
        """A JSON-safe cross-host form, containing public policy inputs only."""
        return {
            "system_prompt": self.system_prompt,
            "static_prefix": self.static_prefix,
            "prompt_digest": self.prompt_digest,
            "summary": self.summary.render(),
            "summary_data": self.summary.to_dict(),
            "vocabulary": [list(item) for item in self.vocabulary],
            "legal_actions": [action.to_dict() for action in self.legal],
            "raw_domains": self.raw_domains,
            "requires": {key: list(value) for key, value in self.requires.items()},
            "targetable_actions": sorted(self.targetable),
            "visible_handles": sorted(self.handles),
            "summary_encoding_version": SUMMARY_ENCODING_VERSION,
        }


def policy_static_prefix(env: Any, *, static_knowledge: str = "") -> str:
    """The exact invariant prefix shared by the loop and rollout bridge."""
    return "\n\n".join(
        block
        for block in (
            static_knowledge,
            static_reference(env),
            objective_block(env),
            survey_block(env),
        )
        if block
    )


def build_policy_turn(
    env: Any,
    *,
    brief: TaskBrief,
    observation: dict,
    step: int,
    clock: dict | None = None,
    receipt: dict | None = None,
    static_knowledge: str = "",
) -> PolicyTurn:
    """Build the canonical current decision view from public environment data."""
    # Kept local to avoid an import cycle: ``loop`` imports this module to use
    # the same static-prefix constructor, while its prompt text remains the
    # canonical user-facing contract.
    from factoriorl.agent.loop import SYSTEM_PROMPT

    vocabulary = action_vocabulary(env)
    legal = legal_actions(vocabulary, env.action_masks())
    raw_domains = env.argument_domains() if any(
        getattr(template, "parameterized", False) for template in env.catalog.templates
    ) else {}
    requires = argument_requirements(env)
    return PolicyTurn(
        system_prompt=SYSTEM_PROMPT,
        static_prefix=policy_static_prefix(env, static_knowledge=static_knowledge),
        summary=summarise(
            observation,
            brief=brief,
            actions=legal,
            step=step,
            arguments=argument_domains(env),
            requires=requires,
            clock=clock,
            receipt=receipt,
        ),
        vocabulary=vocabulary,
        legal=legal,
        raw_domains=raw_domains,
        requires=requires,
        targetable=targetable_actions(env),
        handles=visible_handles(observation),
    )
