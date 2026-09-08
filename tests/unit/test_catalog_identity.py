"""The action space a checkpoint was trained against must be identifiable.

`ResolvedCatalog.digest()` covered `[key, action, payload]` but not `requires`,
so editing a template's availability rule -- which is what the mask builder
consults -- left the digest, and every manifest citing it, unchanged. And
nothing re-derived the digest at load, so a catalog edit that kept the template
count produced a checkpoint that loaded clean, ran, and meant something else.
"""

from __future__ import annotations

import json
from dataclasses import replace

from factoriorl import catalog as catalog_module
from factoriorl.manifest import _catalog_problems
from factoriorl.tasks import get


def _resolved(task_id="repair_belt"):
    spec = get(task_id).spec
    return catalog_module.resolve(spec.catalog, spec.catalog_subset)


class TestDigestCoversAvailability:
    def test_changing_requires_moves_the_digest(self):
        original = _resolved()
        edited = replace(
            original,
            templates=tuple(
                replace(t, requires="never_present") if index == 0 else t
                for index, t in enumerate(original.templates)
            ),
        )
        assert edited.digest() != original.digest()

    def test_the_payload_still_moves_the_digest(self):
        original = _resolved()
        first = original.templates[0]
        edited = replace(
            original,
            templates=(replace(first, payload={**first.payload, "ticks": 999}),)
            + original.templates[1:],
        )
        assert edited.digest() != original.digest()

    def test_an_unchanged_catalog_is_stable(self):
        assert _resolved().digest() == _resolved().digest()

    def test_two_tasks_with_different_subsets_differ(self):
        assert _resolved("repair_belt").digest() != _resolved("deliver").digest()


class TestVerifyRederivesTheCatalog:
    def _manifest(self, task_id, digest):
        return {
            "task": {"id": task_id},
            "profiles": {"catalog_digest": digest},
        }

    def test_a_matching_digest_reports_no_problem(self):
        digest = _resolved("deliver").digest()
        assert _catalog_problems(self._manifest("deliver", digest)) == []

    def test_a_stale_digest_is_reported(self):
        problems = _catalog_problems(self._manifest("deliver", "0000000000000000"))
        assert len(problems) == 1
        assert "action catalog changed" in problems[0]
        assert "deliver" in problems[0]

    def test_a_manifest_without_a_catalog_is_not_an_error(self):
        """Most runs are not release runs and need not carry one."""
        assert _catalog_problems({"task": {"id": "deliver"}, "profiles": {}}) == []
        assert _catalog_problems({}) == []

    def test_an_unknown_task_does_not_crash_verify(self):
        """A renamed or removed task must degrade to a reported problem."""
        problems = _catalog_problems(self._manifest("no-such-task", "abcd"))
        assert len(problems) == 1
        assert "could not be re-derived" in problems[0]


def test_the_digest_payload_is_canonical_json():
    """Stable across processes, so it can be compared across machines."""
    resolved = _resolved()
    payload = json.dumps(
        [[t.key, t.action, t.payload, t.requires] for t in resolved.templates],
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    assert resolved.digest() == hashlib.sha256(payload.encode()).hexdigest()[:16]
