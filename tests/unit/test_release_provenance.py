"""A published number must name a scene set someone can find (PLAN 4.5).

PLAN 4.5 requires the holdout's hash to "match every manifest citing it", and
`tools/freeze_holdout.manifests_citing` exists so that is checkable from the
outside. But nothing checked the other direction: whether a published
*release* file cites a holdout that still exists.

It did not. Seven of eleven `phase4-release-*.json` cite a `content_hash` no
committed holdout carries — `f77aa2c291f720f9` in four v1-era files,
`b41e4f0c576156bc` in three v2-era ones — and two disagree with their own
headers at cell level, so they report numbers measured against several
different scene sets under one heading.

Most of that is recoverable: both hashes were the live holdout at a nameable
commit, which is why P4-B added a `holdout_provenance` block rather than only a
`superseded` marker. One is not.
`phase4-release-v2-restore_power.json`'s first cell cites
`7edc5ee292b90972`, which was never any committed holdout's `content_hash` at
any commit — `git log --all -S` finds it only in the commit that added the
evidence file.

This test is engine-free and runs in CI because `gate_phase4` reads none of
these files.
"""

from __future__ import annotations

import json
import pathlib

import pytest

EVIDENCE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "evidence"
RELEASE_FILES = sorted(EVIDENCE.glob("phase4-release-*.json"))


def _committed_hashes() -> set[str]:
    """The `content_hash` of every holdout in the working tree."""
    found = set()
    for path in EVIDENCE.glob("holdout_v*.json"):
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("content_hash"):
            found.add(body["content_hash"])
    return found


def _citations(body: dict) -> list[tuple[str, str]]:
    """Every (where, content_hash) this document claims, header and cells."""
    out = []
    header = (body.get("holdout") or {}).get("content_hash")
    if header:
        out.append(("header", header))
    for index, run in enumerate(body.get("runs") or []):
        cell = ((run.get("held_out") or {}).get("holdout") or {}).get("content_hash")
        if cell:
            out.append((f"runs[{index}]", cell))
    for family, block in (body.get("families") or {}).items():
        for index, cell in enumerate(block.get("cells") or []):
            if cell.get("holdout_content_hash"):
                out.append((f"{family}.cells[{index}]", cell["holdout_content_hash"]))
    return out


def test_there_are_release_files_to_check():
    """Guard the guard: a glob that silently matched nothing would pass."""
    assert len(RELEASE_FILES) >= 11, [p.name for p in RELEASE_FILES]


@pytest.mark.parametrize("path", RELEASE_FILES, ids=lambda p: p.name)
def test_every_citation_resolves_or_is_declared_unresolvable(path):
    """Either the hash is a live holdout's, or the file says where it went."""
    body = json.loads(path.read_text(encoding="utf-8"))
    live = _committed_hashes()
    provenance = body.get("holdout_provenance") or {}
    unresolved = sorted({digest for _where, digest in _citations(body) if digest not in live})
    if not unresolved:
        return
    assert provenance, (
        f"{path.name} cites {len(unresolved)} hash(es) no committed holdout carries "
        f"({[d[:16] for d in unresolved]}) and carries no `holdout_provenance` block. "
        "A published number whose scene set cannot be found describes nothing."
    )
    assert provenance.get("resolves_to"), (
        f"{path.name}: `holdout_provenance` must name where the hash was the live holdout"
    )


@pytest.mark.parametrize("path", RELEASE_FILES, ids=lambda p: p.name)
def test_a_file_whose_cells_disagree_with_its_header_says_so(path):
    """Two files report several scene sets under one heading. That has to be
    stated in the file, not left for a reader to diff cell by cell."""
    body = json.loads(path.read_text(encoding="utf-8"))
    citations = _citations(body)
    header = dict(citations).get("header")
    if header is None:
        return
    disagreeing = {digest for where, digest in citations if where != "header" and digest != header}
    if not disagreeing:
        return
    provenance = body.get("holdout_provenance") or {}
    assert "disagree" in (provenance.get("cells") or ""), (
        f"{path.name}: cells cite {[d[:16] for d in sorted(disagreeing)]} against a header of "
        f"{header[:16]}, and `holdout_provenance.cells` does not say the two disagree"
    )


def test_the_one_unrecoverable_cell_is_named_as_such():
    """`7edc5ee2…` was never any committed holdout's hash at any commit, so
    that cell's scene set cannot be retrieved even from history. A file that
    merely called it 'historic' would overstate what is recoverable."""
    path = EVIDENCE / "phase4-release-v2-restore_power.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    provenance = body.get("holdout_provenance") or {}
    assert provenance.get("unrecoverable_cells"), (
        "the file with an unrecoverable cell must name it in "
        "`holdout_provenance.unrecoverable_cells`"
    )
    assert "7edc5ee2" in json.dumps(provenance)


def test_the_live_holdout_is_cited_by_something():
    """If no published file cites the current holdout, the live one has no
    results against it and saying otherwise would be wrong."""
    live = json.loads((EVIDENCE / "holdout_v3.json").read_text(encoding="utf-8"))
    citing = [
        path.name
        for path in RELEASE_FILES
        if live["content_hash"]
        in {digest for _w, digest in _citations(json.loads(path.read_text(encoding="utf-8")))}
    ]
    # Asserted empty, not merely observed. `isinstance(citing, list)` was the
    # first version of this line and could never fail -- the kind of check this
    # repo has twice recorded as worthless. Empty is the true current state:
    # every published release file predates the live freeze, so **no published
    # release number has ever been measured against `holdout_v3` as it stands
    # today**. When that changes this test fails, which is the point: it forces
    # docs/CURRENT_STATUS.md to be updated in the same change.
    assert citing == [], (
        f"{citing} now cite the live holdout. That is progress -- update this "
        "test and the results table in docs/CURRENT_STATUS.md together."
    )
