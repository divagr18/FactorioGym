"""The task families (DESIGN.md 3.2). Ten of them, not the six this once said.

Each module registers one family on import. Adding a task means dropping a
module here -- nothing in worker management changes, and an import-direction
test asserts that. A family may also declare its own reference solver on the
``RegisteredTask``, which is what keeps authoring inside one file; see
``docs/AUTHORING_TASKS.md`` for the parts that are not so contained.

Held-out generalisation is by **layout family**, not by seed: DESIGN.md section 3
asks for "unfamiliar seeds and unfamiliar structures" to be reported
separately, so the split lives on the structure.
"""
