"""The six introductory task families (PLAN.md 3.2).

Each module registers one family on import. Adding a task means dropping a
module here -- nothing in worker management changes, and an import-direction
test asserts that.

Held-out generalisation is by **layout family**, not by seed: PLAN.md section 3
asks for "unfamiliar seeds and unfamiliar structures" to be reported
separately, so the split lives on the structure.
"""
