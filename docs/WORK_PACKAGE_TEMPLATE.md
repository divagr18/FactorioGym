# Work package template (PLAN.md §4)

```text
Task ID:
Objective:
Dependencies:
Owned components:
Inputs and contracts:
Required behavior:
Explicit exclusions:
Deliverables:
Verification commands:
Acceptance evidence:
Handoff notes:
```

Rules:

- Identify dependencies before dispatch; never redefine an upstream contract
  to make implementation easier.
- Ownership lanes: integration / runtime / Python systems / learning / tasks /
  experience / verification. One agent may hold several sequentially.
- No overlapping ownership of core runtime files or protocol definitions.
- A task is complete only when required behavior is implemented, verification
  runs against the intended environment, outputs and commands are recorded,
  public behavior is documented, known limitations are stated, no placeholder
  remains on the required path, and an integration check confirms
  compatibility with dependencies.
- "Implemented but not run" and "tests skipped because Factorio was
  unavailable" are incomplete states for engine-dependent work.
