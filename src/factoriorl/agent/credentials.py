"""Credential handling for model adapters (PLAN.md 5.3).

PLAN 5.3 requires that "credentials remain outside run artifacts". Two rules
make that structural rather than a habit:

* **A key is only ever read from the process environment.** There is no
  config-file path here on purpose. The repository's ``.gitignore`` already
  excludes credential-shaped files (``*.credentials*``, ``*.token*``,
  ``.factoriorl.user.json``), which tells you the convention is "a secret lives
  outside version control"; the way to honour that in a run harness is not to
  invent a fourth file shape but to keep the secret in the environment where
  neither the repository nor a run directory can capture it.
* **Redaction lives next to the secret.** An adapter holds its key privately and
  exposes :func:`redactor` over it, so everything the loop records -- the raw
  model text and any provider error string -- passes through a function that
  knows the value to remove. The loop itself never receives the key, so it
  cannot write one out even by accident.

The failure this prevents is concrete: an endpoint that echoes its
``Authorization`` header into an error body, or a proxy that includes the key in
a 401 message, would otherwise land verbatim in ``decisions.jsonl`` next to the
training runs, and run directories are shared as evidence.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

#: What a redacted secret is replaced with. Distinctive on purpose: a reader of
#: a run artifact should be able to tell "a credential was removed here" from
#: "this field was empty".
REDACTED = "<redacted>"

#: The shortest value worth redacting. Redacting a one- or two-character
#: "secret" would rewrite unrelated text throughout an artifact and make the
#: record useless; a credential shorter than this is a configuration mistake,
#: not something to paper over.
MIN_SECRET_LENGTH = 8


def read_credential(env_var: str | None) -> str | None:
    """Read one credential from the environment, or ``None`` if it is unset.

    An empty string counts as unset. A local inference endpoint usually needs no
    key at all, and treating ``""`` as a credential would make the adapter send
    an empty ``Authorization`` header, which some servers reject with a 401 that
    reads like a wrong key rather than a missing one.
    """
    if not env_var:
        return None
    return os.environ.get(env_var) or None


def redactor(secrets: Iterable[str | None]) -> Callable[[str], str]:
    """Build a function that removes the given secrets from any recorded text."""
    values = sorted(
        {s for s in secrets if s and len(s) >= MIN_SECRET_LENGTH},
        key=len,
        reverse=True,
    )

    def redact(text: str) -> str:
        if not text:
            return text
        for value in values:
            text = text.replace(value, REDACTED)
        return text

    return redact
