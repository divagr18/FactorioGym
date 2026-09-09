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
from pathlib import Path

#: What a redacted secret is replaced with. Distinctive on purpose: a reader of
#: a run artifact should be able to tell "a credential was removed here" from
#: "this field was empty".
REDACTED = "<redacted>"

#: The shortest value worth redacting. Redacting a one- or two-character
#: "secret" would rewrite unrelated text throughout an artifact and make the
#: record useless; a credential shorter than this is a configuration mistake,
#: not something to paper over.
MIN_SECRET_LENGTH = 8


#: Default location of the optional environment file, relative to the workspace
#: root. Named `.env` because that is the convention every other tool in this
#: space uses; gitignored, with `.env.example` deliberately un-ignored so the
#: variable *names* can be published without the values.
ENV_FILE_NAME = ".env"


def load_env_file(path=None) -> list[str]:
    """Seed `os.environ` from a `.env` file, and report which names were set.

    This does not weaken the rule above, and the distinction is worth stating
    because it looks like an exception. A key is still only ever *read* from the
    process environment: `read_credential` is unchanged and no adapter gains a
    file path. All this does is populate the environment before anything reads
    it, which is what a shell `set -a; . ./.env` would do -- the same operation,
    performed where both the diagnostic and the run can perform it identically.

    That identity is the point. `factoriorl doctor-agent` read `os.environ`
    while `tools/demonstration.py` carried its own inline loader, so a key
    present only in `.env` made the diagnostic report "not set" about a run that
    then worked. A diagnostic that disagrees with the thing it diagnoses is
    worse than no diagnostic.

    `setdefault`, so a variable already exported wins over the file: an
    explicitly set environment is the more deliberate of the two.

    Returns the variable names it set, never their values, so a caller can
    report provenance without handling a secret.
    """
    from factoriorl.paths import workspace_root

    target = Path(path) if path is not None else workspace_root() / ENV_FILE_NAME
    if not target.is_file():
        return []
    names: list[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        # Quotes are stripped because a shell would strip them; anything else
        # about the value is left alone, including whitespace inside it.
        if key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")
            names.append(key)
    return names


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
