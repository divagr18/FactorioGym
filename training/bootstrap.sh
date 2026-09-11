#!/usr/bin/env bash
# Create the deliberately isolated, experimental Gemma 4 + vLLM runtime.
#
# Unsloth's published metadata has not caught up with the Transformers 5.5+
# and TRL 0.28 combination used by its Gemma 4 RL notebook.  The notebook
# itself installs these packages with --no-deps.  Keep that override isolated
# to this directory and visible in version-control rather than contaminating
# the repository trainer environment.
set -euo pipefail

cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif python3 -m pip --version >/dev/null 2>&1; then
  python3 -m pip install --user --upgrade uv
  uv_bin="$(python3 -m site --user-base)/bin/uv"
else
  # Minimal Ubuntu WSL images do not ship pip. uv's standalone installer keeps
  # the runtime self-contained and avoids apt-level changes to the host.
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv_bin="$HOME/.local/bin/uv"
fi
"$uv_bin" lock
"$uv_bin" sync --locked
"$uv_bin" pip install --python .venv/bin/python --no-deps \
  "unsloth>=2026.4.4" \
  "unsloth-zoo>=2026.4.4"
"$uv_bin" run python - <<'PY'
import importlib.metadata as metadata

for package in ("torch", "transformers", "trl", "vllm", "unsloth", "unsloth-zoo"):
    print(f"{package}=={metadata.version(package)}")
PY
