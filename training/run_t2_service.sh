#!/usr/bin/env bash
# Run one T2 Gemma collection from a persistent WSL systemd unit.
# The Windows bridge is deliberately owned by the caller; this process owns
# only the model and writes all diagnostic output through its service stdout.
set -euo pipefail

: "${FACTORIORL_BRIDGE_URL:?FACTORIORL_BRIDGE_URL is required}"
: "${FACTORIORL_BRIDGE_TOKEN:?FACTORIORL_BRIDGE_TOKEN is required}"
: "${FACTORIORL_T2_ROOT:?FACTORIORL_T2_ROOT is required}"
: "${FACTORIORL_T2_OUTPUT:?FACTORIORL_T2_OUTPUT is required}"

export PYTHONPATH="${FACTORIORL_T2_ROOT}/training${PYTHONPATH:+:${PYTHONPATH}}"

exec /root/factoriorl-t0/.venv/bin/python \
  "${FACTORIORL_T2_ROOT}/training/run_gemma_group.py" \
  --output "${FACTORIORL_T2_OUTPUT}" \
  --seed "${FACTORIORL_T2_SEED:-20260911}" \
  --max-turns "${FACTORIORL_T2_MAX_TURNS:-4}"
