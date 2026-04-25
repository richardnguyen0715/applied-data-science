#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [CONFIG_PATH]"
  echo "Example: $0 src/configs/config.yaml"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONFIG_PATH="${1:-src/configs/config.yaml}"

if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
elif [[ -f "venv/bin/activate" ]]; then
  source "venv/bin/activate"
fi

python -m src.main --config "$CONFIG_PATH"
