#!/usr/bin/env bash
set -euo pipefail

TARGETS="${@:-.}"

uv run ruff format "${TARGETS}"
uv run ruff check --fix "${TARGETS}"
