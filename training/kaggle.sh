#!/usr/bin/env bash
# Run the Kaggle CLI with a freshly issued access token.
# Workaround: with Kaggle CLI 2.2.5 the OAuth login (`kaggle auth login`) sometimes stops being used for API calls
# ("Authentication required" / "Permission denied") although `kaggle auth print-access-token` still works.
# Passing that token explicitly fixes it. Nothing is stored: a new token is requested on every call.
# Usage: training/kaggle.sh kernels status sakshaymahna/teaching-a-unitree-g1-to-walk-reinforcement-learn
set -euo pipefail
KAGGLE="$(dirname "$0")/../.venv/bin/kaggle"
KAGGLE_API_TOKEN="$("$KAGGLE" auth print-access-token 2>/dev/null)" exec "$KAGGLE" "$@"
