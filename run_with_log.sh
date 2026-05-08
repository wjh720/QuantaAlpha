#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 \"direction\" [library_suffix]"
    echo "Example: $0 \"Price-Volume Factor Mining\""
    echo "Example: $0 \"Microstructure Factors\" \"exp_micro\""
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DIRECTION_SLUG="$(printf '%s' "$1" | tr ' /' '__' | tr -cd '[:alnum:]_-')"
LOG_FILE="logs/run_${TIMESTAMP}_${DIRECTION_SLUG}.log"

echo "Log file: ${LOG_FILE}"

if [ $# -ge 2 ]; then
    ./run.sh "$1" "$2" > "${LOG_FILE}" 2>&1
else
    ./run.sh "$1" > "${LOG_FILE}" 2>&1
fi
