#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/catlab/miniconda3/bin/conda run --no-capture-output -n robosuite \
    python -u "$project_dir/smoke_test.py" "$@"
