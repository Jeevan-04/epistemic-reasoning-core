#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT="$ROOT/reproducibility_bundle.zip"
echo "Creating reproducibility bundle at $OUT"
cd "$ROOT"
zip -r "$OUT" \
  tests/logs tests/benchmarks scripts paper/generated_eval_tables.tex paper/figures REPRODUCIBILITY.md || true
echo "Wrote $OUT"
