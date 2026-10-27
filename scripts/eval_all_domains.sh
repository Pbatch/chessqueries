#!/usr/bin/env bash
# Evaluate one ChessQueries checkpoint across every test domain and write the
# per-domain metrics JSONs (outputs/chessqueries_<name>/metrics_<domain>.json).
#
# Usage: eval_all_domains.sh <checkpoint> <name> [resolution]
# The resolution defaults to the one a .safetensors checkpoint records; a Lightning
# .ckpt needs it passed.
set -euo pipefail

checkpoint="$1"; name="$2"
res_args=()
[[ $# -ge 3 ]] && res_args=(--resolution "$3")
if [[ ! -f "$checkpoint" ]]; then
  echo "ERROR: checkpoint not found: $checkpoint" >&2
  exit 2
fi
echo "eval_all_domains: name=$name checkpoint=$checkpoint ${res_args[*]}"

for pair in "chesscam test" "chessred test" "chesscog test" "slcc test"; do
  set -- $pair
  uv run python scripts/eval_cross_dataset.py \
    --checkpoint "$checkpoint" --dataset "$1" --split "$2" "${res_args[@]}" --name "$name"
done
# CVChess has no split (zero-shot, one game).
uv run python scripts/eval_cross_dataset.py \
  --checkpoint "$checkpoint" --dataset cvchess "${res_args[@]}" --name "$name"
echo "eval_all_domains DONE: $name"
