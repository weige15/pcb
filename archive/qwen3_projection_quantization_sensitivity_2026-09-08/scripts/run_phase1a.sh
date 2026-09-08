#!/usr/bin/env bash
# One GPU, local pinned assets, no automatic retry and no overwrite of prior runs.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-results}"
invocation="$(printf '%q ' bash scripts/run_phase1a.sh "$@")"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$output"
log="$output/phase1a_execution_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "$log") 2>&1
blocked() {
  status=$?
  printf '\nLauncher failed: exit=%s; command=%s; log=%s\n' "$status" "$invocation" "$log"
  printf '\n\n## Launcher failure — 未完成\nCommand: `%s`\nExit: %s; full error: `%s`.\nSee existing phase1a_run.json/CSV for completed work (if absent: 0/14). Missing: successful execution, remaining conditions/controls and audited NLL/KL/CI. Do not retry without resolving the recorded error. For missing resources use `/goal pause` and provide a usable GPU/checkpoint/data.\n' "$invocation" "$status" "$log" >> "$output/phase1a_blocked.md"
  exit "$status"
}
trap blocked ERR
source "$HOME/.venv/bin/activate"
nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf 'Launch UTC: '; date -u -Is
printf 'Command: %s\nCUDA_VISIBLE_DEVICES=%s\n' "$invocation" "$CUDA_VISIBLE_DEVICES"
python -u scripts/phase1a.py --config scripts/phase1a_config.json --output "$output" "$@"
