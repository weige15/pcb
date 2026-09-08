#!/usr/bin/env bash
# Exactly one selected idle GPU; no scheduler, automatic retry, or completed-condition overwrite.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:-results/layerwise_projection_sensitivity}"
invocation="$(printf '%q ' bash scripts/run_layerwise_projection_sensitivity.sh "$@")"
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p "$output"
log="$output/execution_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "$log") 2>&1
blocked() {
  status=$?
  printf '\nLauncher failed: exit=%s; command=%s; log=%s\n' "$status" "$invocation" "$log"
  printf '\n## Launcher failure — 未完成\nCommand: `%s`\nExit: %s; error log: `%s`.\nInspect run_manifest.json and conditions/ for completed work (if absent 0/109). Missing: remaining conditions, controls, analysis and audit. Preserve prior evidence; request usable original assets or >=19GiB idle BF16 GPU and user `/goal pause` if resources are unavailable.\n' "$invocation" "$status" "$log" >> "$output/blocked.md"
  exit "$status"
}
trap blocked ERR
source "$HOME/.venv/bin/activate"
nvidia-smi
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU index or UUID explicitly}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf 'Launch UTC: '; date -u -Is
printf 'Command: %s\nCUDA_VISIBLE_DEVICES=%s\n' "$invocation" "$CUDA_VISIBLE_DEVICES"
python -u scripts/layerwise_projection_sensitivity.py --output "$output" "$@"
