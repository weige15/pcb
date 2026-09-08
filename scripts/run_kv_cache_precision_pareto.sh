#!/usr/bin/env bash
# Fixed one-GPU launcher. No scheduler, retries, reduced matrix, or fallback cache path.
set -euo pipefail
cd "$(dirname "$0")/.."
out="results/kv_cache_precision_pareto"
phase="all"
if [[ "${1:-}" == "--phase" ]]; then phase="${2:?missing phase}"; shift 2; fi
if [[ $# -gt 0 ]]; then out="$1"; shift; fi
mkdir -p "$out"
log="$out/execution_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee "$log") 2>&1
source "$HOME/.venv/bin/activate"
nvidia-smi
: "${CUDA_VISIBLE_DEVICES:?Select one explicitly selected idle GPU index or UUID}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
printf 'Launch UTC: '; date -u -Is
printf 'Command: phase=%s output=%s CUDA_VISIBLE_DEVICES=%s\n' "$phase" "$out" "$CUDA_VISIBLE_DEVICES"
if [[ ! -f "$out/prompts.npz" || ! -f "$out/prompt_manifest.json" ]]; then
  python -u scripts/prepare_kv_cache_inputs.py --output "$out"
fi
python -u scripts/kv_cache_precision_pareto.py --phase "$phase" --output "$out" "$@"
