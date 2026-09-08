#!/usr/bin/env bash
# Resource check only; does not run the sensitivity experiment or change the host.
set -uo pipefail
source "$HOME/.venv/bin/activate" || exit 1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

printf 'UTC: '; date -u -Is
printf 'Host: '; hostname
printf 'CWD: '; pwd
printf '\nGPU inventory (a failed device may make nvidia-smi return nonzero):\n'
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,driver_version --format=csv
printf 'nvidia-smi exit: %s\n' "$?"
printf '\nSelected physical GPU:\n'
nvidia-smi -i 0 --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,utilization.gpu --format=csv
printf 'nvidia-smi -i 0 exit: %s\n' "$?"
printf '\nDriver and device nodes:\n'
ls -l /dev/nvidia* 2>&1
printf '\nPython / CUDA / cached assets:\n'
python -u - <<'PY'
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

print('python:', sys.executable, sys.version)
for name in ('torch', 'transformers', 'datasets', 'huggingface-hub',
             'safetensors', 'numpy', 'accelerate', 'pyarrow', 'pytest'):
    try:
        print(f'{name}: {importlib.metadata.version(name)}')
    except importlib.metadata.PackageNotFoundError:
        print(f'{name}: NOT INSTALLED')
for name in ('VIRTUAL_ENV', 'CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES',
             'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE',
             'LD_LIBRARY_PATH'):
    print(f'{name}: {os.environ.get(name)}')

# cuInit tests the driver API independently of PyTorch/model loading.
try:
    cuda = ctypes.CDLL('libcuda.so.1')
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    status = cuda.cuInit(0)
    error_name = ctypes.c_char_p()
    cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cuda.cuGetErrorName(status, ctypes.byref(error_name))
    print('CUDA driver cuInit:', status, error_name.value)
except Exception:
    traceback.print_exc()

cuda_ok = False
try:
    import torch
    print('torch CUDA build:', torch.version.cuda)
    print('torch CUDA available:', torch.cuda.is_available())
    print('torch device count:', torch.cuda.device_count())
    print('torch device 0:', torch.cuda.get_device_name(0))
    print('BF16 supported:', torch.cuda.is_bf16_supported())
    assert torch.cuda.is_bf16_supported(), 'BF16 support required'
    x = torch.ones((2, 2), device='cuda:0', dtype=torch.bfloat16)
    y = x @ x
    torch.cuda.synchronize()
    assert y.dtype == torch.bfloat16 and torch.equal(y, torch.full_like(y, 2))
    print('BF16 CUDA matmul: PASS')
    cuda_ok = True
except Exception:
    traceback.print_exc()

# Inspect only existing local snapshots; no downloads and no substitute assets.
hf_home = Path(os.environ.get('HF_HOME', Path.home() / '.cache/huggingface'))
hub = Path(os.environ.get('HF_HUB_CACHE', hf_home / 'hub'))
assets_ok = True
for repo in ('models--Qwen--Qwen3-4B', 'datasets--Salesforce--wikitext'):
    try:
        root = hub / repo
        revision = (root / 'refs/main').read_text().strip()
        snapshot = root / 'snapshots' / revision
        print('cached asset:', repo, 'revision:', revision, 'snapshot:', snapshot)
        if repo.startswith('models--'):
            config = json.loads((snapshot / 'config.json').read_text())
            print('model config:', json.dumps(config, sort_keys=True))
            index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
            from safetensors import safe_open
            header_keys = set()
            for shard in sorted(set(index['weight_map'].values())):
                path = snapshot / shard
                with safe_open(str(path), framework='pt', device='cpu') as f:
                    header_keys.update(f.keys())
                print('readable checkpoint header:', shard, 'bytes:', path.stat().st_size)
            assert header_keys == set(index['weight_map']), 'checkpoint header/index mismatch'
            for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
                path = snapshot / name
                assert path.is_file() and path.stat().st_size > 0, name
                print('tokenizer asset:', name, 'bytes:', path.stat().st_size)
            print('Checkpoint headers/files present; full model load NOT tested.')
        else:
            import pyarrow.parquet as pq
            path = snapshot / 'wikitext-2-raw-v1/test-00000-of-00001.parquet'
            table = pq.read_table(path, columns=['text'])
            print('readable test parquet:', path, 'rows:', table.num_rows,
                  'nulls:', table.column('text').null_count)
            assert table.num_rows > 0 and table.column('text').null_count == 0
            print('Dataset source readable; tokenization and 64-block selection NOT tested.')
    except Exception:
        assets_ok = False
        traceback.print_exc()

imports_ok = True
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    print('Transformers / datasets entry-point imports: PASS (not a model load)')
except Exception:
    imports_ok = False
    traceback.print_exc()

ok = cuda_ok and assets_ok and imports_ok
print('PREFLIGHT:', 'PASS (not experiment completion)' if ok else 'BLOCKED',
      json.dumps({'cuda_bf16': cuda_ok, 'cached_asset_checks': assets_ok,
                  'entry_point_imports': imports_ok}))
sys.exit(0 if ok else 1)
PY
status=$?
printf '\nPreflight exit: %s\n' "$status"
exit "$status"
