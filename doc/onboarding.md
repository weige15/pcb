# Onboarding

## What This Project Does

研究 Qwen3-4B 的 WQ/WK/WV 權重各自量化為4/8-bit後，對BF16 baseline的NLL/KL/PPL影響。`projection_quantization_sensitivity` 表示「attention projection 權重量化敏感度」，不再以階段編號作主要程式名稱。

2026-09-08已完成14條件/896筆的舊實驗；此次只改命名、路徑及引用，不改量化/抽樣/forward/統計邏輯，也未重跑GPU實驗。

## Quickstart

```bash
source ~/.venv/bin/activate
nvidia-smi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
# 確認GPU2可用後，才啟動新的實驗：
CUDA_VISIBLE_DEVICES=2 bash scripts/run_projection_quantization_sensitivity.sh
```

需要既有venv與固定revision的模型/資料cache；完整命令和失敗處理見 [runbook](runbook.md)。新output預設 `results/projection_quantization_sensitivity/`，已存在manifest/CSV則拒絕覆寫。

## Important Files

| 舊命名 | 目前命名／位置 |
|---|---|
| `phase1a.md` | `projection_quantization_sensitivity.md` |
| `scripts/phase1a.py` | `scripts/projection_quantization_sensitivity.py` |
| `scripts/phase1a_config.json` | `scripts/projection_quantization_sensitivity_config.json` |
| `scripts/run_phase1a.sh` | `scripts/run_projection_quantization_sensitivity.sh` |
| `scripts/verify_phase1a.py` | `scripts/verify_projection_quantization_sensitivity.py` |
| `scripts/phase1a_preflight.sh` | `scripts/check_projection_quantization_resources.sh` |
| `tests/test_phase1a.py` | `tests/test_projection_quantization_sensitivity.py` |
| 舊 `results/phase1a*` 與smoke/log/docs/source | `archive/qwen3_projection_quantization_sensitivity_2026-09-08/`，保留當時檔名/內容 |

新run內的artifact名稱直接表明內容：`block_metrics.csv`、`run_manifest.json`、`experiment_config.json`、`sampled_token_blocks.npz`、`paired_bootstrap_indices.npy`、`sensitivity_analysis.json`、`sensitivity_report.md`、`artifact_verification.json`、`blocked.md`。

## Architecture Map

固定revision官方test parquet → 原row順序雙換行join → tokenizer不加special tokens → 非重疊2049-token blocks → seed42抽64並保存token IDs。

原始BF16 model → eval/no-cache/固定SDPA → BF16/WQ4 smoke與hash/還原控制 → BF16及6種weight RTN → 相同blocks前513/2049 tokens → chunked lm_head → FP32 NLL/KL → CSV → paired bootstrap CI。

只有最終baseline hidden存CPU；無KV cache、packed low-bit GEMM或activation量化。每條件finally還原全state；這些數學和控制邏輯沒有在命名重構中更動。

## Development Workflow

1. 新工作用描述性檔名與新output；不要修改封存裡的原始source、commands、hash或數據。
2. 需要查舊結果時讀 [封存入口](../archive/README.md)。原始manifest不可改成新source hash，否則會錯稱舊結果由新版程式產生。
3. 改程式後先測試、CLI/help、shell語法；改量化或forward才需要相應真模型驗證。只改命名不重新跑完整實驗。
4. 修改檔名/命令同步更新README/runbook；涉及紀錄格式不能混用舊parser與新artifact。
5. 不以CI跨零證明等價，不為取得排序而擴樣本或開始下一階段。

## Testing

- 12 tests覆蓋原數學/控制/錯誤路徑與新命名的入口、防覆寫。
- 封存51個原始檔案的checksums和舊manifest的source hashes逐一核對。
- 以真實896筆的暫存副本跑新分析入口，產生的analysis JSON和2000×64 draws與舊版逐byte相同；原始CSV不變。
- 測試不把暫存重算當新model measurement，不改archive。此次未重跑GPU實驗。
- 無build/lint/type-check framework；shell使用`bash -n`，Python用實際tests/CLI。

## Troubleshooting

[Runbook故障表](runbook.md#common-failures)區分舊路徑、缺資產、CUDA/OOM與report gate。舊basic-2 error999及CPU/CUDA rounding診斷仍完整保存在封存，不是目前待辦。

若新run缺GPU/checkpoint/data或OOM，保存output內`blocked.md`並回報未完成，請使用者`/goal pause`。不reset共享GPU、不改venv、不換小模型、不縮樣本、不無限重試。

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
