# Onboarding

## What This Project Does

Qwen3-4B attention projection **weight** RTN表示誤差實驗。兩種獨立範圍：

- 歷史全層：一次量化所有層同一projection、4/8-bit、512/2048位置，14條件/896筆，保留在 `results/phase1a*`。
- 本輪逐層：每次只量化單一層的WQ/WK/WV為4-bit、其餘原始BF16；固定歷史64 blocks與2048預測位置，108介入+BF16。交付 `results/layerwise_projection_sensitivity/`。

不做scheduler、activation/KV量化、packed GEMM或serving效益推論；接受不同層/metric没有一致排序。

## Quickstart

從repo root執行。使用現有 `~/.venv` 和已釘選的本地模型/資料，先檢查空卡，不改共享環境。

```bash
source ~/.venv/bin/activate
nvidia-smi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
# 檢查GPU2空閒後，只補本輪尚未完成的條件：
CUDA_VISIBLE_DEVICES=2 bash scripts/run_layerwise_projection_sensitivity.sh \
  results/layerwise_projection_sensitivity --resume
```

需要≥19GiB空閒BF16 GPU；完整命令、分析/驗證與恢復見 [runbook](runbook.md)。新實驗另給新output且不加`--resume`。不要重跑歷史14條件來代替逐層介入。

## Important Files

| 檔案 | 用途 |
|---|---|
| `scripts/layerwise_projection_sensitivity.py` | 單weight介入、全state檢查、BF16還原、原子提交及missing-only resume |
| `scripts/layerwise_projection_sensitivity_config.json` | 固定checkpoint/data revisions、36層×3 projection、2048、64 blocks |
| `scripts/run_layerwise_projection_sensitivity.sh` | venv、GPU inventory、offline與數值環境、stdout/stderr/blocked入口 |
| `scripts/projection_quantization_sensitivity.py` | 重用既有RTN、hidden/head分塊FP32 scorer、paired bootstrap；舊14條件runner保留 |
| `scripts/analyze_layerwise_projection_sensitivity.py` | 有效109條件→CSV、216配對差、8集中度profiles、PNG/PDF、報告 |
| `scripts/verify_layerwise_projection_sensitivity.py` | 重新tokenize、checkpoint/NumPy RTN、獨立CI/集中度及artifact核驗 |
| `tests/test_layerwise_projection_sensitivity.py` | 單介入、還原例外、resume/gate、統計與图表的CPU測試 |
| `results/layerwise_projection_sensitivity/protocol.md` | forward前固定的實驗與分析規約 |
| `results/layerwise_projection_sensitivity/conditions/*.json` | 每條件64筆真實measurement＋完整控制；immutable commits |
| `results/layerwise_projection_sensitivity/attempts/` | 命令、環境、進度、partial、失敗、BF16 hidden cache |
| `results/layerwise_projection_sensitivity/source/` | 產生數據/分析的來源快照；不得改hash冒充新版產生 |
| `results/phase1a_run.json`、`phase1a_tokens.npz`、`phase1a.csv` | 唯讀歷史checkpoint/設定/固定blocks/基準來源 |

`d6f629c` 已將歷史bundle搬到 `results/`。舊manifest中的archive/source名稱是當時provenance，不能改寫為新名字；歷史檔案hash由新run完整保存。

## Architecture Map

歷史run/checkpoint/file hashes → 複用原64×2049 tokens → BF16 model載入且400個state hashes匹配 → BF16 64 blocks精確重現原2048 NLL → lossless BF16 hidden cache → 108次單weight RTN4 → 每次介入/評估/還原全state bitwise檢查＋首末block predictive還原 → 原子condition JSON → 表格/paired CI/集中度/圖表 → 獨立artifact audit → 人工逐條completion audit。

前向eval、batch1、use_cache=False、SDPA FLASH_ATTENTION；FP32 log-softmax/reduction，CPU保留baseline hidden以chunk重建KL reference logits。GPU額外保存原始state副本作uint8 exact比較，這是驗證開銷，不是低bit記憶體優勢。

## Development Workflow

1. 先讀協定及目前manifest/最近attempt；只處理缺少或失敗的條件，不看歷史舊blocked就重跑。
2. 保留歷史檔案與已提交condition。不要改數據、flag、hash使gate假通過。
3. 活動實驗期間不改其來源；修復驗證/文件須保留原始source快照與執行provenance，不把新validator說成產生舊measurements。
4. 改量化/forward需另開結果目錄及相應真模型控制；本輪不要擴模型、資料、bits或scheduler。
5. CPU測試與独立重算不能冒充model measurement；green manifest也不能替代prompt-to-artifact完成稽核。

## Testing

- `python -m unittest discover -s tests -v`：數學、資料/label配對、介入、還原例外、完整性gate、CLI、統計與圖表。
- `bash -n scripts/run_layerwise_projection_sensitivity.sh`：shell syntax。
- 歷史896筆分析只在temporary副本重算，analysis及draws逐byte等於歷史；不改真實歷史。
- 獨立verifier與實際109條件/控制、圖表及completion audit覆蓋本輪成功條件；當前結果與已驗證命令以 `completion_audit.md` 為準。
- 無build/lint/type-check framework或PR/CI/deployment需求，不宣稱其通過。

## Troubleshooting

讀 [runbook故障表](runbook.md#common-failures)。缺GPU/checkpoint/data或OOM時保存`blocked.md`、exact command/error、完成及缺少清單，回報未完成並請使用者提供原資源或 `/goal pause`。不reset共享GPU、不改venv、不換小模型、不縮樣本、不無限重試。

BF16 hidden cache約640MiB，SHA256在BF16條件檔案中；留在磁碟供resume，Git忽略它。新output可由固定checkpoint/blocks再產生cache；缺cache的舊run不可直接宣称續跑已驗證。

## Documentation Freshness Checklist

- [x] README quickstart still works（實測resume與原producer完整執行；未重跑相同measurements）.
- [x] Run commands match the current code.
- [x] Test commands match the current code（22 tests）.
- [x] Important files list is still accurate.
- [x] Architecture map matches the current implementation.
- [x] Troubleshooting section includes recent known failures.

## Independent hardware-real static KV-cache screen

The new `results/kv_cache_precision_pareto/` scope is separate from the historical and layerwise weight experiments above. It freezes Qwen3-4B BF16 weights/compute, exact 2048/8192 prompt IDs, 256 output tokens, homogeneous continuous batches at concurrency 1/4/8/16/32, and provisional gates before model forward. `scripts/kv_cache_precision_pareto.py` directly calls FlashInfer paged prefill/decode kernels with BF16 or locally supported one-byte FP8 caches; it does not implement a scheduler or a BF16 decompression fallback.

Use `scripts/run_kv_cache_precision_pareto.sh` only after `source ~/.venv/bin/activate` and `nvidia-smi`, with one explicitly selected idle GPU. The output includes raw metrics, direct-kernel/cache-byte profiles, capability and error/OOM provenance, `pareto_report.md`, `completion_audit.md`, and the CPU-only `scripts/independent_verifier.py` result. A missing direct compressed kernel is a real block, not a reason to reduce the fixed matrix.
