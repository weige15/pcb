# Runbook

## Project Summary

[`projection_quantization_sensitivity.md`](../projection_quantization_sensitivity.md) 定義 Qwen3-4B 的 WQ/WK/WV-only 4/8-bit grouped RTN 權重量化敏感度實驗：BF16 baseline，64 blocks，512/2048預測位置，共14條件。不測packed GEMM加速、KV lifetime或serving。

## Setup

所有命令從repo root執行：

```bash
source ~/.venv/bin/activate
nvidia-smi
```

使用既有環境，不改共享driver/套件。已完成實驗使用 Python3.12.3、torch2.5.1+cu121、Transformers5.16.1、huggingface-hub1.29.0、tokenizers0.23.1、NumPy1.26.4、safetensors0.8.0、pyarrow24.0.0；完整環境記錄在封存run manifest。未驗證全新venv安裝。

需要一張BF16 CUDA GPU，以及本地 `HF_HUB_CACHE`（預設 `~/.cache/huggingface/hub`）：

- 模型/tokenizer：`Qwen/Qwen3-4B`，revision `1cfa9a7208912126459214e8b04321603b3df60c`。
- 資料：`Salesforce/wikitext`，revision `b08601e04326c79dfdd32d625aee71d232d685c3`，`wikitext-2-raw-v1/test-00000-of-00001.parquet`。

程式offline；缺資產時停止，不下載替代模型或縮減樣本。

## Run

先確認所選GPU空閒；可指定index或UUID：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_projection_quantization_sensitivity.sh
```

預設output：`results/projection_quantization_sensitivity/`。重跑需新目錄：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_projection_quantization_sensitivity.sh \
  results/projection_quantization_sensitivity/run_$(date -u +%Y%m%dT%H%M%SZ)
```

固定設定：`scripts/projection_quantization_sensitivity_config.json`；可用 `--config` 指向先前保存的 `experiment_config.json`。程式在正式14條件前先跑一個已選block的兩個contexts，檢查BF16/WQ4、計分對齊及還原。成功訊息 `FINISHED status=measured conditions=14/14` 仍須配合實際artifact與控制核對。

| 新output檔名 | 內容 |
|---|---|
| `block_metrics.csv` | 896筆逐block NLL、ΔNLL、KL與輸入/labels識別 |
| `run_manifest.json` | 命令、環境、revisions、hash、module、控制與進度 |
| `experiment_config.json` | 執行前設定快照 |
| `sensitivity_report.md` | 摘要、paired CI、結論與限制 |
| `sensitivity_analysis.json` | 未四捨五入的摘要與24個對比 |
| `sampled_token_blocks.npz` | 64個2049-token blocks與原始索引 |
| `paired_bootstrap_indices.npy` | seed42的2000×64配對重抽索引 |
| `tokenizer/` | tokenizer設定與資產 |
| `execution_<UTC>.log` / `blocked.md` | stdout/stderr與失敗紀錄 |

不支援中斷續算；保留原output，解除問題後另選目錄重跑。CLI會拒絕覆寫已存在的manifest/CSV。

## Test

```bash
source ~/.venv/bin/activate
nvidia-smi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
bash -n scripts/run_projection_quantization_sensitivity.sh scripts/check_projection_quantization_resources.sh
```

12 tests：原有RTN、row/zero/tie、資料不足、KL方向/chunk/shift、paired bootstrap、PPL、CSV與控制gate，加上改名後CLI/防覆寫及封存完整性回歸。CPU使用真實896筆的暫存副本重算後，analysis JSON與bootstrap draws和歷史版本逐byte相同；不寫入封存、不執行模型forward。

需要真模型smoke時（此次命名重構沒有重跑）：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_projection_quantization_sensitivity.sh \
  results/projection_quantization_sensitivity/smoke_$(date -u +%Y%m%dT%H%M%SZ) --smoke-only
```

smoke成功仍為0/14正式條件。沒有build/lint/type-check framework，不宣稱其通過。

## Evaluate

針對**新命名格式**的run，先啟用venv並檢查GPU：

```bash
python scripts/projection_quantization_sensitivity.py --analyze-only \
  --output results/projection_quantization_sensitivity
python scripts/verify_projection_quantization_sensitivity.py \
  --output results/projection_quantization_sensitivity
```

如果有獨立smoke output，可另加 `--compare-smoke <smoke目錄>`。analyze只重算已有measurement，verifier檢查source/artifact hashes、重新tokenize、NumPy RTN及CI，輸出 `artifact_verification.json`；兩者都不是新GPU實驗。

**歷史結果保留舊格式，不交給新格式parser。** 原verifier與原source已一起封存，`ROOT`會指向當時source快照。若要重做舊artifact audit，先複製到暫存目錄避免改封存：

```bash
snapshot=archive/qwen3_projection_quantization_sensitivity_2026-09-08
tmp=$(mktemp -d)
cp -a "$snapshot/results/." "$tmp/"
python "$snapshot/scripts/verify_phase1a.py" --output "$tmp" \
  --compare-smoke "$tmp/smoke_20260908_fixed"
```

原始records中的cwd/絕對命令是當時執行紀錄，不事後改成新名稱；新實驗請用上方新入口。封存驗證器 `--help`、所有source hashes及artifact bytes在本次已核對；上述完整checkpoint audit未因改名重跑。

## Common Failures

| Symptom | Likely Cause | Inspect | Fix |
|---|---|---|---|
| 找不到舊script或舊results路徑 | 檔案改名／歷史已封存 | [命名對照](onboarding.md#important-files) | 新run改用新入口；舊資料配封存source，不改manifest假裝新版產生 |
| `CUDA unknown error` / cuInit999 | 舊basic-2新程序初始化失敗 | [歷史診斷](debug-report.md) | 管理員修復或換節點，不reset共享GPU |
| `IncompleteSnapshotError` | 必需的指定revision資產未快取 | log/model/data路徑 | 提供原資產；現版已不要求非必需`.gitattributes` |
| tokenizer整串長度warning | 先tokenize整份文本 | 保存blocks shape | 實際forward只有513/2049 tokens，未將整串傳模型 |
| OOM／沒有空GPU | 資源改變或選錯卡 | `nvidia-smi`、execution log | 保存blocked並停止，不減sample／換模型 |
| `Output already contains a run` | 防覆寫保護 | manifest與CSV | 指定新的output，不刪證據假裝續跑 |
| report gate拒絕 | 條件/必要控制未齊 | `run_manifest.json` | 修正實際原因，不改旗標繞過 |
| source/artifact hash mismatch | 修改了產生數據的source/asset，或用錯版本 | 對應run manifest與封存 | 保留對應版本，不重寫hash掩蓋 |

## Recovery Steps

1. 只看當次output的manifest、execution log和`blocked.md`，不要把封存中的舊阻塞當目前狀態。
2. 缺GPU/checkpoint/data、OOM時回報未完成，由使用者 `/goal pause`；資源變動後才再試，不無限重試。
3. `bash scripts/check_projection_quantization_resources.sh` 是資源檢查，固定GPU0；不是敏感度runner，也不能代替64 blocks/14條件。
4. 解除問題後用新output跑smoke，再正式執行；之後核對數據、控制與CI。不自動開始後續實驗。

## Useful Commands

```bash
git status --short
git diff --check
python scripts/projection_quantization_sensitivity.py --help
python scripts/verify_projection_quantization_sensitivity.py --help
```

封存中的PID/exit/log只代表歷史，不以舊PID控制目前程序。

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `projection_quantization_sensitivity.md` | 目前規格 | 只改名稱，實驗條件不變 |
| `scripts/projection_quantization_sensitivity.py` | runner及分析 | 數學與forward未改 |
| `scripts/projection_quantization_sensitivity_config.json` | 固定設定 | 與原run config相同 |
| `scripts/verify_projection_quantization_sensitivity.py` | 新格式artifact核驗 | 不支援混用舊檔名 |
| `tests/test_projection_quantization_sensitivity.py` | 測試與命名回歸 | 暫存fixture不冒充實測 |
| `results/` | 新run輸出／本次refactor驗證 | 不含偽造新版14條件結果 |
| `archive/qwen3_projection_quantization_sensitivity_2026-09-08/` | 完整舊實驗bundle | 51個檔案逐byte保留，含舊source/docs/results及checksums |

## Operational Notes

- 單GPU、eval、batch1、use_cache=False、SDPA固定FLASH_ATTENTION kernel、TF32關閉、deterministic algorithms；不需網路/API key。
- scale/log-softmax/metric reduction FP32、前向BF16、bootstrap FP64。baseline hidden存CPU，不保存大量完整logits。
- CUDA常數除法以FP32倒數乘法實作，RTN臨界值可能與CPU直接除法不同；不保證換PyTorch/CUDA/CPU後bitwise相同。舊完整audit已嚴格驗證216個介入hash。
- 原run在basic-1 RTX3090 24GiB耗1055.90秒，峰值allocated7.810GiB；這是歷史資源紀錄，不是壓縮或加速比較。
- CI抽樣單位是block；不把重算或smoke當獨立樣本，不把跨零解釋為已證明等價。

## Last Verified

- Date: 2026-09-08，命名重構。
- Verified commands: 12 unit/regression tests、兩個新CLI的`--help`、封存verifier的`--help`、新shell入口語法、source/archive hashes、896筆資料CPU分析重算、`git diff --check`。log：`results/naming_refactor_tests.log`。
- Known unverified commands: 改名後完整GPU run/smoke未重跑（僅命名與路徑變更）；全新venv安裝、跨硬體bitwise重現未驗證。原版GPU成功與完成稽核保留在封存。
