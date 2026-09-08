# Runbook

## Project Summary

Qwen3-4B projection-weight quantization sensitivity。歷史全層實驗14條件仍保留；本輪逐層實驗固定2048預測位置，每次只改一層的一個WQ/WK/WV weight為group128 RTN4，共108介入+BF16。協定及交付都在 `results/layerwise_projection_sensitivity/`。不擴展scheduler。

## Setup

所有命令從repo root執行：

```bash
source ~/.venv/bin/activate
nvidia-smi
```

用既有環境，不安裝/更新共享套件或driver。已實跑環境：Python3.12.3、torch2.5.1+cu121、Transformers5.16.1、huggingface-hub1.29.0、tokenizers0.23.1、NumPy1.26.4、safetensors0.8.0、matplotlib3.11.0；完整runtime見attempt/analysis manifests。全新venv安裝未驗證。

本地資產（offline、不下載替代品）：

- Qwen/Qwen3-4B model/tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`。
- Salesforce/wikitext、wikitext-2-raw-v1 test revision `b08601e04326c79dfdd32d625aee71d232d685c3`。
- 歷史 `results/phase1a_run.json`、`phase1a_config.json`、`phase1a_tokens.npz`、`phase1a.csv` 與manifest列出的原始artifact。
- 一張≥19GiB空閒的BF16 CUDA GPU。本輪GPU2 RTX3090的UUID為 `GPU-74d97f46-6284-1055-698a-e2db4e9c744b`；每次先看inventory，不假設仍空閒。

## Run

### 逐層：只補缺少條件

```bash
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b \
  bash scripts/run_layerwise_projection_sensitivity.sh results/layerwise_projection_sensitivity --resume
```

已提交conditions不覆寫，不重跑。完成訊息 `FINISHED 109/109 status=measured` 仍須獨立audit及人工要求對照。全部完成的resume不得執行model forward。

新目錄完整重跑：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_layerwise_projection_sensitivity.sh \
  results/layerwise_projection_sensitivity/rerun_$(date -u +%Y%m%dT%H%M%SZ)
```

可先加 `--max-conditions 1`，實跑BF16與一個完整64-block介入，再移除該參數以`--resume`補其餘；總目標仍109，不縮blocks。活動run持有output file lock，第二個writer不得同時寫。

### 歷史全層（本輪不重跑）

舊結果在 `results/phase1a*`，不是已不存在的archive目錄。新執行全層14條件可用：

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/run_projection_quantization_sensitivity.sh \
  results/projection_quantization_sensitivity/new_run
```

舊runner不支援resume，新output拒絕覆寫；`--smoke-only`是控制不是正式條件。不要把新命名parser直接套到phase1a舊檔名。歷史原始命令/來源SHA保持當時內容；若需當時source，查Git history，不事後改manifest。

## Test

```bash
source ~/.venv/bin/activate
nvidia-smi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
bash -n scripts/run_layerwise_projection_sensitivity.sh scripts/run_projection_quantization_sensitivity.sh
```

CPU synthetic fixtures只放temporary dirs；涵蓋RTN zero/ties/group、scoring/KL方向/shift/chunks、PPL、paired bootstrap、單介入/例外還原、state bitwise、條件/row/cache/gate、集中度/reranking/zero/ties及CLI/圖表。真實896筆只做temporary分析回歸，不重跑舊GPU實驗。

沒有build/lint/type-check framework；不宣稱這些或PR/CI通過。

## Evaluate

109條件與必要控制齊備後，先啟用venv並檢查GPU，再執行CPU分析：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/analyze_layerwise_projection_sensitivity.py \
  --output results/layerwise_projection_sensitivity
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/verify_layerwise_projection_sensitivity.py --output results/layerwise_projection_sensitivity
```

分析器檢查完整條件、paired identity、控制與來源；輸出逐block表、109摘要、216配對contrast/metric、8集中度profiles、2000×64 draws、3組PNG/PDF、報告及analysis manifest。

Verifier重新讀checkpoint全398參數、NumPy重建108 RTN4權重、重新tokenize每一保存block、独立重算paired CI與重新排名的集中度；另外比對歷史/source/artifact hashes。`artifact_verification.json`不是唯一完成證據：須人工查看圖表及 `completion_audit.md` 的逐要求對照。

## Common Failures

| Symptom | Likely Cause | Inspect | Fix |
|---|---|---|---|
| `Existing run requires --resume` | 防覆寫 | output manifest | 舊run加`--resume`，全重跑用新目錄 |
| file lock error | 同output已有活動writer | 最新log與本次PID | 等原writer完成；不要殺別人的process或刪lock繞過 |
| source/hash mismatch | 程式/歷史/cache/數據改變或錯output | manifest與source快照 | 查明差異，不改hash掩蓋；來源快照可重現原producer |
| missing/corrupt BF16 hidden cache | 大型cache沒隨Git保存或檔案損毀 | BF16.json的hidden_cache與磁碟 | 提供原cache；另開新output全重跑可再生成，不能將缺cache宣稱通過resume |
| `<19GiB`／OOM | GPU忙或容量不足 | nvidia-smi、attempt/error | 保存阻塞，提供足夠空GPU，不減樣本/換模型 |
| checkpoint/data缺檔 | 固定revision cache不可用 | 歷史manifest的絕對asset path | 提供原revision，不替代、不自動下載 |
| BF16不匹配歷史 | runtime/kernel/資產差異 | versions、TF32/CUBLAS/SDPA、hash | 先診斷，不放宽為allclose假通過 |
| partial資料但無condition提交 | 該條件未完成或control失敗 | attempts/*.partial.json/failed_controls.json | 保留partial；解決問題後只補該缺少條件 |
| analyzer拒絕不完整run | 條件或控制缺項 | manifest、conditions、attempts | 補真實缺項，不能改status繞過 |

## Recovery Steps

1. 檢查**本輪**最近log/attempt/blocked；舊phase1a blocked只是歷史，不是目前資源狀態。
2. 不覆寫或刪除已提交conditions、歷史檔案、failed attempt。後來attempt失敗不使之前已通過完整還原的原子條件失效。
3. 缺資源/OOM就保存command/error與completed/missing清單，回報未完成並請使用者提供GPU/原資產或 `/goal pause`。不無限重試。
4. 保持相同checkpoint/config/producer source；原資產可用後跑`--resume`，由完整性驗證選出缺少條件。若只有CPU validator已更新，可核驗/分析完整舊run；補missing條件仍拒絕producer source drift，需使用該run保存的`source/scripts/layerwise_projection_sensitivity.py`及原attempt環境，不改原hash。
5. 全部完成後重算analysis、獨立verifier、no-op resume與逐條completion audit；不開始其他實驗。

## Useful Commands

```bash
git status --short
git diff --check
python scripts/layerwise_projection_sensitivity.py --help
python scripts/analyze_layerwise_projection_sensitivity.py --help
python scripts/verify_layerwise_projection_sensitivity.py --help
```

`.pid`和`.exit`只記當次程序，不能用歷史PID控制目前無關程序。先檢查PID命令與啟動時間。

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `scripts/*layerwise_projection_sensitivity*` | runner/config/launcher/analyzer/verifier | 只本輪逐層實驗 |
| `results/layerwise_projection_sensitivity/run_manifest.json` | 設定、來源、歷史hash、完成条件 | 初始producer快照不作事後改寫 |
| `conditions/BF16.json`、`conditions/L*_W?4.json`（本輪目錄內） | 原始measurement與控制 | 每檔64列，全部109檔 |
| `attempts/`（本輪目錄內） | actual command/environment、partial、cache | blocked不冒充成功 |
| `block_metrics.csv`、`layer_summary.csv`、`paired_comparisons.csv` | 原始合表與統計 | 6976/109/216 rows |
| `concentration.json`、`figures/`、`sensitivity_report.md` | 少數層集中度、图表、有限結論 | ALL不是聯合量化損害 |
| `source/`、`analysis_manifest.json` | source快照與分析/plot runtime | byte-identical圖表只在本環境驗證 |
| `artifact_verification.json`、`completion_audit.md` | 自動独立核驗及人工要求對照 | 兩者都要核對 |
| `results/phase1a*`、`results/tokenizer/` | 唯讀歷史資產 | 本輪不改寫 |

## Operational Notes

- 單GPU、BF16、eval、batch1、no cache、FLASH_ATTENTION、deterministic、TF32 off；launcher固定CUBLAS workspace與4 CPU threads、HF offline。
- 原始state GPU副本約7.5GiB是exact control開銷；baseline hidden CPU cache約640MiB，Git忽略但磁碟保留且hash鎖定。不要把這當量化節省。
- CUDA scalar除法使用FP32倒數乘法；NumPy oracle顯式匹配此數值語意並嚴格比對BF16 SHA，未以allclose放寬RTN。
- CI單位為固定64 blocks，2000 resamples不是獨立實驗數。跨零不證等價，多重比較未校正。
- 集中度使用非負平均score；保留原始負ΔNLL，ALL是獨立介入分數之和而非可加性假設。
- 本輪最終時長、實際峰值、已完成條件與測試證據見completion audit；不以歷史run時長替代本輪量測。

## Last Verified

- Date: 2026-09-08。
- Verified commands: 109/109真實條件（兩次producer分別新增2及107）、新版no-op resume（skip109/new0/無forward）、22個CPU tests、shell syntax、三個CLI help；獨立398參數/108 RTN/6976 rows/216配對/8集中度核验；13個分析/圖表artifact重算逐byte相同。完整對照見本輪 `completion_audit.md`。
- Known unverified commands: 全新venv安裝、跨硬體/套件bitwise重現；不能假設換環境圖表PNG/PDF逐byte相同。

## Hardware-real static KV-cache Pareto screen

This independent scope lives under `results/kv_cache_precision_pareto/`; it does not alter historical results or the completed layerwise experiment. The frozen protocol/config are `results/kv_cache_precision_pareto/protocol.md` and `results/kv_cache_precision_pareto/experiment_config.json`. Model weights and compute remain BF16. Only locally supported direct FlashInfer paged prefill/decode cache payloads are system points; fake quantization and full-cache BF16 restoration are excluded.

Before any model command:

```bash
source ~/.venv/bin/activate
nvidia-smi
```

Select one idle GPU explicitly and run:

```bash
CUDA_VISIBLE_DEVICES=<idle-GPU-index-or-UUID> bash scripts/run_kv_cache_precision_pareto.sh --phase all
CUDA_VISIBLE_DEVICES='' python scripts/analyze_kv_cache_precision_pareto.py --output results/kv_cache_precision_pareto
CUDA_VISIBLE_DEVICES='' python scripts/independent_verifier.py --output results/kv_cache_precision_pareto
```

The launcher prepares exact prompt IDs first, then records BF16 cached-decode alignment, BF16/FP8 smoke, quality NLL/KL and retrieval EM, every fixed concurrency (1/4/8/16/32) and repeat, per-token timestamps, cache tensor bytes, direct-read provenance, vLLM/4-bit capability errors, GPU inventory, OOM and other errors. `pareto_report.md` reports provisional GO only for a non-dominated compressed point passing every frozen gate; otherwise it reports provisional NO-GO. If direct compressed attention is unavailable, preserve `blocked.md`, do not reduce the matrix or use a BF16 decompression path, and request `/goal pause` plus the missing resource.
