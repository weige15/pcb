# Runbook

## Project Summary

依 [`../phase1a.md`](../phase1a.md) 測 Qwen3-4B 的 WQ/WK/WV-only 4/8-bit fake quantization；BF16 baseline，512/2048 預測位置，共14條件。不是 packed low-bit inference，沒有加速/記憶體壓縮宣稱。

## Setup

所有命令從 repository root 執行。使用既有 venv，不修改共享 driver/套件：

```bash
source ~/.venv/bin/activate
nvidia-smi
```

實測環境：Python 3.12.3、torch 2.5.1+cu121、Transformers 5.16.1、huggingface-hub 1.29.0、tokenizers 0.23.1、NumPy 1.26.4、safetensors 0.8.0、pyarrow 24.0.0。完整實際版本及硬體以 `results/phase1a_run.json` 為準。未驗證全新 venv 安裝；此處不提供猜測性安裝命令。

需要一張可用 BF16 CUDA GPU；本輪使用 basic-1 RTX 3090 24 GiB。需要本地 Hugging Face hub cache（`HF_HUB_CACHE`，預設 `~/.cache/huggingface/hub`）：

- 模型與 tokenizer：`Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`。
- 資料：`Salesforce/wikitext` revision `b08601e04326c79dfdd32d625aee71d232d685c3`，`wikitext-2-raw-v1/test-00000-of-00001.parquet`。
- 程式 offline，只要求實際必需檔案；不下載替代資產。缺失時須先提供這兩個 revisions 的資產。

## Run

先用 `nvidia-smi` 確認選中卡仍空閒。下列為本輪實際使用的 GPU UUID（別台節點須換成該節點的可用卡）：

```bash
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b \
  bash scripts/run_phase1a.sh results/rerun_$(date -u +%Y%m%dT%H%M%SZ) \
  --config results/phase1a_config.json
```

launcher 自動啟用 venv、檢查 GPU、設 CUDA/離線/FP32數值環境並保存 log。每次需**新 output 目錄**；已有 run manifest/CSV 時拒絕覆寫。不支援中斷續算；保留原證據、解除問題後新目錄重跑。

固定設定 `scripts/phase1a_config.json`；本次執行前快照 `results/phase1a_config.json`。正式 runner 自動先跑一個 block 的 BF16/WQ4 兩長度 smoke，通過才跑全部14條件。完整成功訊息為 `FINISHED status=measured conditions=14/14`；這還須配合完成稽核，不能單靠文字宣稱完成。

輸出在指定 output：`phase1a.csv`（896筆）、`phase1a_run.json`（設定/命令/硬體/控制/module/雜湊）、`phase1a_report.md`（NLL/PPL/KL/CI）、`phase1a_tokens.npz`、`phase1a_analysis.json`、`phase1a_bootstrap_indices.npy`、`tokenizer/`。不保存大量完整 logits 或複製 checkpoint。

## Test

```bash
source ~/.venv/bin/activate
nvidia-smi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
bash -n scripts/run_phase1a.sh scripts/phase1a_preflight.sh
```

10 個 CPU unit tests：獨立 NumPy RTN oracle、零/tie/row 邊界、固定抽樣/資料不足、KL方向/shift對齊/chunk reduction、paired bootstrap、PPL、重複/錯誤CSV、缺控制拒絕report、launcher錯誤記錄。synthetic fixtures 僅存在 temporary directory，不是實驗數據。

真模型 smoke（不是完成的實驗）：

```bash
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b \
  bash scripts/run_phase1a.sh results/smoke_$(date -u +%Y%m%dT%H%M%SZ) --smoke-only
```

成功訊號 `SMOKE PASS` 與 `status=smoke_only_not_complete conditions=0/14`。小樣本還原後 hidden bitwise 相同、NLL相同、KL=0。沒有專案 build、lint 或 type-check framework；不把未配置的工具當已通過。

## Evaluate

```bash
source ~/.venv/bin/activate
nvidia-smi
# 只重算已有實測的統計，不執行新的 forward；寫回相同衍生分析/report。
python scripts/phase1a.py --analyze-only --output results
# 独立重新 tokenize、hash 全 checkpoint、NumPy RTN與 paired CI。
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/verify_phase1a.py --output results \
  --compare-smoke results/smoke_20260908_fixed
```

`--compare-smoke` 可省略；僅在有另一個真實 smoke run 時提供。核對結果寫 `phase1a_verification.json`。這不是重跑所有 GPU forward，也不是完成稽核全部要求的替代品；還需讀 `results/phase1a_audit.md` 的逐項證據與 report 限制。

## Common Failures

| Symptom | Likely Cause | Inspect | Fix |
|---|---|---|---|
| `CUDA unknown error` / cuInit999 | 原basic-2新程序的CUDA初始化失敗 | `results/phase1a_preflight.log`、`doc/debug-report.md` | 請管理員修復或換可用節點；不reset共享GPU |
| `IncompleteSnapshotError` | 指定revision的必要資產未快取；早期runner曾誤要求`.gitattributes` | traceback及run.json路徑 | 現版已限定必需檔案；仍缺權重/資料則pause並提供原資產 |
| tokenizer提示299078超過model_max_length | 整份純文字先tokenize的警告 | 保存token數、block shape | 不需改資料；實際forward只有513/2049 tokens，未傳整串 |
| OOM / 沒有空GPU | 選錯卡或共享資源改變 | `nvidia-smi`、execution log | 停止，保存缺項，要求資源；不減樣本/換模型/改量化條件 |
| `Output already contains a run` | 防覆寫保護 | output中的manifest/CSV | 選新的timestamp目錄；不要刪舊結果冒充續跑 |
| `Blocked/incomplete run cannot produce a results report` | 控制或條件未齊 | run.json status、smoke、interventions | 修正實際原因；不可手改旗標略過 |
| verifier source/artifact hash mismatch | 跑後程式/資產被改 | manifest中的SHA256 | 保留原始實驗版；釐清變更，不重寫舊hash掩蓋 |

## Recovery Steps

1. 讀最新 `results/phase1a_blocked.md`、run manifest 與 execution log；舊basic-2紀錄保留，不代表目前basic-1也不可用。
2. 遇缺GPU/checkpoint/data/OOM，回報未完成，使用者執行 `/goal pause`；不無限重試，不調整正式樣本。
3. 資源狀態改變後才啟用venv並檢查GPU。`bash scripts/phase1a_preflight.sh` 是舊資源預檢入口，固定檢查物理GPU0，不是敏感度runner。
4. 用新 output 跑上述 smoke，通過後再用新 output 正式跑；核對設定與歷史一致。手動中止/外層timeout可能只有launcher log，須補阻塞紀錄，不以CSV已齊作成功。
5. 重新跑 analyze/verifier 並更新逐項audit；不開始下一phase。

## Useful Commands

```bash
git status --short
git diff --check
wc -l results/phase1a.csv
```

本輪背景launcher的PID/exit為 `results/phase1a_formal.pid` / `.exit`；外層限3600秒，log為 `phase1a_formal_launcher.log`。不要對別人的程序發送signal。

## File Locations

| Path | Purpose | Notes |
|---|---|---|
| `phase1a.md` | 使用者實驗規格 | 保留原檔；完成狀態以實際成果/audit為準 |
| `scripts/phase1a.py` | 資料、RTN、控制、計分、統計 | 一次只改一種projection，finally還原 |
| `scripts/phase1a_config.json` | 本輪固定設定 | 不以改設定追求排序 |
| `scripts/run_phase1a.sh` | 可重跑入口 | offline、单GPU、新output |
| `scripts/verify_phase1a.py` | 獨立artifact核驗 | CPU，不代替模型forward |
| `tests/test_phase1a.py` | synthetic單元測試 | 不寫results/假數據 |
| `results/` | 本輪原始/衍生成果與歷史 | 必交三檔在此根目錄 |
| `doc/debug-report.md` | 曾遇錯誤與處理證據 | 保留未明根因與舊主機限制 |

## Operational Notes

- `CUDA_VISIBLE_DEVICES` 支援index或UUID；UUID避免不同主機/PCI排序誤選。不要同時啟兩個runner搶一張卡。
- `HF_HUB_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、TF32關閉、deterministic algorithms、SDPA僅允許FLASH_ATTENTION kernel。不需要API key、網路或服務port。
- scale/log-softmax/metric reduction FP32；前向/還原權重BF16；bootstrap統計FP64。保存baseline最終hidden到CPU，不是啟用KV cache。
- 相同64個blocks共有7×64×(512+2048)=1,146,880次正式計分位置。不是1,146,880個獨立樣本；CI的抽樣單位是block。
- PyTorch CUDA的FP32常數除法會lower成倒數乘法，RTN臨界值可能與CPU直接除法不同。独立verifier明示匹配這項CUDA數值語意後，216個介入hash全部bitwise一致。不要假設換CPU或PyTorch/CUDA版本仍bitwise重現；診斷見audit。
- 正式本機run（含資料準備、smoke、state hashes）實測1055.90秒，峰值allocated7.810GiB/reserved7.930GiB；不當作low-bit壓縮或效能比較。
- 初次snapshot範圍錯誤的smoke輸出保留；不把其0/14當正式數據。

## Last Verified

- Date: 2026-09-08，basic-1。
- Verified commands: venv/BF16 preflight、10 unit tests、`bash -n`、`run_phase1a.sh ... --smoke-only`、正式`run_phase1a.sh results`（exit0、14/14）、`--analyze-only`（四個數據/衍生檔hash未變）、`verify_phase1a.py --output results --compare-smoke results/smoke_20260908_fixed`、`git diff --check`。10個故意破壞控制的manifest亦全數拒絕，原輸出不變。證據見 `results/phase1a_audit.md`。
- Known unverified commands: 全新venv安装、別台GPU/CUDA版本的bitwise重現、網路下載。未新增dependency或改共享環境。
