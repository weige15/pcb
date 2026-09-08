# Debug Report

## 最新驗證（2026-09-08）

本輪basic-1正式run已14/14、exit0；獨立verifier的數值語意問題修正後，398個原始parameter hashes、216個量化hash、252筆module紀錄、所有64個block token IDs及24個paired CI都通過。`phase1a_final_checks_20260908T103535Z.log`另證明10 unit tests、10個失敗控制gate與統計重算hash一致。完整逐項證據見 `results/phase1a_audit.md`。正式runner及原始measurement未在稽核修正時更改。舊basic-2的error999根因仍未知，不是本輪basic-1完成的必要待辦；未修改共享driver/venv。

以下按當時診斷階段保留失敗與修正理由；「尚未完成」指當時狀態。

## 2026-09-08 10:32 UTC — 獨立 RTN audit 的 CPU/CUDA rounding 差異

- 正式14條件已exit0，模型runner未失敗；但 `python scripts/verify_phase1a.py --output results --compare-smoke results/smoke_20260908_fixed` 在第一個WK8量化hash不符而exit1，所以尚未宣稱完成。
- 預期CPU NumPy重建能驗證每個已記錄介入；實際 `model.layers.0.self_attn.k_proj.weight` 的283/2,621,440個BF16值不同（max abs 0.0009765625）。原始checkpoint hash一致。
- 根因證據：`results/phase1a_rtn_audit_diagnosis_20260908T103156Z.log`。同一原始tensor：NumPy直接 FP32 `max/127` vs CUDA的scale有1021組不同；NumPy `max * float32(1/127)` vs CUDA為0組不同，生成量化權重也0個不同且SHA256等於run manifest。純PyTorch CUDA再次量化也與原記錄bitwise一致。典型normalized值63.499996 vs63.5000038落在RTN臨界值兩側。
- 分類：**跨運算後端的FP32常數除法lowering差異／獨立verifier過強的跨後端假設**。不是缺檔、GPU不可用、OOM、permission、資料錯誤、模型被污染或正式結果無法重現。CUDA此操作以FP32倒數乘法實作，runner仍是規格要求的FP32 scale與相同RTN規則。
- 最小修正（先診斷後執行）：不改正式runner、不改既有結果；獨立NumPy audit將scale明確寫成FP32倒數乘法以匹配已釘選PyTorch CUDA實作，繼續嚴格要求**所有**實際量化BF16 hash bitwise一致。補充CPU/CUDA臨界值不可假設bitwise相同的可重現性限制；若其餘module仍不一致則繼續診斷，不能放寬成僅allclose假裝通過。

## 2026-09-08 10:05 UTC — 新節點恢復後的 runner 檔案範圍錯誤

- 現在 host 為 **basic-1**，不是下方歷史中的 basic-2。`results/phase1a_preflight_20260908T095523Z.log` 實測 cuInit=0、BF16 matmul PASS；沒有宣稱 basic-2 已修好。
- cwd `/nfs/home/s314511048/pcb`，同一 `~/.venv`（Python 3.12.3、huggingface-hub 1.29.0）。命令：`CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b bash scripts/run_phase1a.sh results/smoke_20260908 --smoke-only`。
- 預期讀取已釘選的模型/tokenizer/test parquet；實際在 `snapshot_download(..., local_files_only=True)` 拋 `IncompleteSnapshotError: 1 file(s) are missing (.gitattributes)`，尚未載入模型或跑任何條件。原 traceback、命令與 manifest 保存於 `results/smoke_20260908/`。
- 分類：**runner 檔案範圍設定 / Hugging Face API 使用錯誤**；不是 CUDA、OOM、permission、shell、distributed 或 import 問題。資產本體未缺：三個 shards、config、tokenizer 存在且先前 header/index 檢查通過；完整權重運算仍待驗證。無 concurrency 根因證據。
- 假設與證據：①需要的 checkpoint 缺失——traceback 僅列 `.gitattributes`，檔案盤點反對；②runner 不當要求整個 repo 完整——已讀 `_snapshot_download.py:321`，確實依 cached tree 檢查全部檔案，且支援 `allow_patterns`，符合症狀。
- 最小修正（執行前診斷）：snapshot 請求指定必要 model/tokenizer 檔案及唯一 test parquet 的 `allow_patterns`；不下載/偽造 `.gitattributes`，不跳過實際權重檢查，不改共享 venv。舊失敗證據保留，在新 output 重新驗證 smoke。
- 同輪 unit test 最初有一處 FP32 sum/mean 與 cross-entropy 相差 5.05e-7，屬測試十進位位數判定過緊；改用明示 `atol=rtol=1e-6`，8 tests PASS。量化/計分公式未因此改動。
- Reviewer（read-only）指出 report gate 與 launcher 早期失敗記錄缺口；須在正式跑前補控制 evidence gate、早期阻塞記錄，以及保留 config/mode 的 rerun 命令。這些不改實驗條件。

以下為舊節點的完整歷史診斷，保留原文：

## Symptom

Phase 1a 在資源預檢階段阻塞，尚未載入 Qwen3-4B 或執行任何敏感度條件。`nvidia-smi` 能列出 GPU 0 的空閒記憶體，但目前程序不能初始化 CUDA。

## Reproduction Command

Working directory: `/nfs/home/s314511048/pcb`
Shell: `bash`
Runtime: Python 3.12.3
Environment: `source ~/.venv/bin/activate`，使用 `/nfs/home/s314511048/.venv/bin/python`。

Relevant environment variables:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

其餘相關環境、版本及完整 traceback 見 [`../results/phase1a_preflight.log`](../results/phase1a_preflight.log)。

首次保存的診斷執行命令：

```bash
mkdir -p results
set -o pipefail
bash scripts/phase1a_preflight.sh 2>&1 | tee results/phase1a_preflight.log
```

保留既有 log 的再次驗證命令：

```bash
set -o pipefail
bash scripts/phase1a_preflight.sh 2>&1 | tee "results/phase1a_preflight_$(date -u +%Y%m%dT%H%M%SZ).log"
```

## Expected Behavior

Driver API `cuInit(0)` 回傳 0；PyTorch 可見所選 GPU 並能完成 BF16 CUDA matmul。這只是資源門檻，不是實驗完成證據。

## Actual Behavior

**使用者要求的純 PyTorch 複驗：2026-09-08 09:52 UTC。** 啟用同一 venv、先執行 `nvidia-smi`；以 GPU UUID 分別限定空閒 GPU 0、1，各啟動一個全新 Python 程序。不先呼叫 ctypes/cuInit，直接執行 `torch.cuda.init()`，原定接著配置 256×256 FP32/BF16 tensors、matmul、同步並核對值。**兩個程序皆在 `torch.cuda.init()` 失敗，exit 1；配置與矩陣運算均未執行。** PyTorch 為 2.5.1+cu121。確切 Python source、兩個命令、UUID、硬體狀態與完整 traceback 已保存至 [`../results/pytorch_gpu_probe_20260908T095238Z.log`](../results/pytorch_gpu_probe_20260908T095238Z.log)。

`nvidia-smi` 仍回報 GPU 2 handle Unknown Error，GPU 7 仍有既存 Python 程序與利用率；沒有測試或干擾 GPU 7。這些證據只能確認目前新程序無法初始化 CUDA，**不能證明 GPU 0/1 硬體損壞，也不能認定 GPU 2 是根因**。下一步仍是由管理員核對 NVIDIA/Xid kernel logs、driver/device 狀態與程序裝置存取；未 reset、重啟或更動共享環境。此為單次使用者要求的診斷，不是恢復 Phase 1a 實驗。

**前次複驗：2026-09-08 09:49 UTC，round 2。** 同一 host、venv 與 GPU 0，單次 CUDA-only probe 再現 `cuInit(0) = 999`、CUDA unavailable / device count 0、`get_device_name(0)` RuntimeError；exit 1。GPU 0 inventory 為 24124 MiB free、0% utilization，仍不能作 BF16 運算。未重做資產盤點、未嘗試修復共享環境。確切命令及完成缺項保存在 [`../results/phase1a_blocked.md`](../results/phase1a_blocked.md) 的 round 2 區段；原始 stdout/stderr：[`../results/phase1a_cuda_recheck_20260908T094923Z.log`](../results/phase1a_cuda_recheck_20260908T094923Z.log)。本輪沒有新增根因證據，以下分類與假設維持不變。

首次診斷：2026-09-08 09:42 UTC，host `basic-2`：

- GPU 0：RTX 3090，24576 MiB 總記憶體、24124 MiB 空閒、0% utilization；driver 580.159.03。
- `nvidia-smi` 同時印出 GPU 2 / PCI `0000:1F:00.0` 的 Unknown Error，**仍回傳 exit 0**。
- 直接呼叫 `libcuda.so.1` 的 `cuInit(0)`：999 / `CUDA_ERROR_UNKNOWN`。
- PyTorch 2.5.1+cu121：`is_available() == False`、`device_count() == 0`，裝置存取拋出 RuntimeError。
- 預檢正確以 exit 1 回報 BLOCKED；尚未分配模型，並非模型 OOM。
- 初次以 GPU 0 UUID 限定的獨立 PyTorch probe 亦失敗；之後只做了上述一次 PCI 順序 / index 0 的保存診斷，沒有逐卡無限重試。
- 本地 checkpoint 三個 shard 的 headers 可讀、key 集合與 index 一致；tokenizer 檔案存在；WikiText test parquet 可讀（4358 rows，text 無 null）。未驗證完整模型載入、tokenization 或足夠的 blocks。

## Error Log

完整原始輸出：[`../results/phase1a_preflight.log`](../results/phase1a_preflight.log)。關鍵片段：

```text
Unable to determine the device handle for GPU2: 0000:1F:00.0: Unknown Error
CUDA driver cuInit: 999 b'CUDA_ERROR_UNKNOWN'
torch CUDA available: False
torch device count: 0
RuntimeError: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero.
PREFLIGHT: BLOCKED {"cuda_bf16": false, "cached_asset_checks": true, "entry_point_imports": true}
Preflight exit: 1
```

## Failure Layer Classification

Most likely layer:

* Command problem: no — Python 已可執行。
* Permission problem: unknown — device nodes 存在且一般節點可讀寫；未查證 driver/cgroup 存取限制。
* Shell/script invocation problem: no — 已啟用指定 venv。
* Environment problem: possible — CUDA driver/runtime 環境仍可能有問題。
* Dependency problem: no evidence — 套件存在，入口 imports 成功；完整模型相容性未驗證。
* Python/package/import problem: no for observed failure — 直接 driver API 也失敗。
* GPU/CUDA problem: yes。
* Distributed/torchrun problem: no — 非分散式工作。
* Filesystem/path problem: no for inspected files。
* Data/checkpoint/model file problem: no evidence — 基礎檔案檢查通過，完整驗證未完成。
* Code logic problem: no evidence — 尚未執行實驗程式。
* Configuration problem: possible — library resolution 等尚未查證。
* Resource problem: yes — 無可實際使用的 CUDA 裝置。
* Concurrency/race problem: unknown — 共享 host，未更動其他程序。
* Unknown/insufficient evidence: yes — CUDA error 999 的底層原因未確認。

Final classification: CUDA driver/device initialization failure；不是已證明的模型、資料或量化邏輯錯誤。

## Hypotheses

### Hypothesis 1: 主機 driver/device 狀態異常

Why it could explain the symptom: GPU 2 handle 異常，driver 初始化也失敗。
Evidence for: `nvidia-smi` Unknown Error + 獨立 `cuInit` 999；PyTorch-only probe 同樣失敗。
Evidence against: GPU 0 inventory 正常，不足以證明 GPU 2 是所有 CUDA 失敗的原因。
How to verify: 管理員檢查 kernel NVIDIA/Xid logs、driver 與裝置狀態，在維護後重新執行預檢。

### Hypothesis 2: 程序環境或 CUDA library/device access 不一致

Why it could explain the symptom: 套件 CUDA build 為 12.1，LD_LIBRARY_PATH 包含其他 CUDA 路徑；存取政策亦可能不同。
Evidence for: 初始化失敗，原始環境已保存。
Evidence against: CUDA 12.1 配較新 driver 本身不是不相容證據；device nodes 存在；GPU UUID 與 index 兩種選擇都未解決。
How to verify: 管理員核對實際載入的 libcuda、kernel driver、裝置存取政策，或在已知正常節點執行同一預檢。不要猜測性升降級共享 venv。

## Most Likely Root Cause

可確定的失敗層是目前 `basic-2` 的 CUDA 初始化；主機 driver/device 狀態异常是較受證據支持的假設，但 error 999 不能單獨定位底層根因，也不能斷言 GPU 2 是原因。checkpoint、資料與量化尚未參與失敗。

## Minimal Fix

請提供可通過 BF16 CUDA 預檢的 GPU 節點，或請管理員處理本機 CUDA/driver/device 異常後通知重試。本輪未 reset GPU、重啟 host、重載 driver、使用 sudo、修改共享 venv 或終止其他使用者程序；未改用 CPU、小模型或縮小正式樣本。

## Verification

Round 2 的單次複驗，以及使用者後續要求的 GPU 0/1 純 PyTorch 獨立程序測試，均在 CUDA 初始化失敗。FP32/BF16 配置與 matmul 成功路徑未驗證；不能據此判定實體硬體損壞。檔案核對仍無實驗 runner、CSV、run manifest 或結果 report，正式條件完成數 0/14。不要在未收到資源狀態改變資訊前循環重試。

資源恢復後才執行上述 timestamped 預檢命令。預期 `cuInit` 回傳 0、`BF16 CUDA matmul: PASS`、資源檢查 exit 0；仍須再驗證完整 Qwen3 載入與資料準備，固定正式設定，實作並完成小樣本 BF16/WQ4 和所有正式條件。

本輪 `bash -n scripts/phase1a_preflight.sh` 及內嵌 Python 語法檢查通過；實際預檢失敗路徑已執行並回傳 1。GPU 成功路徑未驗證。這些檢查不代表 Phase 1a 成功。
