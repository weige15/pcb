# Phase 1a — 阻塞歷史與恢復進度

## 最新進度 — round 3 已交付（2026-09-08 10:38 UTC）

**本輪阻塞已解除，實驗交付完成。** 正式run 14/14條件、896真實逐block數據，exit0；398個原始checkpoint parameter hashes、216種module/bit量化BF16 hashes、252筆介入紀錄、全部token IDs與24個paired CI均完成核對。獨立smoke與正式結果精確一致，10 unit tests與10個失敗控制gate通過。完成稽核：[phase1a_audit.md](phase1a_audit.md)；數據/設定/結論：[phase1a.csv](phase1a.csv)、[phase1a_run.json](phase1a_run.json)、[phase1a_report.md](phase1a_report.md)。

最後遇到的獨立verifier CPU/CUDA FP32常數除法差異已診斷：NumPy匹配CUDA的FP32倒數乘法後，全216量化hash仍嚴格bitwise通過；沒有修改正式runner或數據，詳見audit與debug report。不是缺GPU或OOM，目前不需要pause。未做下一phase。

以下為本輪進行中與舊阻塞歷史，原文保留，勿把舊0/14或pause要求當最新狀態。

## round 3 啟動紀錄（2026-09-08 10:12 UTC）

- 本輪實際 host 是 **basic-1**（舊失敗 host 是 basic-2）；BF16 CUDA 預檢已通過，見 `phase1a_preflight_20260908T095523Z.log`。不是宣稱舊主機修好。
- 已新增 runner / 固定 config / launcher / 10 個 unit tests；已修正首次 smoke 的 snapshot 必需檔案範圍錯誤（缺 `.gitattributes` 非缺 checkpoint）。失敗輸出保留於 `smoke_20260908/`；診斷見 `../doc/debug-report.md`。
- `smoke_20260908_fixed/phase1a_run.json`：兩個 contexts 的 BF16/WQ4 小樣本、CausalLM 對齊、非目標 hash、還原後 baseline bitwise 一致均通過。這不是正式結果。
- 299,078 tokens → 145 個完整2049-token blocks → 固定seed42取64；正式條件使用 GPU2 UUID `GPU-74d97f46-6284-1055-698a-e2db4e9c744b`。
- 10:12 UTC 啟動 `CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b bash scripts/run_phase1a.sh results`，外層 timeout 3600秒；進度見 `phase1a_formal_launcher.log` 與 `phase1a_run.json`。**目前正式跑進行中，尚未宣稱完成**；完成後仍須核對896筆、必要控制、原始資產、CI與可重跑命令。
- 沒有重做 basic-2 的失敗 GPU probe，也沒有擴大模型、資料、條件或下一 phase。

以下保留前兩輪與使用者指定診斷的歷史原文；其中「未完成／請pause」描述的是當時阻塞狀態：

## 使用者要求的 PyTorch 測試（2026-09-08 09:52 UTC）

依「GPU 壞掉了嗎？用 PyTorch 試試看」要求，啟用 venv 並檢查 `nvidia-smi` 後，分別以 UUID 限定 GPU 0、1，在兩個全新程序直接呼叫 `torch.cuda.init()`（未先呼叫 ctypes）。兩者均拋 CUDA unknown error、exit 1，尚未到 FP32/BF16 tensor 配置及 matmul。命令、測試 source 與 traceback：[`pytorch_gpu_probe_20260908T095238Z.log`](pytorch_gpu_probe_20260908T095238Z.log)；診斷更新於 [`../doc/debug-report.md`](../doc/debug-report.md)。

這證明目前程序不能使用 CUDA，**不等於證明 GPU 硬體壞掉**。GPU 2 inventory 異常的根因仍未知，GPU 7 的既存程序未受干擾。需管理員檢查 NVIDIA/Xid logs 與 driver/device/access 狀態。本次只作使用者指定診斷，未恢復已 budget-limited 的實驗 goal，完成數仍 0/14。

## 最新進度 — round 2（2026-09-08 09:49 UTC）

依續行要求先 `source ~/.venv/bin/activate`、`nvidia-smi`，閱讀既有阻塞紀錄後，**僅複驗一次 GPU 0 的 CUDA/BF16 路徑**。未重做 checkpoint/data 盤點，未逐卡重試。

- GPU 0 inventory：RTX 3090、24124 MiB free、0% utilization；但 `cuInit(0) = 999`，PyTorch CUDA unavailable、device count 0。`get_device_name(0)` 再次拋出相同 RuntimeError，BF16 matmul 尚未執行，probe exit **1**。
- 新原始證據：[`phase1a_cuda_recheck_20260908T094923Z.log`](phase1a_cuda_recheck_20260908T094923Z.log)。舊 log 保留未覆寫。
- 本輪實際檔案核對：`scripts/` 仍只有資源預檢；`phase1a.csv`、`phase1a_run.json`、`phase1a_report.md` 均不存在。下方逐項完成稽核仍適用：**0 / 14 條件、0 必要控制，runner 未實作，沒有 NLL/KL/CI**。
- CUDA 阻塞未解除；依規格停止，不改用 CPU/替代模型、不縮小正式樣本、不修改 driver/venv、不開始其他 phase。Goal **未完成**。

### 本輪確切複驗命令

cwd `/nfs/home/s314511048/pcb`；以下為本輪 shell command，timestamp 會產生新 log：

```bash
source ~/.venv/bin/activate && set -o pipefail
log="results/phase1a_cuda_recheck_$(date -u +%Y%m%dT%H%M%SZ).log"
(
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
  printf 'UTC: '; date -u -Is
  printf 'Host: '; hostname
  printf 'CWD: '; pwd
  nvidia-smi -i 0 --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.free,utilization.gpu,driver_version --format=csv
  python -u -c 'import ctypes, os, sys, torch
print("Python:", sys.executable, sys.version)
for key in ("VIRTUAL_ENV", "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
    print(key + ":", os.environ.get(key))
print("torch:", torch.__version__, "CUDA build:", torch.version.cuda)
cuda = ctypes.CDLL("libcuda.so.1")
print("cuInit(0):", cuda.cuInit(0))
print("CUDA available:", torch.cuda.is_available(), "device count:", torch.cuda.device_count())
print("Device:", torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "BF16 required"
x = torch.ones((2, 2), device="cuda:0", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert y.dtype == torch.bfloat16 and torch.equal(y, torch.full_like(y, 2))
print("BF16 CUDA matmul: PASS")'
  status=$?
  printf 'CUDA recheck exit: %s\n' "$status"
  exit "$status"
) 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
printf '\nSaved: %s\n' "$log"
exit "$status"
```

請使用 **`/goal pause`**，提供可通過 BF16 CUDA 預檢的 GPU 節點，或由管理員修復 `basic-2` 後通知繼續。沒有資源狀態改變的資訊前，不應再次執行相同 probe；本輪未代替使用者 pause、未標記完成。

## 上一輪進度（保留歷史）

2026-09-08，round 1：檢查既有專案（只有規格、README、AGENTS，沒有既有實驗可續跑）→ 檢查本地資源 → 保存 CUDA 初始化失敗證據 → 依規格停止。本輪沒有擴大研究範圍。

**完成條件數：0 / 14。必要模型控制：0。未產生任何 NLL、KL、PPL 或 CI；不能作 WQ/WK/WV 敏感度結論。實驗 runner 尚未實作。**

## 阻塞與原始證據

- Host `basic-2`，GPU 0 為 RTX 3090、24124 MiB 空閒，但無法實際使用 CUDA。
- `libcuda.so.1: cuInit(0)` 回傳 **999 / CUDA_ERROR_UNKNOWN**。
- PyTorch 2.5.1+cu121：CUDA unavailable、device count 0，存取裝置拋 RuntimeError。
- `nvidia-smi` 另回報 GPU 2 handle Unknown Error；不能因此直接認定 GPU 2 是根因。即使 `nvidia-smi` exit 0，也不表示 CUDA 可以運算。
- 本輪資源預檢 exit **1**，發生在模型載入之前，不是 OOM。

可重跑的**資源預檢**：[`../scripts/phase1a_preflight.sh`](../scripts/phase1a_preflight.sh)。這不是尚未完成的敏感度實驗 runner。

原始 command（cwd `/nfs/home/s314511048/pcb`）：

```bash
mkdir -p results
set -o pipefail
bash scripts/phase1a_preflight.sh 2>&1 | tee results/phase1a_preflight.log
```

- 完整 stdout/stderr、硬體、版本、環境與 traceback：[`phase1a_preflight.log`](phase1a_preflight.log)。
- 分層診斷、假設、恢復後驗證命令：[`../doc/debug-report.md`](../doc/debug-report.md)。
- `bash -n` 與內嵌 Python 語法檢查通過；預檢失敗回傳非零已實測。成功 GPU 路徑未測，不以預檢替代正式驗證。

## 已確認的本地資產（尚未固定正式 run 設定）

| 項目 | 實際檢查 | 限制 |
|---|---|---|
| 模型 | `Qwen/Qwen3-4B`，cache revision `1cfa9a7208912126459214e8b04321603b3df60c` | 三個 safetensors shard headers 可讀且 key 集合符合 index；未載入完整權重運算 |
| 模型 config | BF16；36 layers；Q 32 heads / KV 8 heads；head_dim 128 | 僅原始 config，不是 runtime 設定；正式執行仍须覆寫 use_cache=False |
| tokenizer | 同 snapshot，tokenizer.json/config、vocab、merges 存在 | 未實際 tokenize，未保存最終 tokenizer 設定 |
| 資料 | `Salesforce/wikitext`，`wikitext-2-raw-v1/test`，cache revision `b08601e04326c79dfdd32d625aee71d232d685c3` | test parquet 可讀：4358 rows、text 無 null；未檢查完整 token/block 數 |
| 環境 | Python 3.12.3；torch 2.5.1+cu121；transformers 5.16.1；datasets 2.17.1；其餘見 log | 入口 imports 通過；Qwen3 完整 forward 尚未驗證 |

路徑根：`/nfs/home/s314511048/.cache/huggingface/hub/`。預檢 offline、未下載替代模型或資料。正式環境/revisions/backend 仍必須在正式跑之前固定；不能把這份環境盤點冒充完整 run manifest。

## 對規格的完成稽核 / 下一輪缺項

目標是依 [`../phase1a.md`](../phase1a.md) 交付**可重跑實驗 runner 和固定設定、同一批 64 blocks 的 14 個真實條件、必要控制、逐 block 原始數據、相對 BF16 比較與探索性 CI**。下表核對實際檔案與本輪命令輸出；沒有用語法通過或資源清單當作實驗完成。

| 明確要求 | 實際證據或應交付位置 | 本輪狀態 |
|---|---|---|
| 保留既有檔案，先確認 GPU/模型/資料/版本/硬體 | `phase1a_preflight.log`、新增預檢 script；未修改既有三個檔案 | 盤點完成；CUDA 阻塞，模型與資料只部分验证 |
| 原始 Qwen/Qwen3-4B、BF16 baseline，固定模型/tokenizer/data revisions、套件和硬體 | 上表與 log 只有候選 revisions/版本 | 正式 run 尚未鎖定 |
| Transformers eval、batch=1、固定 attention backend、純文字 teacher forcing、use_cache=False，非聊天/thinking | 尚無 runner 或實測 log | 未實作/未驗證 |
| 一次所有層只改一種 projection；activation/KV/其他權重不變；每條件從原始權重開始、不疊加 | 未有 intervention 程式及控制結果 | 未實作/未驗證 |
| 七種模型：BF16、WQ8、WQ4、WK8、WK4、WV8、WV4；兩 context 共14條件 | `results/phase1a.csv` 尚不存在 | 0 / 14 |
| 相同 symmetric RTN；每 output row input 維分組128、不跨row；FP32 scale；qmax=2**(b-1)-1；maxabs/qmax；零group保零；round/clamp/dequant 後 BF16 | 尚無 quantizer 或測試 | 未實作/未驗證 |
| 保存修改 module 名稱、shape、參數量、相對權重誤差；承認 GQA 32:8 的非等參數比較 | config 證實32:8；無實際修改 modules | 模型結構盤點完成；介入紀錄未完成 |
| WikiText指定test原始順序，以兩個換行接text；不加特殊token；保存資料revision、文字hash和tokenizer設定 | 僅讀取指定 parquet；無文字hash/tokenizer實際設定 | 未完成 |
| 非重疊2049-token blocks、丟尾、seed42取64；保存索引/token IDs；不足64則阻塞 | 未產生抽樣產物 | 未完成；不得偷偷減少樣本 |
| 同blocks前513/2049 tokens，shifted labels計分512/2048位置；14條件同輸入及位置 | 無模型forward或對齊控制 | 未完成 |
| 每block平均NLL、ΔNLL vs BF16、KL(P_BF16\|\|P_test)；FP32 softmax/log-softmax/reduction；按位置分塊、不囤完整logits | `results/phase1a.csv` 不存在 | 未實作/未驗證 |
| PPL = exp(所有計分token平均NLL)，不得平均block PPL | 無 metric 程式或數值 | 未實作/未驗證 |
| 先小樣本BF16/WQ4；核對module、非目標不變、移除介入回baseline、logits/metrics有限且計分對齊，再正式跑 | 無 smoke/control 結果 | 0，尚未開始 |
| 各bit/context分開Q/K/V paired差；同block索引2000次paired bootstrap、seed42、探索性95%CI | `results/phase1a_report.md` 不存在 | 未實作/未驗證 |
| 不把deterministic重跑當樣本；CI跨零不證等價；接受無明顯差異 | 目前沒有任何差異證據 | 待真實數據後遵守，不能提前下結論 |
| 最小可重跑的完整程式與設定 | 目前只有可重跑資源預檢 | 敏感度runner/設定未交付 |
| `results/phase1a.csv` 逐block數值 | 檔案不存在；完整預期7×2×64=896筆 | 未交付，沒有空表或假數據充數 |
| `results/phase1a_run.json` 完整設定與執行命令 | 檔案不存在；目前只有預檢命令/log | 未交付，不能以預檢manifest取代 |
| `results/phase1a_report.md` 數據、CI、結論與限制 | 檔案不存在 | 未交付，不能以阻塞報告取代 |
| 結論限於此模型/資料/量化/contexts；非packed GEMM，不宣稱加速/真實壓縮、不推論KV lifetime/serving | 本輪未做模型實驗、未作上述推論 | 範圍維持；未開始其他phase |
| 資源不足保存命令/錯誤/完成與確切缺項至 `results/phase1a_blocked.md`；回報未完成並要求pause | 本文件、preflight log、debug report | 本輪阻塞記錄完成，goal仍未完成 |

## 停止與需要的輸入

請使用 **`/goal pause`**，並提供可通過 BF16 CUDA 預檢的 GPU 節點，或請管理員修復 `basic-2` 的 CUDA 初始化問題後通知繼續。未替使用者 pause，也未標記 goal complete。

下一輪先讀本文件，只在資源狀態有變後以 timestamped log 重跑預檢；不要重複全卡重試或修改共享 driver/venv。通過後的下一個必要步驟：固定正式設定、實作資料/RTN/metrics/控制及小樣本 BF16/WQ4，驗證後再跑14條件並完成交付與稽核。
