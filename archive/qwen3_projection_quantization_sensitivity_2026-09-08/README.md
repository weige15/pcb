# pcb

## Phase 1a：Qwen3-4B projection-weight sensitivity

對所有層單獨做 WQ/WK/WV 的 4/8-bit grouped RTN fake quantization，與 BF16 比較 NLL/KL。固定 WikiText-2 raw test、64 blocks、512/2048 預測位置，共14條件；**不是 low-bit 加速實驗**。

### 執行與測試

需要既有 `~/.venv`、可用 BF16 CUDA GPU，以及已快取的指定 revisions。先檢查空卡，將下方 GPU index 換成可用卡：

```bash
source ~/.venv/bin/activate
nvidia-smi
CUDA_VISIBLE_DEVICES=2 bash scripts/run_phase1a.sh results/rerun_$(date -u +%Y%m%dT%H%M%SZ)
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
```

入口會先跑 BF16/WQ4 smoke；舊 output 不覆寫。只跑 smoke：同命令加 `--smoke-only`。重算統計：`python scripts/phase1a.py --analyze-only --output results`。

### 成果與文件

- [實驗規格](phase1a.md)；[原始逐 block 數據](results/phase1a.csv)、[設定與控制紀錄](results/phase1a_run.json)、[結果與 CI](results/phase1a_report.md)。
- [完成稽核](results/phase1a_audit.md)：逐項核對真實證據，不以測試綠燈代替14條件。
- [Runbook](doc/runbook.md)：revisions、離線快取、命令、驗證與故障恢復。
- [Onboarding](doc/onboarding.md)：程式結構與安全續作。
- [阻塞歷史](results/phase1a_blocked.md)、[診斷](doc/debug-report.md)：舊basic-2 CUDA錯誤與本輪basic-1恢復紀錄。

主要程式與固定設定在 `scripts/`，synthetic unit tests 在 `tests/`，真實原始/衍生輸出在 `results/`。缺GPU、資料、checkpoint或OOM時保留證據並回報未完成，不改模型或減少樣本。
