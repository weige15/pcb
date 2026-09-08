# Qwen3 Attention Projection Quantization Sensitivity

比較 Qwen3-4B 的 **WQ／WK／WV 權重單獨做 4/8-bit 量化**，對 BF16 baseline 的 NLL、KL 與 PPL 影響。固定 WikiText-2 raw test、64 blocks、512/2048 預測位置，共14條件；不是 low-bit 加速實驗。

## 執行與測試

需要既有 `~/.venv`、可用 BF16 CUDA GPU，以及指定 revisions 的本地模型/資料。先確認空卡，再設定 `CUDA_VISIBLE_DEVICES`：

```bash
source ~/.venv/bin/activate
nvidia-smi
CUDA_VISIBLE_DEVICES=2 bash scripts/run_projection_quantization_sensitivity.sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
```

入口先跑 BF16/WQ4 smoke。新執行預設寫入 `results/projection_quantization_sensitivity/`，已有結果會拒絕覆寫；可指定新的 output 目錄。只跑 smoke 時加 `--smoke-only`。

## 名稱與用途

| 檔案 | 用途 |
|---|---|
| [`projection_quantization_sensitivity.md`](projection_quantization_sensitivity.md) | 實驗規格 |
| `scripts/projection_quantization_sensitivity.py` | 權重量化、介入控制、NLL/KL 計分與統計 |
| `scripts/projection_quantization_sensitivity_config.json` | 固定 revisions、條件、抽樣與數值設定 |
| `scripts/run_projection_quantization_sensitivity.sh` | 實驗執行入口 |
| `scripts/verify_projection_quantization_sensitivity.py` | 數據、權重雜湊與 paired CI 核驗 |
| `scripts/check_projection_quantization_resources.sh` | GPU／模型／資料資源預檢 |

新輸出用內容命名：`block_metrics.csv`、`run_manifest.json`、`sensitivity_report.md`、`sensitivity_analysis.json`、`sampled_token_blocks.npz`。

## 已完成實驗與文件

2026-09-08 的14條件、896筆原始數據已完成；本次只調整命名，**沒有重跑 GPU 實驗或改寫歷史結果**。

- [已完成實驗：報告](archive/qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a_report.md) · [逐block數據](archive/qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a.csv) · [完成稽核](archive/qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a_audit.md)
- [封存說明](archive/README.md)：舊階段代號只留在歷史bundle，原始命令、來源、檔名與SHA256不作事後改寫。
- [Runbook](doc/runbook.md)：執行、測試、驗證與故障恢復。
- [Onboarding](doc/onboarding.md)：結構、命名對照及安全修改方式。
- [歷史診斷入口](doc/debug-report.md)
