# Debug Report

## 歷史診斷位置

完整的原始診斷已隨2026-09-08實驗封存：

[Qwen3 projection quantization sensitivity — 原始Debug Report](../archive/qwen3_projection_quantization_sensitivity_2026-09-08/doc/debug-report.md)

保留的內容包括basic-2 CUDA error999、snapshot非必需檔案檢查，以及NumPy/PyTorch CUDA FP32 rounding差異。原命令、traceback、來源hash和當時結論均未改寫。

## 本次命名重構

- 新入口為 `scripts/run_projection_quantization_sensitivity.sh`，新output檔名見 [runbook](runbook.md)。
- 原始bundle的51個檔案與source hashes皆保持不變。
- 12個tests通過；新分析入口以896筆原始測量的暫存副本重算，analysis JSON和bootstrap draws逐byte一致。
- 只做命名、路徑與文件調整；沒有重新執行GPU實驗或新增模型結果。

驗證log：[`results/naming_refactor_tests.log`](../results/naming_refactor_tests.log)。
