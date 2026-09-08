# Projection quantization sensitivity outputs

新實驗預設輸出至 `results/projection_quantization_sensitivity/`：

- `block_metrics.csv`：每block的NLL、相對BF16的ΔNLL與KL。
- `run_manifest.json`：固定設定、命令、環境、來源與必要控制。
- `sensitivity_report.md` / `sensitivity_analysis.json`：摘要、paired CI與結論。
- `sampled_token_blocks.npz` / `paired_bootstrap_indices.npy`：可重現輸入與重抽索引。

目前未因命名重構重新跑GPU實驗，沒有冒充由新版程式生成的結果。

2026-09-08已完成的14條件/896筆真實結果連同原程式移至 [具描述性名稱的歷史bundle](../archive/README.md)，不是刪除數據。`naming_refactor_tests.log` 只記錄本次命名回歸測試。
