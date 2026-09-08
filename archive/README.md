# Historical experiments

## Qwen3 attention projection quantization sensitivity — 2026-09-08

Bundle：[qwen3_projection_quantization_sensitivity_2026-09-08/](qwen3_projection_quantization_sensitivity_2026-09-08/README.md)

- [結果與CI](qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a_report.md)
- [896筆逐block原始數據](qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a.csv)
- [完整設定、來源hash與控制紀錄](qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a_run.json)
- [完成稽核](qwen3_projection_quantization_sensitivity_2026-09-08/results/phase1a_audit.md)
- [封存checksum清單](qwen3_projection_quantization_sensitivity_2026-09-08/snapshot_checksums.json)

原識別碼為 `phase1a`。這些名稱只留在封存，避免事後改寫原始實驗紀錄；bundle包含當時的程式、規格、docs、tokenizer、數據、logs及smoke。51個原始檔案逐byte保存，未因命名重構而更改recorded commands或hash。

原manifest中的cwd/絕對命令代表**當時**的實際執行位置；現在bundle已搬移。歷史核驗使用bundle內原verifier，需在暫存output副本操作；新實驗使用repo根目錄的描述性入口。詳見 [目前runbook](../doc/runbook.md#evaluate)。
