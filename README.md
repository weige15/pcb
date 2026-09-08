# Qwen3 Attention Projection Quantization Sensitivity

研究 Qwen3-4B 的 WQ／WK／WV **權重表示誤差**對 BF16 的 NLL、KL 與 PPL 影響，不是 low-bit 加速或 scheduler 實驗。

## 逐層實驗（本輪）

固定歷史 checkpoint、64 blocks、每 block 2048 個預測位置、group128 RTN4。每次只量化**一層的一個 projection weight**，其餘仍原始 BF16，共108介入＋BF16。

- [固定協定](results/layerwise_projection_sensitivity/protocol.md)
- [報告及逐層圖表](results/layerwise_projection_sensitivity/sensitivity_report.md)
- [逐block NLL/KL](results/layerwise_projection_sensitivity/block_metrics.csv) · [配對比較](results/layerwise_projection_sensitivity/paired_comparisons.csv)
- [完成稽核](results/layerwise_projection_sensitivity/completion_audit.md)（只在全部實測及驗證完成後判定）

需要既有 `~/.venv`、固定 revision 本地 checkpoint/data，以及至少19GiB空閒的 BF16 CUDA GPU。從 repo root 執行，先確認空卡：

```bash
source ~/.venv/bin/activate
nvidia-smi
# 本輪目錄已存在：僅補缺少的條件；全部完成時不再forward。
CUDA_VISIBLE_DEVICES=2 bash scripts/run_layerwise_projection_sensitivity.sh \
  results/layerwise_projection_sensitivity --resume
# 重新執行全部條件，必須使用新目錄：
CUDA_VISIBLE_DEVICES=2 bash scripts/run_layerwise_projection_sensitivity.sh \
  results/layerwise_projection_sensitivity/rerun_$(date -u +%Y%m%dT%H%M%SZ)
```

設定在 `scripts/layerwise_projection_sensitivity_config.json`；每條件原子提交64筆數據及全state/輸出還原控制。`--max-conditions 1` 可先完成BF16及第一個完整介入再resume，不減少總樣本或總條件。

## 測試、重新分析與獨立驗證

先啟用環境並檢查GPU，再執行（下列均不跑model forward）：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
python scripts/analyze_layerwise_projection_sensitivity.py --output results/layerwise_projection_sensitivity
python scripts/verify_layerwise_projection_sensitivity.py --output results/layerwise_projection_sensitivity
```

未完成109條件或控制失敗時，分析器拒絕產出完整報告。不要只以manifest、CSV列數或tests綠燈認定完成。

## 歷史全層實驗（保留不重跑）

[原規格](projection_quantization_sensitivity.md)：一次量化所有層的一種projection、4/8-bit、512/2048位置，共14條件、896筆。

歷史bundle已由commit `d6f629c` 搬到 `results/`，原檔名、命令與SHA256不作事後改寫：
[報告](results/phase1a_report.md) · [數據](results/phase1a.csv) · [歷史稽核](results/phase1a_audit.md)。舊source名稱只是歷史provenance，不冒充新版程式產生。

全層實驗入口仍為 `scripts/run_projection_quantization_sensitivity.sh`；它不支援續跑，重新執行需指定新output。本輪沿用其量化與計分函式，不重跑舊14條件。

[Runbook](doc/runbook.md)：環境、命令、恢復與限制。 [Onboarding](doc/onboarding.md)：程式結構與安全修改。缺資源時保留 `blocked.md`，回報未完成並提供所缺GPU/原始資產，不縮樣本或換模型。

## Hardware-real static KV-cache screen (new independent scope)

`results/kv_cache_precision_pareto/` is a separate Qwen3-4B experiment. It keeps weights/compute BF16 and tests only locally supported direct-read FlashInfer BF16/FP8 KV payloads under fixed homogeneous continuous batches. The protocol freezes 2048/8192 prompts, 256 output tokens, concurrency 1/4/8/16/32, seeds, warmups/repeats and provisional quality/SLO gates before model forward. It does not modify or reinterpret `main@1eef360` or the earlier result directories.

After checking an idle GPU:

```bash
source ~/.venv/bin/activate
nvidia-smi
CUDA_VISIBLE_DEVICES=<idle-GPU> bash scripts/run_kv_cache_precision_pareto.sh --phase all
CUDA_VISIBLE_DEVICES='' python scripts/analyze_kv_cache_precision_pareto.py --output results/kv_cache_precision_pareto
CUDA_VISIBLE_DEVICES='' python scripts/independent_verifier.py --output results/kv_cache_precision_pareto
```

The runner records raw per-request/repeat metrics, timestamps, cache bytes, direct-kernel provenance, errors/OOM and the honest local 4-bit capability result. If direct compressed attention is unavailable, it writes `blocked.md` and must not use a BF16 decompression fallback.
