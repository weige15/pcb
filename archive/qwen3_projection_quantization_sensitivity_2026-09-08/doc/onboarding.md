# Onboarding

## What This Project Does

本輪唯一問題：Qwen3-4B 各層的 Q/K/V projection weights 單獨降低精度，會如何影響純文字 teacher-forced 預測分布與 NLL？接受差異不明顯、排序隨條件變動。規格見 [`../phase1a.md`](../phase1a.md)，不延伸到activation/KV或下一phase。

## Quickstart

從repo root：

```bash
source ~/.venv/bin/activate
nvidia-smi
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=2 bash scripts/run_phase1a.sh results/smoke_$(date -u +%Y%m%dT%H%M%SZ) --smoke-only
```

先確認GPU2在當前節點可用，否則換空卡index/UUID。需要指定模型和資料cache；版本、路徑與正式重跑命令見 [runbook](runbook.md)。單元測試預期10 tests OK；smoke為0/14正式條件，不能宣稱完成。

## Important Files

| 路徑 | 用途 |
|---|---|
| `phase1a.md` | 唯一實驗規格；原文保留 |
| `AGENTS.md` | 必須先venv+nvidia-smi、最小變更與驗證規則 |
| `scripts/run_phase1a.sh` | 單GPU/offline/記錄錯誤入口 |
| `scripts/phase1a_config.json` | revisions、14條件、固定seed/chunk/backend |
| `scripts/phase1a.py` | 真實資料/權重/forward/metric/controls/分析 |
| `scripts/verify_phase1a.py` | CPU重tokenize、原始權重hash、NumPy RTN與CI核验 |
| `tests/test_phase1a.py` | 暫存synthetic測試，不是實測 |
| `results/phase1a.csv` | 7×2×64=896筆原始block measurements |
| `results/phase1a_run.json` | 精確設定、命令、hash、硬體、module與控制紀錄 |
| `results/phase1a_report.md` | BF16比較、paired CI、有限結論 |
| `results/phase1a_audit.md` | 規格到實際證據的完成稽核 |

## Architecture Map

Pinned official test parquet → 原始row順序以兩個換行join → 原tokenizer無特殊token → 非重疊2049-token blocks → seed42取64並保存token IDs。

Pinned原始BF16 model → eval / no-cache / 固定SDPA Flash kernel → smoke BF16/WQ4 / hash / 還原控制 → BF16及6種RTN介入 → 相同blocks的513/2049-token forward。

`model.model` 最終hidden states → 未修改 `lm_head` 每64位置輸出logits → FP32 NLL、KL(BF16 || test) → 逐block CSV → PPL與paired bootstrap CI。baseline hidden暫存在CPU，不保存完整logits或KV cache。每種介入finally還原並檢查全model state hash。

外部依賴僅既有PyTorch、Transformers、HF hub、NumPy、safetensors、pyarrow；不需要API或服務。

## Development Workflow

1. 先讀最新blocker、實際manifest/log和audit；不要把歷史0/14當目前狀態，也不要只信`measured`欄位。
2. 只修本輪最近失敗；不修改實驗條件追求差異或新排序。程式修改後重新核對spec和tests。
3. 已完成run的程式/config/hash不可悄悄更換；需要新版本實验時使用新output並保留舊artifact。
4. 先unit+smoke再正式14條件。report需896筆及全部必要控制；deterministic重跑不是新增獨立樣本。
5. 修改命令/檔案/錯誤模式時更新README/runbook。本輪完成後不自動開始下一phase。

## Testing

- CPU：上述`unittest`。覆蓋量化、分組、零/tie、資料不足、計分方向/對齊、bootstrap、PPL、CSV篡改及失敗gate。
- GPU smoke：一個已選block、兩個contexts、BF16/WQ4；介入模組正確、其他state不變、還原後logits路徑與NLL/KL回baseline。
- 正式：14×64筆同輸入計分，finite檢查與每條件state還原。只有全部完成才可報告。
- Artifact audit：`python scripts/verify_phase1a.py --output results`，需先venv+nvidia-smi；這個命令不跑模型，不是完整GPU重現的替代品。
- 無既有build/lint/type-check pipeline，不宣稱其通過。`bash -n`可驗證launcher語法。

## Troubleshooting

詳細錯誤表與恢復命令見 [runbook](runbook.md#common-failures)。舊basic-2 CUDA error999根因未確定；本輪basic-1可用不證明舊卡修好。早期snapshot缺`.gitattributes`問題已改成只檢查必要資產。tokenizer長串warning不代表forward超長，實際只輸入513/2049 tokens。

缺GPU/checkpoint/data、OOM時保存`phase1a_blocked.md`，回報未完成並要求使用者`/goal pause`；不reset GPU、改共享venv、換小模型、減正式樣本或反覆重試。新output防止覆寫歷史。

## Documentation Freshness Checklist

- [ ] README quickstart still works.
- [ ] Run commands match the current code.
- [ ] Test commands match the current code.
- [ ] Important files list is still accurate.
- [ ] Architecture map matches the current implementation.
- [ ] Troubleshooting section includes recent known failures.
