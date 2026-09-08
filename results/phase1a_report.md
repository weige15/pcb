# Phase 1a — Qwen3-4B projection-weight sensitivity

## 結果與方法

14/14 條件、896 筆逐 block 實測；每條件 64 個 blocks。單位：natural-log nats/token。
BF16 原始 Qwen/Qwen3-4B；WikiText-2 raw test；同一批 seed 42 非重疊 2049-token blocks，分別取前 513/2049 tokens。
Transformers eval、batch=1、SDPA（固定 PyTorch FLASH_ATTENTION kernel）、use_cache=False；純文字 teacher forcing。
所有層一次只改 Q/K/V 中一種 weight；每 output row 以 input 維分組 128、FP32 scale、symmetric RTN、dequant 回 BF16。
NLL/KL 的 log-softmax 與 reduction 為 FP32；保存 baseline 最終 BF16 hidden states 於 CPU，透過未修改的 lm_head 按 64 位置重建兩組 logits 計分，不囤完整 logits。
PPL = exp(所有計分 token 的平均 NLL)；各 block 計分長度一致，故等同 block NLL 均值，不是 block PPL 均值。

| Context（預測數） | 條件 | NLL | PPL | ΔNLL vs BF16 | KL(BF16 || test) |
|---:|---|---:|---:|---:|---:|
| 512 | BF16 | 2.91952682 | 18.532516 | +0.00000000 | 0.00000000 |
| 512 | WQ8 | 2.91804361 | 18.505049 | -0.00148305 | 0.00115036 |
| 512 | WQ4 | 2.94791365 | 19.066134 | +0.02838700 | 0.01867247 |
| 512 | WK8 | 2.92028666 | 18.546603 | +0.00075956 | 0.00121304 |
| 512 | WK4 | 2.93364668 | 18.796049 | +0.01412003 | 0.01989491 |
| 512 | WV8 | 2.91950393 | 18.532092 | -0.00002282 | 0.00123649 |
| 512 | WV4 | 2.96441770 | 19.383413 | +0.04489078 | 0.03152605 |
| 2048 | BF16 | 2.62236190 | 13.768204 | +0.00000000 | 0.00000000 |
| 2048 | WQ8 | 2.62154007 | 13.756894 | -0.00082181 | 0.00108853 |
| 2048 | WQ4 | 2.65759993 | 14.262018 | +0.03523798 | 0.01835794 |
| 2048 | WK8 | 2.62346339 | 13.783378 | +0.00110126 | 0.00115997 |
| 2048 | WK4 | 2.62835002 | 13.850897 | +0.00598801 | 0.01877187 |
| 2048 | WV8 | 2.62266064 | 13.772318 | +0.00029877 | 0.00122907 |
| 2048 | WV4 | 2.66645432 | 14.388860 | +0.04409224 | 0.02920421 |

## Q/K/V paired 比較（探索性 95% CI）

相同 block 索引配對，2000 次 bootstrap、seed 42；所有比較共用保存的 draws，percentile CI。
差值方向為左項減右項；ΔNLL 差等於 NLL 差。未做多重比較校正，不能作確認性顯著性宣稱。

| Context | bit | 差值 | metric | mean | 95% CI |
|---:|---:|---|---|---:|---|
| 512 | 8 | WQ - WK | delta_nll | -0.00224262 | [-0.00332688, -0.00113524] |
| 512 | 8 | WQ - WK | kl | -0.00006268 | [-0.00015351, +0.00001820] |
| 512 | 8 | WQ - WV | delta_nll | -0.00146023 | [-0.00257368, -0.00037138] |
| 512 | 8 | WQ - WV | kl | -0.00008613 | [-0.00014185, -0.00002225] |
| 512 | 8 | WK - WV | delta_nll | +0.00078239 | [-0.00038551, +0.00191689] |
| 512 | 8 | WK - WV | kl | -0.00002345 | [-0.00009849, +0.00008646] |
| 512 | 4 | WQ - WK | delta_nll | +0.01426697 | [+0.00715816, +0.02230033] |
| 512 | 4 | WQ - WK | kl | -0.00122244 | [-0.00237278, +0.00007241] |
| 512 | 4 | WQ - WV | delta_nll | -0.01650378 | [-0.02513463, -0.00787669] |
| 512 | 4 | WQ - WV | kl | -0.01285358 | [-0.01458132, -0.01103040] |
| 512 | 4 | WK - WV | delta_nll | -0.03077075 | [-0.04017087, -0.02193616] |
| 512 | 4 | WK - WV | kl | -0.01163115 | [-0.01311901, -0.01035458] |
| 2048 | 8 | WQ - WK | delta_nll | -0.00192307 | [-0.00253363, -0.00135723] |
| 2048 | 8 | WQ - WK | kl | -0.00007144 | [-0.00014288, -0.00002393] |
| 2048 | 8 | WQ - WV | delta_nll | -0.00112057 | [-0.00172690, -0.00050116] |
| 2048 | 8 | WQ - WV | kl | -0.00014054 | [-0.00021184, -0.00008744] |
| 2048 | 8 | WK - WV | delta_nll | +0.00080250 | [+0.00025318, +0.00134574] |
| 2048 | 8 | WK - WV | kl | -0.00006909 | [-0.00014133, -0.00001936] |
| 2048 | 4 | WQ - WK | delta_nll | +0.02924997 | [+0.02486089, +0.03372595] |
| 2048 | 4 | WQ - WK | kl | -0.00041393 | [-0.00108634, +0.00028508] |
| 2048 | 4 | WQ - WV | delta_nll | -0.00885426 | [-0.01402111, -0.00395897] |
| 2048 | 4 | WQ - WV | kl | -0.01084627 | [-0.01168300, -0.00998326] |
| 2048 | 4 | WK - WV | delta_nll | -0.03810423 | [-0.04346553, -0.03293272] |
| 2048 | 4 | WK - WV | kl | -0.01043234 | [-0.01120255, -0.00969766] |

## 描述性結論

- context=512, 8-bit，delta_nll 由大到小：WK8 (0.00075956) > WV8 (-0.00002282) > WQ8 (-0.00148305)。這只是本次均值排序，請合併 paired CI 判讀。
- context=512, 8-bit，kl 由大到小：WV8 (0.00123649) > WK8 (0.00121304) > WQ8 (0.00115036)。這只是本次均值排序，請合併 paired CI 判讀。
- context=512, 4-bit，delta_nll 由大到小：WV4 (0.04489078) > WQ4 (0.02838700) > WK4 (0.01412003)。這只是本次均值排序，請合併 paired CI 判讀。
- context=512, 4-bit，kl 由大到小：WV4 (0.03152605) > WK4 (0.01989491) > WQ4 (0.01867247)。這只是本次均值排序，請合併 paired CI 判讀。
- context=2048, 8-bit，delta_nll 由大到小：WK8 (0.00110126) > WV8 (0.00029877) > WQ8 (-0.00082181)。這只是本次均值排序，請合併 paired CI 判讀。
- context=2048, 8-bit，kl 由大到小：WV8 (0.00122907) > WK8 (0.00115997) > WQ8 (0.00108853)。這只是本次均值排序，請合併 paired CI 判讀。
- context=2048, 4-bit，delta_nll 由大到小：WV4 (0.04409224) > WQ4 (0.03523798) > WK4 (0.00598801)。這只是本次均值排序，請合併 paired CI 判讀。
- context=2048, 4-bit，kl 由大到小：WV4 (0.02920421) > WK4 (0.01877187) > WQ4 (0.01835794)。這只是本次均值排序，請合併 paired CI 判讀。
- 24 個探索性 paired CI 中 5 個跨零；跨零不是已證明等價。接受沒有明顯差異，不挑選或加跑條件追求排序。

## 控制與可重跑證據

- 正式跑前以一個已選 block 的兩個 contexts 跑 BF16/WQ4；控制結果保存在 `phase1a_run.json:smoke`。不把控制或 deterministic 重跑當獨立樣本。
- 第一/最後 64 個 scoring positions 的 chunked lm_head logits 與 CausalLM API bitwise 一致；shifted NLL 與直接 cross entropy 核對。
- 每次介入核對所有 unique parameters 與 buffers 的 SHA256，只改 36 個目標 weights；評估前後不變，還原後全模型 state hash 回 baseline。
- 小樣本還原後 hidden states bitwise 一致、NLL 相同、KL=0；正式末態 hash 回原始 checkpoint。所有逐 block logits/metrics 檢查有限。
- 詳細 module 名稱、shape、參數量、相對 L2 誤差在 `phase1a_run.json:interventions`；Q 每條件 377,487,360 個參數，K/V 各 94,371,840。
- 固定設定與實際命令：`phase1a_run.json`；原始逐 block：`phase1a.csv`；抽樣索引/token IDs：`phase1a_tokens.npz`；tokenizer 設定：`tokenizer/`。
- 統計可獨立重算：`python scripts/phase1a.py --analyze-only --output results`；bootstrap draws：`phase1a_bootstrap_indices.npy`；未四捨五入摘要/CI：`phase1a_analysis.json`。

```bash
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b bash scripts/run_phase1a.sh results/rerun_$(date -u +%Y%m%dT%H%M%SZ) --config /nfs/home/s314511048/pcb/results/phase1a_config.json
```

## 限制

- 結論只限此 Qwen3-4B checkpoint、WikiText test 抽樣、RTN 規則與兩個 contexts。不是跨模型/資料/seed 的推論。
- Qwen3-4B 是 GQA（Q 32 heads、KV 8 heads）；Q 介入參數量為 K/V 的四倍，是原生端到端敏感度，不是等參數量/等成本比較。
- fake quantization 後仍 BF16 GEMM；沒有 packed low-bit GEMM，不宣稱加速或實際記憶體壓縮。
- activation、其他權重與運算設定未量化；沒有啟用 KV cache，不從本實驗推論 KV lifetime、prefill/decode 差別或 serving 效益。
- 相鄰原文 blocks 可能相關；block bootstrap 只描述此有限樣本下的探索性變異，不保證獨立性或人口層級涵蓋率；多重比較未校正。
- BF16 前向與 FP32 reduction 有數值誤差；小 KL 與均值排序應審慎解讀。統計運算使用保存的 FP32 measurements，不把 deterministic 重跑算獨立樣本。
- 未開始 activation/KV cache 或下一階段；不以任一排序作下一階段已成立的證據。
