# Phase 1a 完成稽核

## 目標與判定

將 `phase1a.md` 轉為以下實際交付：**可重跑的 Qwen/Qwen3-4B BF16 與 WQ/WK/WV-only 4/8-bit runner、固定設定及輸入；同64 blocks、512/2048位置的14個真實條件；必要介入/還原/計分控制；逐block NLL/ΔNLL/KL、正確PPL、paired bootstrap CI與有限結論。** 不要求必然有明顯差異，不得用寫好程式、完整manifest或測試綠燈代替實驗。

**稽核判定：本輪實驗交付 PASS，沒有本目標的必需缺項。** 原始規格與AGENTS未改；未開始後續phase。這是對檔案、執行輸出、獨立重新計算及控制證據的結論，不是僅引用`status=measured`。

實際正式命令：

```bash
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b bash scripts/run_phase1a.sh results
```

外層 `timeout --signal=TERM 3600`；`phase1a_formal.exit` 為0。basic-1／RTX3090 GPU2；2026-09-08 10:12:41–10:30:17 UTC，1055.90秒（含準備、控制及雜湊）。PyTorch峰值 allocated 7.810 GiB、reserved 7.930 GiB；未OOM。量化不是記憶體壓縮，這個峰值只是本次執行的資源紀錄。

## Prompt-to-artifact checklist

| 明確要求／完成條件 | 已檢查的實際證據 | 結果 |
|---|---|---|
| 依 `phase1a.md`，保留既有檔案、只完成本輪 | `git diff --exit-code -- AGENTS.md phase1a.md` 成功；README只增補入口；舊preflight/debug/smoke/log均保留 | PASS |
| 先確認GPU、模型、資料與版本/硬體 | `phase1a_preflight_20260908T095523Z.log`：basic-1 cuInit=0、BF16 matmul PASS、資產headers/test parquet可讀；不是將舊basic-2問題冒充修好 | PASS |
| Qwen/Qwen3-4B原始checkpoint、BF16 baseline；固定model/tokenizer/data revision | `phase1a_run.json:config/model_source/checkpoint_sha256/versions/cuda`；model/tokenizer=`1cfa9a…60c`，data=`b08601…85c3`（完整SHA在manifest）；独立audit讀全部shards比對398個原始parameter hash | PASS |
| 正式前固定設定與重跑命令 | `scripts/phase1a_config.json`、run內config、`phase1a_config.json`快照；run開始前source/config/spec SHA256已寫入；終態verifier核對未變；`rerun_command`保留快照與GPU，smoke模式亦保留 | PASS |
| Transformers、eval、batch1、固定backend、純文字teacher forcing、不產生thinking/chat | `scripts/phase1a.py:hidden_for/main`、`runtime_model_config/runtime_attention_backend/numerics`；SDPA僅允許PyTorch FLASH_ATTENTION kernel；無generate/chat template，真實forward通過 | PASS |
| use_cache=False，不區分prefill/decode，不推論KV lifetime/serving | 每次forward明示False並核對past_key_values=None；report限制明示無KV/serving推論 | PASS |
| 只改attention projection weights；一次所有層一種；activation/其他weights/buffers保留設定 | 每次介入實際全state SHA256差集正好36個目標weights；`interventions[*].changed_tensors/non_target_unchanged`；eval前後未變，沒有activation/KV量化 | PASS |
| 每條件從原始权重開始、禁止疊加 | `intervention`入口全state對原始hash；finally還原；7次介入紀錄（1 smoke＋6 formal）都restored/unchanged_during_eval=True，正式final_state_matches_checkpoint=True | PASS |
| 七模型條件×兩contexts=14 | CSV逐格核對：BF16/WQ8/WQ4/WK8/WK4/WV8/WV4 ×512/2048均64筆；無重複，共896筆與1,146,880個計分位置；正式log逐格64/64與exit0 | PASS |
| 相同symmetric RTN、row內group128、FP32 scale、qmax、零group、round/clamp/dequant BF16 | unit tests涵蓋zero/ties/row/shape；source實作符合公式；独立NumPy重建216個不同module/bit介入全部BF16 SHA256相同，252筆含smoke的module紀錄均核對。FP32 CUDA常數除法細節見下節 | PASS |
| 保存實際module名稱、shape、參數量、相對權重誤差 | `phase1a_run.json:interventions`；Q [4096,2560]×36、377487360參數；K/V [1024,2560]×36、94371840參數；原始/量化hash與relative L2均驗證 | PASS |
| GQA Q32/KV8；不是等參數/等成本比較 | 原config/runtime config確認32:8；report明示Q介入參數量4倍，僅比較原生端到端影響 | PASS |
| 指定WikiText-2 raw test，原始順序雙換行join、不加特殊token；revision、text hash、tokenizer設定 | 獨立重新讀4358 rows官方parquet、雙換行join、用保存tokenizer重tokenize；文字hash與299078-token stream hash完全一致；`data`與`tokenizer/`保存設定 | PASS |
| 非重疊2049-token blocks、丟尾、seed42抽64，保存索引/token IDs，不足64要阻塞 | 獨立核對145 full blocks、1973 tail tokens、64唯一抽樣索引；`phase1a_tokens.npz`每一token等於原始stream切片；資料不足unit test確認拒絕縮样本 | PASS |
| 同blocks前513/2049 tokens、shifted labels計512/2048位置；各條件同輸入與計分位置 | 896列input/labels SHA256重新按保存token IDs核對；首尾64位置chunk logits與CausalLM API bitwise一致；NLL vs直接cross entropy核對；每格token數32768/131072 | PASS |
| 每block NLL、ΔNLL vs BF16、KL(P_BF16\|\|P_test)；FP32 log-softmax/reduction；按位置分塊 | CSV三項896列全部有限；delta逐列重算，BF16 delta/KL全0；已讀metric/score源碼；unit test驗證KL方向而非反向、dense/chunk一致；實際只保留兩個64位置logits chunks，baseline final hidden存CPU | PASS |
| PPL由全部計分token平均NLL取exp，不能平均block PPL | 每格固定block長度；NumPy重新計算14個均值與exp與analysis相同；特製unit fixture會區分exp(mean NLL)和mean(exp NLL) | PASS |
| 先小樣本BF16/WQ4再全量；finite、介入/非目標/還原/對齊必要控制 | `smoke_20260908_fixed/`先成功；正式run又先跑控制再寫CSV；两個context還原hidden bitwise相同、NLL相同/KL0，独立smoke記錄與正式smoke及正式sample0行精確一致，不增加樣本數 | PASS |
| 各bit/context Q/K/V paired差、相同block索引2000次bootstrap seed42、95%探索性CI | `phase1a_bootstrap_indices.npy` shape[2000,64]完全等於seed42 draws；`phase1a_analysis.json`24個pair/metric CI；獨立對兩組以相同draws重抽後相減，均值與percentile界限一致 | PASS |
| 不將deterministic重跑當獨立樣本、CI跨零不證等價、接受無明顯差異 | 正式樣本仍64，smoke不併入；report列全部24個比較，包括5個跨零CI，明示探索性/未校正/跨零非等價；8-bit微小變化未觸發加跑 | PASS |
| `results/phase1a.csv`逐block原始數據 | 實際189841 bytes、896列＋header，三項metrics及配對識別齊全；不是空表、synthetic fixture或只有均值 | PASS |
| `results/phase1a_run.json`完整設定及執行命令 | 實際185278 bytes；revisions、套件/Python/host/GPU/driver、env、數值設定、原始資產與source hashes、抽樣、controls、module、timing、commands齊全 | PASS |
| `results/phase1a_report.md`數據、CI、結論與限制 | 已逐行閱讀14格摘要、24 CI、排序與全部限制；`--analyze-only`重算後report/analysis/CSV/draws四個hash未變 | PASS |
| 最小可重跑程式與設定、必要測試可用 | `scripts/phase1a.py`、config、`run_phase1a.sh`；10 unit tests、shell語法及git diff whitespace檢查通過；README/runbook/onboarding記載已實跑入口；无新增dependency | PASS |
| 驗證器/manifest綠燈真正覆蓋目標，不接受proxy完成 | 先獨立重tokenize／checkpoint hash／216個RTN／24 CI，再對照本表；另外把真manifest中10類控制改成失敗，analysis全部拒絕且原輸出不變，涵蓋「CSV齊但最後restore失敗」情況 | PASS |
| 缺GPU/checkpoint/data/OOM時保存command/error/completed/missing至blocked並回報未完成 | 舊CUDA及初次snapshot失敗紀錄完整保留；launcher synthetic inventory failure測試有log/blocked且exit非0；本輪資源足夠、無OOM，所以現在無需pause | PASS（歷史/條件式） |
| 結論限本模型/資料/方法/contexts；不宣稱packed low-bit加速或實際記憶體壓縮，不開始後續phase | report限制完整，實際程式無packed GEMM／KV／activation介入；只跑固定14格，沒有擴資料或挑排序 | PASS |

## 實測結果的有限摘要

- BF16平均NLL：512為 **2.91952682**（PPL18.532516），2048為 **2.62236190**（PPL13.768204）。
- 4-bit ΔNLL 在兩長度的均值排序均 **WV > WQ > WK**：512分別+0.04489078/+0.02838700/+0.01412003；2048分別+0.04409224/+0.03523798/+0.00598801。
- 4-bit KL 的WV也最大；Q與K的KL配對CI在兩長度都跨零。**NLL排序不能當成所有metric相同的排序**，也不能證明Q/K等價。
- 8-bit各條件 |ΔNLL| ≤0.00148305 nats/token，KL約0.00109–0.00124。少數負ΔNLL只是此固定樣本結果，不解釋成通用改善。
- 完整值與探索性95%CI見 [`phase1a_report.md`](phase1a_report.md) 與未四捨五入的analysis JSON。

## 稽核中發現並解決的問題

1. **非必需snapshot檔案**：初次smoke被缺`.gitattributes`擋下；改為只要求指定模型/tokenizer/test必需檔案後，真模型與正式run均成功。未偽造缺檔、未更換資產。
2. **Reporter可能假報控制成功**：read-only reviewer發現CSV896筆但最終restore失敗仍可重算report的漏洞。正式run前已加入控制gate；最終對真manifest作10個失敗變體均被拒绝，包括最後restore=False。未藉修改manifest旗標補成功。
3. **跨CPU/CUDA FP32 rounding**：原verifier用NumPy直接max/127，與PyTorch CUDA用FP32倒數乘法的scale相差少量ULP，會翻轉半步邊界。診斷在真實WK layer0 8-bit找到283/2621440個權重差異；NumPy明示max×float32(1/127)後與CUDA的scale、BF16權重、已記錄hash完全一致。依相同已釘選數值語意重算後，**全216個實際介入hash皆bitwise相同**；沒有放寬為allclose忽略量化差異，也未改正式runner或數據。相對L2使用FP64 oracle核對時才容許FP32 reduction rounding誤差。

細節：[`phase1a_rtn_audit_diagnosis_20260908T103156Z.log`](phase1a_rtn_audit_diagnosis_20260908T103156Z.log)、[`../doc/debug-report.md`](../doc/debug-report.md)。**不保證將quantizer搬到CPU或換PyTorch/CUDA後仍bitwise重現**；固定CUDA執行環境及已記錄hash是重跑契約的一部分。

## 統計解讀檢查（11/11）

| 風險 | 本次核對 |
|---|---|
| Simpson's paradox | 各bit/context分開列，不以混合總排序取代分層結果 |
| Ecological fallacy | 推論單位為這批blocks/端到端模型，非個別token或所有語言模型 |
| Berkson/selection bias | 固定原始順序、seed42不依metric篩選；仍承認單一語料的外部效度限制 |
| Collider bias | 無依結果調整的控制共變數或事後篩選 |
| Base-rate neglect | 非診斷sensitivity/specificity研究；NLL/KL是全vocab分布計分 |
| Regression to mean | 沒有挑極端loss blocks作介入，不將負ΔNLL解釋成普遍改善 |
| Survivorship bias | 全896筆俱在，無丟棄NaN／困難樣本；失敗smoke不是正式樣本 |
| Look-elsewhere effect | 全24對比完整揭露、未做多重校正，僅探索性；不以CI越過0作確認性宣稱 |
| Garden of forking paths | phase規格與正式config先固定；未在看到排序後改sample/bit/方法 |
| Correlation/causation | 只有明確weight介入的本模型端到端比較，不推導KV或普遍機制 |
| Reverse causality | 先介入再forward，沒有以結果反向選介入條件 |

整體解讀為 **CAUTION：有限語料、64 blocks可能相關、探索性多重比較、8-bit微小效應與浮點邊界**；這些不妨礙完成規定的敏感度實驗，但限制科學推論。

## 實際驗證命令與原始證據

每次先 `source ~/.venv/bin/activate`、`nvidia-smi`。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -s tests -v
bash -n scripts/run_phase1a.sh scripts/phase1a_preflight.sh
python scripts/phase1a.py --analyze-only --output results
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/verify_phase1a.py --output results \
  --compare-smoke results/smoke_20260908_fixed
git diff --check
git diff --exit-code -- AGENTS.md phase1a.md
```

- 正式stdout/stderr：`phase1a_execution_20260908T101238Z.log`／`phase1a_formal_launcher.log`。
- 正式exit code：`phase1a_formal.exit`。
- Independent audit：`phase1a_verification_20260908T103311Z.log`、`phase1a_verification.json`。
- 統計重算hash一致、10失敗控制gate、10 unit tests與syntax/whitespace：`phase1a_final_checks_20260908T103535Z.log`。
- 兩個smoke source/config hash與正式run一致；獨立smoke與正式sample0 metrics精確一致。

## 未驗證但不屬本目標必需交付的事項

沒有在全新venv或別台GPU上重装重跑整套14條件；完整本機入口已真實成功執行，獨立smoke亦精確重現，但不能宣稱跨環境bitwise再現。無build/type/lint framework、CI、PR、push或部署要求，未假稱其通過。沒有進行packed kernels、serving、KV/activation或後續phase。

保留所有正式數據、初次失敗與修正證據；沒有剩餘阻塞需要以假資料或降低規格填補。
