Phase 1a — Qwen3-4B projection-weight sensitivity

狀態：待實作與執行；本檔是實驗規格，沒有實測結果。

唯一問題

在相同量化規則與輸入下，單獨降低 WQ、WK、WV 的表示精度，對 Qwen3-4B 的預測分布與語言模型損失，是否造成不同程度的影響？接受差異不明顯或排序隨條件改變。

固定範圍

待測模型：Qwen/Qwen3-4B 原始 checkpoint，BF16 baseline。Qwen3-4B 指實驗模型；Pi 的 agent 模型另行設定。

先確認本地 GPU、模型和資料可用性；記錄模型/tokenizer revision、資料 revision、套件版本與硬體。正式跑之前固定這些設定。

使用 Transformers、eval 模式、batch size 1、固定 attention backend；評估純文字 teacher forcing，不產生聊天或 thinking 回答。

只改 attention projection weights；一次改所有層的一種 projection。activation、KV storage 和其他權重維持 baseline 設定。每個條件從原始權重開始，禁止疊加量化。

本輪不區分 prefill/decode；teacher-forced forward 使用 use_cache=False。不從此實驗推論 KV lifetime 或 serving 效益。

量化與條件

7 個模型條件：BF16；WQ-only 8/4-bit；WK-only 8/4-bit；WV-only 8/4-bit。

使用相同 symmetric round-to-nearest fake quantization。沿每個 output row 的 input dimension，每 128 個權重一組；不跨 row 分組。scale 以 FP32 計算。

對 b-bit：qmax = 2**(b-1)-1、scale = max(abs(group))/qmax；全零 group 直接保留零。dequant = clamp(round(w/scale), -qmax, qmax) * scale，再轉回 BF16 運算。

這是表示誤差實驗，沒有 packed low-bit GEMM；不得宣稱加速或實際記憶體壓縮。

保存實際修改的 module 名稱、shape、參數數量與相對權重誤差。Qwen3-4B 是 GQA（Q 32 heads、KV 8 heads），本輪測原生模型中的端到端影響，不是等參數量或等成本的理論比較。

資料與量測

資料：Salesforce/wikitext、wikitext-2-raw-v1、test split。固定原始順序，以兩個換行連接 text，tokenize 時不額外加入特殊 token；保存 revision、文字雜湊與 tokenizer 設定。

將 token stream 切成不重疊的 2049-token blocks，丟棄末尾不足部分；用固定 seed 42 抽取 64 個 blocks。保存抽樣索引和 token IDs；不足 64 個時回報資料問題，不偷偷降低樣本数。

對同一批 blocks 分別取前 513、2049 tokens，用 shifted labels 評估 512、2048 個 next-token predictions。所有條件共享完全相同的輸入與計分位置，共 7 × 2 = 14 個條件。

記錄每個 block 的平均 NLL、相對 BF16 的 ΔNLL，以及 KL(P_BF16 || P_test)。softmax/log-softmax 與 metric reduction 用 FP32；按位置分塊計算，避免同時保存大量完整 logits。

PPL 由全部計分 token 的平均 NLL 取 exp；不可平均各 block 的 PPL 當整體 PPL。

先以小樣本跑通 BF16 和 WQ4，再跑完整固定條件。核對介入 module、非目標權重未變、移除介入後回到 baseline，以及 logits/metrics 有限且計分對齊。

各 bit/context 分開比較 Q/K/V 的 paired 差值；以相同 block 索引作 2000 次 paired bootstrap（seed 42），報告探索性的 95% CI。不得把 deterministic 重跑算成獨立樣本，也不得把 CI 跨零解釋成已證明等價。

交付與停止

在目前實驗專案新增最小可重跑的程式與設定；保留既有檔案。交付 results/phase1a.csv（逐 block 數值）、results/phase1a_run.json（完整設定與執行命令）、results/phase1a_report.md（數據、CI、結論與限制）。

每輪只檢查最近失敗、修正必要程式並更新簡短進度；不自動擴大資料、模型、bit、任務或開始下一階段。

完成標準：14 個條件與必要控制均有真實結果，且能以保存命令重跑。只寫好程式不等於實驗完成；沒有明顯差異也可以完成。

缺 GPU、checkpoint、資料或遇到 OOM 時，保存命令、錯誤、已完成項與確切缺項至 results/phase1a_blocked.md；報告未完成，不用替代模型、假數據或無限重試填補。Pi 的 pause 由使用者控制；回報需要 /goal pause，不要把阻塞冒充實驗達成。

結論限於此模型、資料、量化方法與兩個 context 長度。後續的 activation、KV cache、phase 實驗各自另開 goal；第二至第五階段等第一階段證據後再決定。
