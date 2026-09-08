# 逐層 projection-weight 敏感度：固定實驗協定

本協定在本輪模型 forward 之前保存。只回答：單一 layer 的 WQ/WK/WV group128 RTN 4-bit 表示誤差，是否有少數層特別敏感？不要求一致排序，不擴展 scheduler、activation、KV、模型、資料或 bit 數。

## 固定條件

- 來源為現有 `results/phase1a_run.json`、`phase1a_config.json`、`phase1a_tokens.npz`；歷史 bundle 已被使用者移到 `results/`，不使用已不存在的 archive 路徑，也不改寫歷史紀錄。
- Qwen/Qwen3-4B model/tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`；WikiText raw test revision `b08601e04326c79dfdd32d625aee71d232d685c3`。沿用全部 checkpoint hashes 與 400 個原始 parameters/buffers hashes。
- 完全沿用保存的 64 個 2049-token blocks 與抽樣順序；每個 block 固定 2048 個 shifted next-token 預測，共每條件 131072 個位置。
- 條件：BF16 + `L00_WQ4`、`L00_WK4`、`L00_WV4` … `L35_WV4` = 109 條件（108 介入），6976 逐 block 列。層號為 **0-based**。
- 只量化該層該 projection 的 weight，其他所有參數仍是原始 BF16；buffers 與運算設定不變。GQA Q32/KV8，WQ shape [4096,2560]、WK/WV [1024,2560]，不是等參數或等成本比較。
- 重用既有 `fake_quantize`：每 output row 沿 input 維 group128；FP32 maxabs/7、ties-even round、clamp[-7,7]、全零 group 保零，dequant 回 BF16。仍 BF16 GEMM，不能宣稱 packed low-bit 加速/記憶體壓縮。
- Transformer eval、batch1、use_cache=False、SDPA 僅 FLASH_ATTENTION，deterministic、TF32 off。保存環境、執行命令、來源快照及 GPU 資源紀錄。

## 計分、必要控制、續跑

- 重用既有 hidden/head chunked scorer，FP32 log-softmax 和 reduction，KL 方向 `P_BF16 || P_intervened`；保存 NLL、ΔNLL、KL，保留微小負 roundoff 而不裁切測量值。
- 本轮 BF16 全64 blocks 必須與歷史2048結果逐block精確一致；保存 lossless BF16 hidden cache 供 KL 和續跑使用。此控制/重跑不增加統計樣本。
- 首末 block 的首末64 positions 比較 CausalLM API logits 與 shifted cross entropy，驗證計分對齊。
- 每個介入前、剛介入、評估後、還原後，逐一檢查**全400個 parameters/buffers**。GPU保留不可變的原始tensor副本，以uint8 bitwise equality逐一比對，避免每次將全checkpoint複製回CPU雜湊；首次全model SHA256仍須精確匹配歷史原始checkpoint。介入差集必須恰為一個指定weight；另存原始/量化SHA256、shape、參數數、相對L2誤差。
- 每次還原後另在首末 block 驗證 hidden bitwise相同、NLL相同及KL=0。例外時finally還原；未通過全部控制的條件不能標為完成。
- 每條件完成後原子保存獨立 JSON（64列 + 控制）。`--resume` 僅執行缺少的條件；已完成條件、來源設定、token/cache/artifact hashes 必須先驗證，拒絕覆寫。失敗的部分輸出保留在 attempts/，不當成完成條件。`--max-conditions` 只限制本次完成多少完整介入，不改樣本數或總目標，可用於先檢驗第一條件再續跑。
- 無資源/OOM時保存 exact command/error、已完成/缺少清單，不換模型、不縮樣本、不無限重試；回報未完成並請求所缺資源及使用者 `/goal pause`。

## 預先固定的分析與圖表

- 每層 Q−K、Q−V、K−V 的 ΔNLL 與 KL **配對比較**，共216個對比；以相同64 block索引、seed42、2000 paired bootstrap、percentile探索性95% CI。另提供108介入vs BF16的ΔNLL/KL CI與NLL/PPL摘要。
- 敏感度集中度：對各projection各自以及跨Q/K/V的layer score，計算36層的非負平均KL分數，以及 `max(mean ΔNLL, 0)` 分數；跨projection分數為三個獨立介入的score相加，**不是聯合量化的實測或可加性主張**。
- 報告top1/top4/top8占比、達50%/80%分數所需層數、effective layers `(sum s)^2/sum(s^2)`；完整排序與累積曲線。top4=36層約11%，描述性參考為top4占比≥50%稱「高度集中」，不作假設檢定或普適界線。
- 同一組draws重抽blocks並每次重新排名，給top4占比及effective layers探索性CI；不能將事後選出top層的固定排名當成無選擇偏誤的確認性結论。
- 負ΔNLL保留在逐層表/圖；集中度的正部定義明示，若總正部為0就記為undefined而非捏造占比。KL很小仍可能包含BF16數值擾動。
- 交付層號×metric曲線（Q/K/V與探索性CI）、216配對對比圖、累積集中曲線，PNG300dpi及vector PDF；由真实CSV/JSON生成。
- 跨零不是證明等價；全部比較揭露，無多重比較校正。結論限此checkpoint、64 blocks、2048位置、RTN；有限語料blocks可能相關，不推廣到其他模型/資料或scheduler。

## 完成證據清單

1. 可重跑runner/config/launcher、唯讀歷史hash快照、實際命令/環境/來源快照。
2. 109有效條件、6976行逐block NLL/ΔNLL/KL、同token/label hash與原始BF16重現。
3. 108次單weight介入與完整還原（全state+首末block輸出），固定BF16和無疊加。
4. 配對summary/CI、集中度完整表、圖表與有限結論。
5. 自動測試、獨立checkpoint/RTN/statistics/artifact核验、續跑no-op實測、逐條prompt-to-artifact completion audit。測試/manifest綠燈不能單獨取代這些證據。
