# Review findings 與修正證據

原始唯讀reviews保留在 `runner_review.md` / `statistics_review.md`，沒有改寫reviewer當時的判斷。本文件是parent修正/驗證紀錄，不冒充第二次independent review。

| Finding | 最小修正 | 真實驗證 |
|---|---|---|
| P1 完整run的no-op resume跳過BF16 cache核驗 | `completed_results()`共同路徑核對cache metadata、output內路徑、存在與SHA256，不需forward；`validate_hidden_cache()` | `test_resume_inventory_rejects_corruption_and_incomplete_gate`覆蓋corrupt/missing/metadata缺失；109格synthetic no-op將`run_missing`設為必須不可呼叫；真實12:34:12 attempt skip109、new0、no_forward_needed=True，cache SHA與109条件hash不變 |
| P1 後續失敗使之前已commit條件永久無法分析 | analysis gate承認先前已逐條通過所有介入/還原控制的原子commit，核對producer載入/identity；blocked/running來源attempt保持原狀；成功attempt仍須核對final state與newly_completed | `test_interrupted_attempt_keeps_prior_valid_commits_for_analysis`以109格temporary fixture重現commit→later blocked→resume→analysis，原blocked不改；把該舊commit的restored改False仍拒絕 |
| P2 exact ties被印為嚴格排序 | `projection_order()`只對exact equal顯示`=`，不引入任意近似容差 | all equal、兩種two-way tie與1e-12不相等回歸測試 |
| P2 缺analysis/plot runtime版本 | `analysis_manifest.json`加入actual command、Python、NumPy/Matplotlib/Pillow/torch版本、Agg backend與byte identity限制 | 已讀actual manifest；在同環境重算13個分析/圖表artifact全部逐byte相同，`completion_evidence.json`保存原hash |

## Producer provenance沒有被改寫

- GPU實驗於12:31:26 UTC完成後，才修改runner的CPU驗證/來源核對邊界。原producer快照在 `source/scripts/layerwise_projection_sensitivity.py`，其hash仍精確等於原run manifest。量化、forward、打分、介入與還原函式沒有改動。
- 新validator可核對舊producer快照，不假稱舊結果由新版source生成。**若要補新的missing條件，仍強制current source和該run原producer hash完全相同**；不相同就拒絕，需要使用保存的原producer source。這個拒絕已實際測試，記錄於completion evidence。
- 本輪新版no-op的實际validator來源另外保存在 `attempts/20260908T123412Z/validation_source/scripts/` 及該attempt的`validation_source_sha256`；沒有更新原producer hash來掩蓋版本差異。
- 兩次GPU producer分别新增2條件及107條件；新版no-op新增0條件。沒有為修cache gate重跑GPU實驗，也沒有改任一condition或歷史結果。

最終22個unit/regression tests全部通過：`final_unit_tests_20260908T123334Z.log`。另對真實L00_WQ4控制/row做14種in-memory失敗變體，全部拒絕，原始檔案不變。這些結果不取代獨立checkpoint/RTN/statistics audit及人工圖表檢查。
