# 逐層 projection-weight 敏感度：完成稽核

## 目標重述與判定

本目標的具體交付是：

1. **可重跑程式/固定設定**：沿用歷史Qwen3-4B checkpoint、相同64 blocks及group128 RTN；只評2048個next-token位置/block。
2. **真實BF16＋108介入**：36層×WQ/WK/WV，每次只量化一個weight為4-bit，其餘原始BF16；逐次驗證介入、非目標不變及還原，不能累積量化。
3. **數據與分析**：逐block NLL/ΔNLL/KL、配對比較、逐層圖表，分析是否少數層敏感，不要求一致排序。
4. **歷史與續跑**：只補缺少條件，保留舊結果/失敗證據，不擴展scheduler；缺資源時如實阻塞而非縮規格。
5. **真正完成證據**：實際執行、獨立數值重算、控制失敗測試、可重算圖表及本逐要求稽核，不能僅靠manifest或tests。

**判定：上述實驗與交付已完成，沒有本目標的必需缺項。** 以下逐项以實際檔案/執行/核驗為依據，而非以`status=measured`單獨認定。沒有擴展模型、資料、bits、KV/activation或scheduler。

## 實際執行與續跑

每次先 `source ~/.venv/bin/activate`、`nvidia-smi`；model forward僅用當時空閒GPU2，UUID `GPU-74d97f46-6284-1055-698a-e2db4e9c744b`，basic-1 RTX3090。

```bash
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b \
  bash scripts/run_layerwise_projection_sensitivity.sh results/layerwise_projection_sensitivity --max-conditions 1
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b \
  bash scripts/run_layerwise_projection_sensitivity.sh results/layerwise_projection_sensitivity --resume
# 完成後，以補強cache gate的新版runner驗證no-op（無model forward）：
CUDA_VISIBLE_DEVICES=GPU-74d97f46-6284-1055-698a-e2db4e9c744b \
  bash scripts/run_layerwise_projection_sensitivity.sh results/layerwise_projection_sensitivity --resume
```

| Attempt（UTC，2026-09-08） | 真實新增／跳過 | 結果 | 時間 | 峰值torch allocated |
|---|---|---|---:|---:|
| `attempts/20260908T112911Z/attempt.json` | BF16、L00_WQ4；skip0 | bounded_partial，兩條完整64-block且全控制通過 | 127.615秒 | 15.813GiB |
| `attempts/20260908T113140Z/attempt.json` | 其餘107介入；skip2 | complete；`resume_20260908T113138Z.exit`=0 | 3586.610秒 | 15.823GiB |
| `attempts/20260908T123412Z/attempt.json` | new0、skip109 | complete、`no_forward_needed=true` | 4.957秒 | 未配置CUDA/model |

兩次producer新增集合互斥，聯集恰109。全部逐條還原後才commit；無失敗條件/NaN被丟棄。GPU producer約61.90分鐘（含載入/hash/控制），不是speedup benchmark。所有資源充足，沒有OOM，GPU任務已退出並釋放GPU2。

## Prompt-to-artifact checklist

| 明確要求／成功條件 | 已檢查的實際證據 | 結果 |
|---|---|---|
| 在pcb實作並真實執行，而非只有方案 | repo `scripts/layerwise_projection_sensitivity.py` / config / launcher；兩個producer attempt與stdout逐條COMMITTED，exit0；109個非空condition JSON | PASS |
| 沿用已封存Qwen3-4B checkpoint | `historical_run.json`與原 `results/phase1a_run.json`相同；model/tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c`；3個checkpoint shard SHA及config SHA核對；兩次載入400個state hashes精確匹配歷史；獨立safe_open核對398個BF16參數 | PASS |
| 原始BF16基準，其他參數BF16 | loader檢查全部parameters BF16、36層/Q32/KV8、tied head；BF16 64筆NLL逐筆精確等於歷史2048 rows；baseline self-KL/ΔNLL全0 | PASS |
| 相同64 blocks，不偷偷改抽樣或縮样本 | `sampled_token_blocks.npz` byte hash等於歷史npz；独立重tokenize4358原始rows/299078 tokens、雙換行join、不加special tokens；145 full blocks、尾1973 tokens丟棄；seed42選64順序及每個token精確一致 | PASS |
| 固定2048個預測位置 | 6976列均`context=n_tokens=2048`；input 2049 tokens、labels=`tokens[1:]`，input/labels hashes逐列比對；每條件131072位置，109條件共14286848計分位置但只有131072不同的語料位置，不冒充獨立樣本 | PASS |
| 每次只量化一層的一個WQ/WK/WV為4-bit，共108 | condition集合精確等於L00–L35×WQ4/WK4/WV4；108個control的`changed_tensors`各恰一個指定weight；name/shape/參數數核對，無8-bit/Wo/全層介入 | PASS |
| group128 RTN規則沿用 | producer重用既有`fake_quantize`：row內input分組128、FP32 maxabs/7、ties-even round/clamp、零group保零、dequant BF16；NumPy獨立重建108個量化weight SHA精確等於本輪與歷史對應RTN4 SHA；relative L2獨立FP64核對 | PASS |
| 每次核對介入及還原、無疊加 | 所有400個parameters/buffers逐一uint8 bitwise比對（原始GPU副本）；pre、applied、eval後、finally還原全通過，差集恰一個weight；每次還原首末block hidden bitwise相同、NLL相同/KL0，108×2=216 predictive checks | PASS |
| 其餘原始權重、buffers與運算不變 | full-state comparisons含non-persistent buffers；activation/KV未量化；eval、batch1、use_cache=False、SDPA僅FLASH_ATTENTION、deterministic/TF32 off；每次載入同runtime版本，環境/數值設定實記於attempt | PASS |
| 打分有限、方向及位置正確 | scorer原碼與unit tests核對KL(BF16\|\|test)、FP32 log-softmax/reduction；首末block×首末64位置四個API檢查：head logits與CausalLM bitwise相同、shifted NLL與direct CE相同；所有block logits/metrics檢查finite | PASS |
| 逐block NLL/ΔNLL/KL | `block_metrics.csv`6976列（109×64），與不可變condition rows逐欄相同、無重複；delta逐列重算；不是只有均值或synthetic fixture | PASS |
| 配對比較 | `paired_comparisons.csv`216列：36層×3個Q/K/V配對×ΔNLL/KL；`layer_summary.csv`109摘要與各vs-BF16 CI；相同64 block配對、seed42的2000×64 draws；獨立對兩arm相同draws重抽再相減，均值/percentile95%CI全部一致 | PASS |
| NLL/PPL合理彙總 | 每block FP32 NLL；統計層FP64平均，PPL=exp(mean NLL)，不是mean(block PPL)；独立重算109格。BF16 aggregate與歷史FP32 aggregate約8e-8差異僅彙總dtype，64個原始block值完全相同 | PASS |
| 分析是否集中少數層 | `concentration.json`與analysis內8 profiles；事先protocol固定positive-mean-ΔNLL / nonnegative-mean-KL、top1/4/8、50/80%所需層數、effective layers；ALL不是聯合量化損害；獨立重算36層完整排序/曲線、2000次重新排名的CI | PASS |
| 逐層圖表 | `figures/layerwise_metrics.{png,pdf}`、`paired_layerwise_differences.{png,pdf}`、`layer_concentration.{png,pdf}`；300dpi與vector PDF，3組均已實際開圖檢查：層0–35、nats/token、圖例/CI/零線完整，無截字，峰值/符號及集中曲線吻合表格 | PASS |
| 接受無一致排序 | report揭露所有排序頻數：ΔNLL出現6種Q/K/V排列，KL出現4種；25/108 ΔNLL對比CI與21/108 KL對比CI跨零，明示不證等價；40/108負ΔNLL保留。沒有為追求排序追加條件 | PASS |
| 可重跑程式、命令、環境交付至指定目錄 | `source/scripts/`原producer/分析/核验source快照；`experiment_config.json`；每attempt實際command/env/versions/GPU；report可複製的UUID/new-output/resume/analyze/verify命令；README/runbook/onboarding已更新 | PASS |
| 每次只補未完成條件 | 第二producer明確skip原先2條件、只新增107；完成後新版no-op skip109/new0/無forward；所有109 condition SHA不變；source drift若要執行新measurement仍拒絕，不混用producer | PASS |
| 保留歷史結果 | 新manifest `history_sha256`覆蓋開始時42個既有結果檔案；独立verifier及completion evidence再次核對全相同。未改 `results/phase1a*`/tokenizer/旧smoke/log；只修回歸測試對已搬移路徑的引用 | PASS |
| 不擴展scheduler | scope與實際程式只有一次有界GPU續跑、無排程策略/cron/新任務種類；未量化activation/KV、未做其他模型/bit/資料 | PASS |
| 缺GPU/checkpoint/data/OOM如實阻塞並保留證據 | launcher/runner保留command/error、completed/missing、partial/failed control及blocked，無替代模型/縮樣本/無限重試；synthetic inventory failure測試確認非0和blocked（僅temporary fixture）。本輪無資源阻塞，所以無需要求提供資源/pause | PASS（條件式） |
| 重要缺口修正而非忽略review | 四個review findings及處理詳見`review_resolution.md`；cache common gate、blocked attempt既有commit續跑、exact ties、analysis runtime已補；沒有改原producer快照/hash冒充新版產生 | PASS |
| tests/verifier確實覆蓋要求，不只proxy綠燈 | 22 tests涵蓋RTN/score、單weight/例外/bitwise、條件/row/cache/gate、paired/concentration/zero/ties、blocked→resume→analysis；另對真實control/row 14個失敗變體全部拒絕；本表補足實測/完整交付/視覺與scope證據 | PASS |
| 可以重新生成數據衍生交付 | 同環境執行analyzer兩次：13個CSV/JSON/draws/report/PNG/PDF artifacts全部逐byte相同，109個原始condition SHA不變；hash清單保存在`completion_evidence.json` | PASS |

## 科學結論（有邊界）

- **正ΔNLL分數**在各projection內較集中：top4層占WQ **59.9%**、WK **72.0%**、WV **51.7%**。WK layer32單層占其正ΔNLL分數53.5%，平均ΔNLL **+0.01673276 nats/token**。WV的top4 CI跨50%描述性門檻（47.2%–55.6%），不能當確認性集中結論。
- 三projection分數相加的layer排行，正ΔNLL top4為**32、22、21、34**，占 **39.4%** [35.6%,42.9%]；達80%需16層。不同projection的高敏感層不完全重疊，不能说整體損害必然只在四層。
- **KL分數不高度集中**：ALL top4只占 **14.3%** [14.1%,14.6%]，effective layers **34.64/36**，達80%需27層；各projection也相當分散。
- 因此不是跨metric的一致「少數層主導」。全部結論限此checkpoint/64 blocks/2048位置/RTN4，且CI探索性、未校正多重比較；不推導scheduler或多層聯合量化可加性。

## 自動驗證命令與結果

先啟用venv及檢查GPU，再執行：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
bash -n scripts/run_layerwise_projection_sensitivity.sh scripts/run_projection_quantization_sensitivity.sh
CUDA_VISIBLE_DEVICES='' python scripts/analyze_layerwise_projection_sensitivity.py --output results/layerwise_projection_sensitivity
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/verify_layerwise_projection_sensitivity.py --output results/layerwise_projection_sensitivity
git diff --cached --check -- . ':!results/**'
```

- `final_unit_tests_20260908T123334Z.log`：22/22 PASS。
- `verification_20260908T123429Z.log` / `artifact_verification.json`：原始398參數、108個RTN、6976列、109摘要、216配對對比、8集中度profiles及原資料重新tokenize獨立核验PASS。
- `completion_evidence.json`：兩個producer與no-op清單、13 artifacts逐byte再現、14個真實控制/row失敗變體拒絕、109條件hash、原producer與新validator來源界線。
- `execution_*.log`、`resume_launcher_20260908T113138Z.log`：完整實測及no-op stdout/stderr；不是僅存一個successful旗標。
- Source/docs staged whitespace check通過。未排除artifacts的`git diff --cached --check`會把CSV標準CRLF及原始nvidia-smi/tqdm/reviewer Markdown的行末空白列為warning；這項未宣稱全量通過。為保持原始bytes/hash，不對數據或原始log做trim；這些artifact另以CSV/JSON解析、內容與hash核驗。

## 保留的限制（不是已驗證事項）

- BF16 lossless cache為671089860 bytes，留在本機 `attempts/20260908T112911Z/baseline_hidden.pt`；Git忽略大cache，但BF16.json鎖定SHA。僅取得Git checkout時不具此cache，須取得原cache或另開新output從原checkpoint/blocks重跑；不能假稱缺cache的resume通過。
- 原producer所有source/measurement不變；cache gate修補後只做CPU no-op/測試，没有重跑相同109個GPU條件。新GPU measurement仍必須吻合所屬run的producer source；不放宽hash绕過。
- 沒有在全新venv/另一種GPU/其他套件版本上重裝重跑；PNG/PDF byte identity只驗證本環境。沒有build/lint/type-check framework、PR/CI/push/deployment要求，不宣稱其通過。
- Reviewer是修補前的唯讀review；修補後由parent執行回歸/獨立數值與本稽核，沒有冒稱第二次reviewer審核。

沒有剩餘本目標的必需實驗或交付；若後續要研究不同checkpoint/語料/精度、聯合介入或scheduler，須另立範圍，不能算本次已完成的證據。
