# ULFR-v1 numerical specification



## 3. 固定數值環境

**確定性排名與統計**：（Python 3.10.22、NumPy 1.26.4、SciPy 1.13.1，NumPy 內建 OpenBLAS 0.3.23.dev、SciPy OpenBLAS 0.3.27），`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`，並在程序內以 threadpoolctl 核實每個 BLAS pool 為 1 thread。§6.2 使用同一正式排名環境。

**Posterior**：隔離環境 `data/study3_stats/ch5_reproducibility/environment`（Python 3.12.0、CmdStanPy 1.3.0、NumPy 2.2.6、SciPy 1.15.3、ArviZ 0.22.0）與 `toolchain/cmdstan-2.37.0`，系統 g++ 13.3.0。正式 binary 於正式根目錄以相同來源、`O=3`、stanc `--O1` 重新編譯並記錄 hash。每 chain 一個 thread。

兩個環境讀同一 release 的統計輸入：posterior 只讀本 release 產生的 11 個 model-data NPZ。實際 CPU dispatch、BLAS library 路徑與 binary hash 於執行時保存。

## 4. 排名與分數契約

- 輸入：七項凍結檔（`features_full/features.npz`、`tfidf_sub.npz`、`tfidf_tr.npz`、`label_analysis/units.tsv`、`unit_table/units_k.npz`，及 兩份歷史彙總 JSON 僅作比對），214 個核心 TextGrid 數值界線，s62 另讀既有 `work/data/pose/*.pose.json` 與 `work/data/aligned_trees`。不重新抽取 embedding、不重訓 TF-IDF、不重跑 ASR／VLM／pose／face。s62 程式內由既存 pose 記錄計算描述子是其原登錄程式的一部分，主要 14d 須等於凍結 `units_k.npz` 的 K（原 assertion 保留）。
- 第五章與補表、探索：原 `l2rows`／標準化，float32 cosine，dense `Q @ C.T` 或凍結 sparse 乘法，每來源 512-query chunk，候選為 feature-row 順序，`argsort(-score, kind="stable")`，不 round。排除：W1 自身；W2 另排同 clip；W3 另排同播出日；X 為其他來源社群全部候選；診斷 DIAG_LANG 為全部語言、排同播出日。
- §6.2：原 `unit_norm` float64 cosine，每有序語言對，top-1 為 `np.argmax`（第一個最大值）；item 8 以 `-inf` 遮蔽同 clip。正式記錄用 stable 降序排序保存完整有效候選，其第一名必須等於原 argmax。
- 不改 dtype、chunk、正規化、排序、thread 或 backend 以接近舊值。
- 每個排名條件保存：query 列（凍結 `features.npz` ids 的 row index）與順序、候選 bank、逐 query 完整有效排名（int32 row）與分數（原計算 dtype，float32 或 float64）、offsets、n_cand、n_rel、chance、AP 是否定義、排除規則、來源程式與條件描述；另存 top-10 衍生視圖。內容相同的排名以 payload hash 去重，manifest 列出所有條件別名。
- 逐 query 指標由各原程式輸出保存；另以獨立程式只讀完整排名重算 hit／chance／P@3／AP，與原程式逐 query 值核對。

## 5. 統計契約（全部沿用）

- 原主要程式：seed 20261019（主要）、20261021（敏感度）、20261021／20261019（§6.2 主要／item 8）、20261023（by-label）；各 10,000 次；共用 RNG 的呼叫順序由原程式決定。
- 補表：PCG64 seed 20261020，每條件語言內播出日重抽，query-weighted ratio-of-sums。
- 探索：PCG64 seed 20261024，72 個來源／日共享權重，固定 adopted overlap cells／weights，24 項家族，ddof=1，95% max-standardized deviation，聯合有效門檻 99%。
- 正式版保存自身的 bootstrap 權重與 draws；不宣稱找回舊 175 條件逐條件權重。未定義值保留為未定義。

## 6. 階層模型契約（全部沿用 `posterior_20261020/RUN_SPEC.md` 與 `COMPUTE_CONTINUATION.md`）

- 11 套 base：H0 W1/W2/W3/X、HA W1/W2/W3/X、HB W1/W2/W3；公式、Helmert contrasts、先驗、offset、結構性 0/1、權重、預測目標不變。
- 55 套敏感度：prior_half、prior_double、flexible_baseline、add_clip、no_day_slopes；2 套 benchmark：H0_W1 day_holdout（固定第一個來源／日）與 day_weighted（PCG64 202610281）。
- Seeds：base seed（H0 202610250+p、HA 202610260+p、HB 202610270+p）+ 10000×v（v 依上列順序，holdout 6、weighted 7）。
- 第 1 次 attempt：4 chains、2,000 warmup／2,000 retained、adapt_delta .95、max_treedepth 12、chain IDs 1–4、`sig_figs=16`。實作：H0 base 用 original Stan coding，其他全部用已驗證等價的 native_events coding（與歷史採用 attempt 相同）。
- HA base 的 Windows warm-start fixed-metric（attempt 4）及 warm-start 4000/4000（attempt 5）是 Windows 中斷後的計算處置，正式版不沿用；HA base 與其他 base 相同從第 1 次 attempt 開始。
- 敏感度與 benchmark 依 COMPUTE_CONTINUATION 規則，以同模型完成並診斷的正式 base fit 最終狀態與對角 metric 作提案初始化，仍做完整 2,000 次 adaptive warmup；不使用舊 draws。
- 診斷：參數與全部報告 estimands Rhat ≤ 1.01、bulk/tail ESS ≥ 400、0 divergences、0 treedepth saturation、E-BFMI > .3。未通過者最多一次修復：同 seed、同實作、4,000/4,000、adapt_delta .99、max_treedepth 14。仍未通過即列為未解決，不再加抽、不換 seed、不改設計。全部 attempt 保留。
- 後處理：原 `ch5_posterior_report.summarize`（conditional／population-average、GH40 與 GH80 檢查、PPC、paired-hit 相關、PSIS、day influence、固定 contrasts）、`collect`、identification audit、refit support；只改輸入輸出路徑與 model-data hash 來源。
- 資源：先跑 H0_W1 base benchmark 量測時間／RSS，再排程；最多 7 套同時（28 chains／32 logical CPUs）。
