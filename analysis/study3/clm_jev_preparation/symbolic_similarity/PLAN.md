# Finite-bank symbolic and similarity retrieval

## 固定母體與來源

原六日／六日 split。Q 為原 items 次序下 seed half 的全部 10 個首輪肯定項目。主 C99 為 scoring half 全部 99 個可判斷項目，11 個肯定、6 日／44 clips。敏感度 C112 為 scoring half 全部 112 個首輪項目，保留 13 個 cannot-tell。原序、固定 IDs 與來源 SHA 由 SOURCE_CONTRACT 核對，沒有重測投票或挑 seed。

逐筆 labels、IDs、scores、排名與 case table 屬計算輸入或輸出；群體摘要另行保存。原 1,964 語句 pool 沒有完整顧問參照，估計對象為這個有限觀察 bank。

## 固定路徑與計算

Phone 沿原 saved in_A，Forms 沿 saved form in {marker,suffix}，不重跑偵測或修規則。C99 budgets 為8／90，C112 budgets 為8／95。三條 similarity 為 Audio（Whisper／prosody／K14）、ASR XLM-R、ASR TF-IDF。只讀 item5 凍結 float64，不加入 face／CLM／MFCC／form indicators。

float64、C-contiguous，BLAS／OpenMP 單 thread。raw cosine 的 zero-vector score 為 0。每 seed 在各 bank、各通道的有效 candidates 上作 population z-score（ddof=0）；sd=0 時有效位置為 0。K 僅 query／candidate 皆非 missing 才有效，每 seed 少於兩個有效 K candidates 時該 seed 的 K 不進融合。Audio 每 pair 取可用通道 z 分數等權平均，再對全部 10 seeds 平均。文字兩路亦 raw cosine→逐 seed bank z-score→全部 seeds 平均。最終用原候選順序 stable descending argsort。沒有 temperature、訓練權重、max-over-seeds 或最佳 seed 選擇。

prosody 使用完整凍結240-row無標籤材料作每維 mean／sd，NaN 以該維 mean 補，all-missing／sd=0 設0，再 L2。參數共用兩 banks；各 bank 的 score z-score 重新計算。K missing rows 不插補，其他非預期 NaN／Inf 或 ID／來源不符即停。

## 比較與未知判斷

兩個 symbolic budgets 各對三條 similarity，共六對照／bank。保存全部 known-positive targets 的 symbolic only／similarity only／both／neither，precision／reference-coverage 的 numerator／denominator，原設計 1/incl_prob 的描述式加權數值。unknown bank 的 point precision 保留 undefined，另列已知部分與 unknown-returned 數，避免當作陰性。

對 C112 的13個 unknown 用全部8,192種共享 binary completions 枚舉。每個 completion 同時套 symbolic 與三條 similarity，求個別 precision／coverage 及直接配對差的 logical bounds，分未加權及原權重。配對 bounds 不由兩個 marginal bounds 相減。unknown 原旗標保持，completions 是界定可能值的假設，不回寫答案。

K=0 回傳空集合，precision與matched差無定義，reference非空時coverage=0。K=bank size 全取並標記無排序區分力。沒有 relevant reference 時coverage無定義。Q空則不執行幾何路徑。實際Q/C／budget意外改變先停止。
