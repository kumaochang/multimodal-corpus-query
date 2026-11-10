# Clip-level fusion metrics

## 固定設定

- 身份沿 `corpus/demo_data.json` 的 30 個 clips、原序與 `pilot/labels.json` 的採用標籤。與 `labels_log.json` 的兩筆更正身份核對。
- 四個區塊依序為 text=T（字幕 XLM-R 768d）、audio=A（MFCC86）、prosody=P（phone-interval8d）、kinematic=K（18d）。`full_fusion_whisper` 是歷史 key，實際 audio 為 MFCC。保留全部十五個非空配置，沒有 Whisper 替換、K14 替換或新權重。
- T／A／P 使用 `analysis/_emb_cache.npz`。K18 使用 `analysis/_q8_cache.npz` 的 `ids`／`K18`，來源定義見 `q8_gesture_conditional.py` 與 `q5_baselines.py::_pose_features`，並以保存 kinematic 相似度與單通道／四通道結果核對。
- 先核每檔 ID 唯一、集合與順序，再對齊至固定 query 序，保留原序及 permutation。此分析使用上述保存數值，不重新擷取特徵。
- 沿 `pilot/experiment.py` 的 normalize(block)、hstack、normalize(fused)、cosine_similarity 計算，不另縮放。記錄實際 dtype、C/F layout、套件、BLAS／thread。以單 thread 執行一次。
- 原排序為自身分數設 −1，再 `np.argsort(kind="quicksort")[::-1]`。為核對歷史平均值先重現其完整排序，再移除自身，保存兩種排序。所有 exact ties、名次及候選身份隨排名 cache 保存。程式檢查自身位於原第 30 位，才使用對應的歷史 AP 關係。
- 修補排名每 query 只有其餘 29 個候選。P@1 為第一名相關指示，P@3 為前三名相關比例。AP 為相關候選位置 precision 的平均，以所有 relevant candidates 作分母；另以 relevance 累積和公式獨立驗算。
- 全部 30 queries 的 AP 應有定義。若沒有 relevant candidate，AP 設 undefined 並記分母，不能填零。空候選亦無分數。有效分母與退化數量隨結果保存。
