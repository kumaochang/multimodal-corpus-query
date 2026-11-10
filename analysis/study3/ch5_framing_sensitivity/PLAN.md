# Kinematic framing sensitivity

## 固定來源與集合

`analysis/study3/ulfr_local/LOCAL_PATH_MAP.json` 指定正式 numeric、方法 metadata 與凍結輸入的相對位置。reader 驗證 release、manifest 與檔案 hash，不替換缺失資料。六個 `ch5_retrieval|analysis|kin14|kept|own|X|<source>` 條件使用保存的完整排名與逐 query 指標；原數值檔案保持不變。

原 universe 是524個 eligible kept／own 語句，全部 within_shot 且 k_missing=0；
固定原 feature-row 順序、原模型標籤及各來源之外的其他五來源聯集候選 bank。
Explaining 是 query group，候選保留全部原分類。X 跨來源社群，並非嚴格跨實際發話語言。
query 與 candidate 都使用同一處理後表徵；候選不隨 bootstrap 改變。
凍結資料中 F 的 face_extent 有6個 NaN，其餘七欄無缺值，無 Inf。

## 構圖量及固定處理

沿用 `units_k.npz/F` 的八欄：bbox_h、bbox_area、face_extent、pose_extent、
fallback_frac、ppf_mean、frac_1p、frac_3plus。它們量測身體尺度、可見範圍、
偵測／fallback 與畫面人數，且本來就是 §5.4／§6.2 的構圖變項。
F 用整句全部 pose records；原 K 仍只用 tight records。不取新影格或重算特徵。

固定完整集合的處理版本，全部執行，不依結果挑選：

1. **release_original**：只讀正式完整排名，回算既有逐 query 指標。
2. **local_original**：僅對同一524個 K 重排 X，核對本機浮點運算與正式排名的銜接。
   此版本是數值驗證，不取代原結果。若 hit 改變，另列數值差異對 lift 的貢獻，
   保留 release→處理及 local_original→處理兩種配對差值，不替換 baseline。
3. **center_only**：K 減去524個語句的 pooled column means，控制截距中心化本身對 cosine 的作用。
4. **pc3_zero**（主要敏感度）：F 的 NaN→0，column z-score 分母 std+1e-9，
   再中心化，SVD 前3個 PCs；以截距+PCs 對凍結 tight K 作 pooled OLS，取殘差。
   沿用 §6.2 residualize 的線性方法，特別改為本問題的 tight K。
5. **pc3_median**：相同方法，但 NaN 用524個語句各欄 median；檢查零填補的影響。

另固定 **complete-case** 分支：若 F 任一欄缺失，移除該語句的 query 與 candidate，
在剩餘共同集合上各算 **cc_original、cc_center、cc_pc3**。
以 cc_original−local_original 分辨集合變化，以 cc_pc3−cc_original 分辨處理變化；
另列 cc_pc3−cc_center。full 與 cc 的集合、分母與 candidate 範圍分開報告。
不以更有利的集合或 PC 數取代主敏感度。沒有標籤進入 PCA／OLS。

§6.2 的 `14d_resid` 是 all-record 18d 刪除位置後再殘差化，並非 tight K 的單純修正版。
其 ordered-pair、distance-class 等權彙整及 by-label 結果只作既有證據對照，不能代替 X query-weighted 結果。

## 排名、指標、比較量

除正式排名外，各衍生版本均以 float64 擬合轉換，之後沿用原 L2 路徑轉 float32、
零向量 cosine=0、來源內以最多512 queries 的矩陣乘法、stable 降序排序，
同分按原 candidate bank 順序。各候選完整分數與完整排名均保存。
thread 限1，記錄本機 Python、NumPy、SciPy、BLAS、硬體與環境，不更動原環境。
若 full rank 有平台差異，逐項記錄，不只以彙總接近宣稱一致。

優先報告 Explaining mean(hit−chance)，chance 是每個 query 的同標籤候選比例。
同時計算 P@1、P@3（不足3取實際數）及 AP／MAP；無 relevant 保留於 P@1／lift、
AP無定義不補零；空候選無分數且列數。程式分別處理這些退化情況。
主要配對變化：pc3_zero−release_original、pc3_zero−local_original、
center_only−local_original、pc3_zero−center_only；另列 median 與 cc 分支上述固定對比。
所有區間為 pointwise 描述性區間，無 multiplicity 調整或通過類別檢定的聲稱。

直接構圖診斷：在 pc3_zero 的三維 PC 空間，計算每個原 rank-1 鄰居與 query 的
平方距離，減去同 query 全候選平均平方距離，另列同標籤候選平均距離作比較。
這衡量原鄰居是否接近於所量測構圖，而不是構圖的因果效應或隨機排名檢定。
保存 PC explained variance、design rank、逐維 R²、殘差正交性及 changed-neighbor／hit 數。

## 重抽樣與不確定性

seed **2026102801**，NumPy default_rng，10,000次。依原524個 query universe
各來源的 observed eligible 播出日有放回抽同樣日數，同日全部 queries 一起進出；
來源順序 ami,pwn,tao,tay,trv,tsu（凍結計算順序，不是正文表序）。
保存每次抽到的 day indices 及每個 query 的共享權重，完整與 cc 使用相同權重。
每次重新加總有效分子／分母，pooled query-weighted，不等權平均來源或 pairs。
MAP 使用有效 AP 分母；失敗 draws 列數，不補零。95% percentile CI 用有效 draws。
paired change 從同一 draw 的兩個估計相減，不以區間重疊判斷。
rankings、候選、PCA／OLS 參數都固定；不在每次 bootstrap 重擬殘差化。
因此區間條件於觀察到的候選資源與轉換，不涵蓋新候選或重新估計構圖映射的變異。
原 CI 是原 seed 20261019 的全程 RNG stream；保留原數字，另列本次共享權重重抽區間，
不宣稱本次區間逐位重現原 RNG 狀態，也不混接不同版本點值與區間。
