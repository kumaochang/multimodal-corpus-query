# Utterance-level representation fusion

## Question and exposure

The comparison asks how combining fixed feature blocks changes same-model-class retrieval on an identical utterance bank, and how its pattern relates to the earlier clip-level combinations. The fixed configurations were defined after the single-representation results were known.

## Frozen population and sources

`INVENTORY.json` pins input SHA-256 values, original program, formal release manifest and numeric index. The release is ULFR-v1-20261025. Inputs are resolved relative to the repository and verified before loading.

The common/kept universe contains 384 utterances, 63 source-days and 109 clips. The original feature-row order is retained for queries and candidates, with ordered-ID SHA-256 `c075770fb9c7abed22435cdb3611047f7ee4082b241d38ee8be9eb55495eb0aa` (UTF-8 IDs joined by newline, including final newline). Source/date sorting controls resampling only and never silently reorders candidate ties.

| Source | Queries | Days | Clips |
|---|---:|---:|---:|
| Amis | 63 | 11 | 17 |
| Atayal | 90 | 10 | 19 |
| Paiwan | 50 | 11 | 21 |
| Tao | 42 | 9 | 14 |
| Truku | 53 | 11 | 18 |
| Tsou | 86 | 11 | 20 |

Common eligibility reproduces nonzero subtitle XLM-R plus within-shot and `k_missing == 0`, then kept. This exactly matches the saved common eligibility mask. TF-IDF is finite on all selected rows, with 14,347 nonzero entries and zero zero-vector rows. No new eligibility selection is needed. The visibility/shot-selected common universe is the target population, rather than all 14,012 utterances.

W1 uses other same-source utterances. W2 additionally excludes the query clip. W3 additionally excludes the query broadcast day. X uses other sources. Candidate counts are respectively 41–89, 35–89, 29–89 and 294–342. Every configuration has the same candidate mask within a pool. Model classes remain frozen, and pooled remainder never replaces the original relevance labels.

## Configurations and numerical recipe

The 35 configuration lists are enumerated in `INVENTORY.json`. Choices are T absent/subtitle XLM-R/subtitle TF-IDF, A absent/Whisper/MFCC86, P absent/present and K absent/present, excluding all absent. Primary presentation uses XLM-R/Whisper/P/K. All four method sets and pools are reported, with duplicate configurations computed only once.

Retain original preprocessing from pinned `pipeline/analysis/ch5_retrieval.py`. MFCC86 is globally standardized on all 14,012 rows in float64 before row normalization. Prosody15 uses within-source mean imputation and standardization on the full original source rows, before selecting common/kept. Whisper and subtitle XLM-R use their frozen vectors. K is the frozen 14-dimensional representation. TF-IDF retains its fitted vocabulary, IDF and sparse float64 payload. No new fitting of TF-IDF or feature extraction occurs.

The original `l2rows` casts to float32, including its sparse branch. Preserve that exact source function through a side-effect-free adapter. Normalize each included block, concatenate in T,A,P,K order, then normalize the concatenation and compute cosine. A zero block contributes zeros; the final zero vector has cosine zero. Do not replace this recipe with an assumed arithmetic average of channel cosines. Sparse and dense reductions can differ numerically.

For TF-IDF, normalize through the original sparse branch, then densify only the selected 384 rows for concatenation. Never densify all 14,012 rows. Preserve float32 C layout and one BLAS thread. Full 384-by-384 scores can be reused for pool masks. Stable descending sort retains original candidate-row order for exact ties. Save native score dtype and full rankings, not only top-k.

The original native single-route path and the final-normalized one-block fusion path are retained as distinct numerical recipes. Full saved one-block score matrices, all four pool rankings, per-query metrics and masks define the adopted local paired controls. BASELINE_VERSIONS.json binds the unchanged original control archive and its original freeze separately from the portable public program identities. Every control must reproduce before the remaining configurations are computed; the numerical recipe is not chosen by effect direction.

## Metrics and estimands

For each of 35 configurations and four pools, report query-weighted P@1, P@3 and MAP. P@1 is mean hit. P@3 divides by min(3, available candidates). AP uses the complete self-excluded ranking and original relevant class. A query with zero relevant candidates contributes to P@1/P@3 and chance but has undefined AP. Empty candidate queries have no score. Keep counts of both cases and separate effective denominators.

P@1/P@3 chance is the query-specific relevant share in its candidate mask. Lift@1 is hit minus that chance. For identical banks the P@1 and lift differences must agree. MAP has no class-share null claim. Report all queries, Explaining and pooled remainder with coverage status, with separate local class/source summaries. The inferential family below concerns pooled all-query contrasts only.

## Shared day resampling and intervals

Use PCG64 seed 2026110501 and 10,000 draws. Iterate sources in Amis/Atayal/Paiwan/Tao/Truku/Tsou order (`ami,tay,pwn,tao,trv,tsu`), dates lexicographically within source. For each source, call `rng.integers(0, D, size=(10000,D))` once and convert selections to multiplicities. Save those integer selections and the ordered source-date key table. The same multiplicities apply to every query, configuration, pool and metric. Recompute pooled numerators and metric-specific valid denominators. This conditions on fixed candidates and rankings.

All 420 pooled configuration/pool/metric coordinates receive pointwise percentile 95% intervals, using finite draws and `numpy.quantile(..., method='linear')` at .025 and .975, plus undefined draw counts. Query-group summaries use these same weights with their own denominators.

The adopted main family has 12 coordinates in contrast-major order: T+A minus T, T+P minus T, T+K minus T, T+A+P+K minus T, each ordered P@1/P@3/MAP, using XLM-R/Whisper and W3. Calculate paired differences within each draw. For joint-finite draws let s_j be SD with ddof=1, and c the .95 linear quantile of max_j abs((delta_bj - delta_j)/s_j). The simultaneous band is delta_j ± c*s_j. At least 9,900 joint-finite draws, finite observed estimates and all positive s_j are required. Otherwise suppress the whole family's simultaneous band while retaining all coordinates, pointwise intervals and degeneration records. Never drop a coordinate and recalibrate. No new p-value scan or claim of coverage over other exploratory families.
