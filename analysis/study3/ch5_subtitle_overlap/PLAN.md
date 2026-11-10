# Subtitle overlap in paired saved rankings



## Frozen population and saved rankings

- ULFR-v1-20261025, HA W3 `supplement_new|A|six_common|union_MI|W3|{xlmr_sub,tfidf_sub}|{source}`; all 2,753 common M-union-I queries and their original within-source union candidate banks, excluding self and same source-day. All original model labels remain separate.
- Primary rank1; one fixed descriptive sensitivity averages ranks 1 through min(3, number of candidates) within each query/route. No scanning ranks 4-10 for outcome summaries. Full rankings are read solely for identity, ordering, eligibility, tie-contract and top-10 validation.
- Read the verified 12-block ZIP directly without extraction or modifying the general formal reader. ZIP SHA: `b957bdb60b838346226b315466394ed83b0b41351ff905cdc372138006413209`. Formal condition-index SHA: `9dbe943dbfe6f972a8fc35cde35ea5edd811ad91d42a54ca4b8bf78c8ecb5889`.
- Verify every payload, the index/manifest contract, full/top-10 IDs and bitwise scores, both routes' query/bank order, full candidate permutations, W3 exclusions and stable order among equal saved float32 scores. Do not compute new scores or rank orders.
- Global IDs follow formal unit_rows and frozen feature IDs; cross-check the formal HA_W3 model data's saved query universe and hits. Read necessary metadata only, not embedding arrays.
- Assigned subtitles join exactly through `utt_id` and the existing `subtitle_text` field in frozen restore subtitle files. Their per-file hashes must match `corpus/study3/index_manifest.json`; record this upstream subtitle provenance separately from the formal numeric release. Never guess a match from wording. Missing/duplicate or conflicting mappings stop or are explicitly marked; no fallback to model-labeling OCR summaries, ASR, another time window or interface text.

## Text transformations and measures, fixed before scores

Original strings are never changed or copied into tracked outputs. All transformations occur only in new in-memory derived strings.

Primary normalization `NFC_WS`: Unicode NFC; replace each nonempty run of Unicode whitespace with a single ASCII space and strip boundary whitespace (`' '.join(s.split())`). Preserve letter case, punctuation, symbols and all other characters. Units are Unicode code points, not words, morphemes or grammatical units. Record Python and Unicode database versions.

Only normalization sensitivity `NFC_WS_noP`: apply NFC_WS, remove characters whose Unicode general category starts with `P`, then collapse/strip whitespace again. Do not remove symbols or lowercase. This examines punctuation dependence; it is not selected by whether it strengthens a result. Report both versions.

Primary measure: multiset character-bigram Dice, `2 * sum_g min(count_query(g), count_neighbor(g)) / ((len(query)-1)+(len(neighbor)-1))`, using adjacent code-point pairs including retained spaces. It is defined only if each normalized string has length at least two. No IDF, corpus fitting, segmenter or learned component.

Secondary: exact equality of the normalized strings, on the same paired eligible queries as Dice. It distinguishes identical strings from partial character sharing. No prefix-length or additional similarity metrics are added: they are unnecessary for this minimal question and would add length dependence and analyst choice.

Missing mapping, empty normalized string (length 0), and short nonempty string (length 1) are distinct states. Undefined overlap is null, never zero. An observed zero Dice value for two sufficiently long strings is valid. Repeated subtitles with different IDs remain distinct queries/candidates; no text deduplication changes the frozen estimand.

## Pairing, denominators, aggregation and support

- Route order is `[xlmr_sub, tfidf_sub]`; all direct differences are **TF-IDF minus XLM-R**. Query weighting is equal within each reported group, retaining empirical source/class/speech composition. No class/language reweighting.
- For each normalization and rank window, require the query and every required neighbor string to have a valid Dice definition in both routes. This shared query denominator is used for both route summaries, exact equality and paired differences. Top3 averages all available saved first min(3,n) scores, and is undefined if any required pair is invalid; it does not average only the valid neighbors.
- Report native valid counts by route, common valid n, exclusions/reasons, number of neighbors, and n/day/clip/source support. Top3 and the alternative normalization may have different denominators; report their sample change rather than attributing it wholly to a measurement change. Also report rank1 on the common rank1-and-top3 support, and primary/alternative normalization on their common support, to separate denominator changes.
- Complete groups: all plus each of Explaining, Reporting, Urging, Celebrating and Comforting; each crossed with speech all/M/I. Retain zero cells with n=0 and null summaries. M/I are existing machine speech groups, not independently verified speech identities. Report source-by-class-by-speech support; source subdivisions stay descriptive.
- For each route's Dice distribution, normalized query/neighbor length and paired Dice differences, report n, mean, SD (null when n<2), minimum, quartiles/median, maximum. Quantiles use linear interpolation at .25/.5/.75; do not call them confidence or credible intervals. Exact-match results include numerator/denominator and the paired mean difference.
- Category-match cross-summary: fixed rank1 original-label agreement for each route, four paired states (both match, XLM-R only, TF-IDF only, neither). Cross-tab the above all/class/speech groups with these states, using the same valid query denominator for each score configuration. These are selected descriptive subgroups, not mechanisms or independent replications. Also report the all-query raw state counts so eligibility attrition is visible. For top3, the state remains the rank1 state; exact and Dice values are top3 averages.
- Preserve all five categories and M/I support, including sparse/null combinations. No new inferential eligibility rule, category test, pooled-remainder relabeling, or formal significance claim. Queries sharing days/candidates are dependent; no independent-query inference is made.
