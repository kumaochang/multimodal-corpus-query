# Materials and analysis guide

This guide connects the dissertation's saved results, source programs, input contracts, and command-line entry points. Paths below are relative to the repository root unless a link resolves them from this document. Run commands from the repository root. `python` denotes the interpreter for the relevant analysis environment.

## Finding the results

The clip-level and utterance-level analyses are successive stages of the same retrieval investigation. The Paiwan text-corpus analysis is a separate investigation of form-based access to directives. Within each stage, retain the result's candidate population, weighting, outcome, and interval definition when comparing it with another result.

### Formal utterance-level results

The [ULFR-v1 materials](../../materials/releases/ulfr-v1/) contain selected group results from **ULFR-v1-20261025**. Their [manifest](../../materials/releases/ulfr-v1/MANIFEST.json) identifies the saved files and analysis version.

| Result files under `materials/releases/ulfr-v1/` | What to read |
|---|---|
| `retrieval/primary.json` | Registered retrieval results, including the condition-specific outcomes and support. |
| `retrieval/sensitivity.json` | The corresponding saved sensitivity comparisons. |
| `retrieval/joint-interval-family.json` | The fixed contrast family and its joint intervals. |
| `models/base-contrasts.csv` | Hierarchical-model contrast summaries and prediction targets. |
| `models/selected-attempt-status.csv` | The selected fitting attempts and their diagnostic status. |
| `models/predictive-check-summary.csv` | Model-check summaries. |
| `models/day-refit-comparisons.csv` | The saved day-refit comparisons. |
| `applications/pair-class-retrieval.json` | Directed source-pair and pair-class bodily-movement retrieval. |
| `applications/activity-class-comparisons.json` | Activity-category comparisons for that application. |

The [focused two-way HA W3 materials](../../materials/releases/ha-w3-two-way-v1/) use the formal HA W3 model-data cache and adopted overlap weights, with a separate lower-order model specification. Its [manifest](../../materials/releases/ha-w3-two-way-v1/MANIFEST.json) and five `contrasts/*.json` files distinguish the base, prior-scale, day-weighted, and day-holdout comparisons. This is a different model specification on the shared formal inputs, rather than a replacement name for the full HA model.

### Clip-level and later comparisons

| Analysis | Saved results | Program or specification |
|---|---|---|
| Clip-level retrieval and 15 feature combinations | [`retrieval_results.json`](../../analysis/clip_level_map_audit/retrieval_results.json), [`fusion15_results.json`](../../analysis/clip_level_map_audit/fusion15_results.json), and [`fusion15_results.csv`](../../analysis/clip_level_map_audit/fusion15_results.csv) | [`complete_fusion_map.py`](../../analysis/clip_level_map_audit/complete_fusion_map.py) and [`PLAN_F0.md`](../../analysis/clip_level_map_audit/PLAN_F0.md) |
| Clip-level category baselines and composition diagnostics | Original programs `analysis/q1q2_composition.py`, `q3q4_reductions.py`, and `q5_baselines.py` through `q11_power.py` | The programs retain the individual diagnostic definitions. |
| Paiwan translation-rule comparison | The rule program prints its contingency table and statistics; the coverage-sensitivity program writes its report when `--output` is given. | [`study2_rebuild.py`](../../analysis/paiwan_texts/study2_rebuild.py) and [`study2_recall_check.py`](../../scripts/study2_recall_check.py) |
| Surface-form counts | The program prints word-final counts and writes `rule_count_summary.json` to the chosen output directory. | [`symbolic_surface_counts.py`](../../analysis/symbolic_surface_counts.py) |
| Phone detection | [`phone_detection_results.json`](../../analysis/phone_detection_results.json) | [`q21_speaker_recall_audit.py`](../../analysis/q21_speaker_recall_audit.py) |
| Subtitle overlap in saved HA W3 retrieval | `analysis/study3/ch5_subtitle_overlap/SUMMARY.json`; reader and derived caches under `data/study3_stats/ch5_subtitle_overlap/` | [`ch5_subtitle_overlap/pipeline.py`](../../analysis/study3/ch5_subtitle_overlap/pipeline.py) |
| Utterance-level feature combinations | [`RESULTS.json`](../../data/study3_stats/ch5_fusion/fusion_local/RESULTS.json); `FUSION_WIDE.csv`, `FUSION_RESULTS.csv`, `FIXED_CONTRASTS.csv`, and `STAGE_COMPARISON.csv` in [`ch5_fusion_preparation/`](../../analysis/study3/ch5_fusion_preparation/) | [`run_adopted.py`](../../analysis/study3/ch5_fusion_preparation/run_adopted.py), [`PLAN.md`](../../analysis/study3/ch5_fusion_preparation/PLAN.md), and `INVENTORY.json` |
| Judgment prediction and retest | [`REGISTERED_RESULTS_AUDIT.json`](../../analysis/study3/clm_jev_preparation/REGISTERED_RESULTS_AUDIT.json) and [`REGISTERED_RESULTS_COVERAGE.json`](../../analysis/study3/clm_jev_preparation/REGISTERED_RESULTS_COVERAGE.json) | Adopted [`timing_exception_entry.py`](../../analysis/study3/clm_jev_preparation/timing_exception_entry.py), retained [`registered_entry.py`](../../analysis/study3/clm_jev_preparation/registered_entry.py), `evaluation.py`, and `contracts.py` |
| Directive cases and form-based comparison | [`directive_completion/RESULTS.json`](../../analysis/study3/directive_completion/RESULTS.json) | [`derive.py`](../../analysis/study3/directive_completion/derive.py) and [`symbolic_similarity/PLAN.md`](../../analysis/study3/clm_jev_preparation/symbolic_similarity/PLAN.md) |
| Finite-bank random-ranking and reference context | [`retrieval_reference_context/RESULTS.json`](../../analysis/study3/retrieval_reference_context/RESULTS.json) | [`reference_and_rank_context.py`](../../analysis/study3/retrieval_reference_context/reference_and_rank_context.py) |
| Prosody and Whisper augmentation | [`RESULTS.json`](../../data/study3_stats/prosody_exploration/P1_augmented_local/RESULTS.json), [`P1_MODELS.csv`](../../analysis/study3/prosody_preparation/P1_MODELS.csv), and [`P1_CONTRASTS.csv`](../../analysis/study3/prosody_preparation/P1_CONTRASTS.csv) | [`run_augmented.py`](../../analysis/study3/prosody_preparation/run_augmented.py) and [`PLAN.md`](../../analysis/study3/prosody_preparation/PLAN.md) |
| Fifteen-variable prosodic description | [`P2_DESCRIPTIONS.csv`](../../analysis/study3/prosody_preparation/P2_DESCRIPTIONS.csv) | [`prosody_preparation/runner.py`](../../analysis/study3/prosody_preparation/runner.py), task `p2` |
| X/Explaining framing sensitivity | [`SUMMARY.json`](../../analysis/study3/ch5_framing_sensitivity/SUMMARY.json) and [`RESULTS.csv`](../../analysis/study3/ch5_framing_sensitivity/RESULTS.csv) | [`run_sensitivity.py`](../../analysis/study3/ch5_framing_sensitivity/run_sensitivity.py) and [`PLAN.md`](../../analysis/study3/ch5_framing_sensitivity/PLAN.md) |

CSV files can be read directly in a spreadsheet or text editor. JSON files preserve nested conditions and named comparisons. For example, this standard-library command lists the top-level fields of the saved primary retrieval results without running an analysis:

```sh
python -c "import json; from pathlib import Path; x=json.loads(Path('materials/releases/ulfr-v1/retrieval/primary.json').read_text(encoding='utf-8')); print(list(x))"
```

## Preparing inputs

Use the original directory relationships rather than moving individual programs into a common script folder. The main roles are:

| Location | Role |
|---|---|
| `analysis/` | Study-specific runners, fixed comparison definitions, and saved group summaries. |
| `pipeline/analysis/` | Retrieval, eligibility, activity labeling, language judgment, and feature-analysis implementations used by the runners. |
| `pipeline/stages/` | Source-layer processing and feature preparation, including subtitles, transcription, utterance boundaries, phones, prosody, and shots. |
| `scripts/` | Shared model/data preparation, posterior reporting, Stan sources, sampling, and the Paiwan coverage-sensitivity program. |
| `data/study3_stats/` | Analysis-specific numerical products. The formal numerical root is `formal_release/v1/`; later fusion and prosody results have separate locations. |
| `materials/releases/` | The selected formal and focused-model results described above. |

### Frozen utterance-level inputs

The formal ranking entry uses `data/study3_runs/`. Its principal files are `features_full/features.npz`, `features_full/tfidf_sub.npz`, `features_full/tfidf_tr.npz`, `label_analysis/units.tsv`, and `unit_table/units_k.npz`. The exact source identities are recorded in [`RELEASE_MANIFEST.json`](../../analysis/study3/formal_release/RELEASE_MANIFEST.json) and checked by [`fr_common.py`](../../analysis/study3/formal_release/fr_common.py). Feature rows and unit metadata must refer to the same units in the stored order.

The bodily-movement programs also read the existing aligned trees under `work/data/aligned_trees/` and pose records under `work/data/pose/`. The registered ranking runner calls the original four programs with their own options; it does not infer new representations from the saved group summaries.

The formal model stage takes its eleven model-data caches from `data/study3_stats/formal_release/v1/supplement/runA/execution/`, with their prefit reports in `execution_reports/`. The focused two-way runner gives each input an explicit role in [`SOURCE_CONTRACT.json`](../../analysis/study3/ch5_ha_w3_two_way/SOURCE_CONTRACT.json): formal model data, overlap weights, prefit, base contrasts, feature-row identities, and unit metadata. Keep the contract's byte counts and hashes with the corresponding files.

The [`ULFR reader path map`](../../analysis/study3/ulfr_local/LOCAL_PATH_MAP.json) relates `numeric_core`, `writer_metadata`, and `frozen_restore_runs` to the formal results, their metadata, and the frozen input tree. It works with the numerical manifest, condition index, unit-row index, and [`METADATA_MANIFEST.json`](../../analysis/study3/ulfr_local/METADATA_MANIFEST.json). Its [`reader`](../../analysis/study3/ulfr_local/ulfr_reader.py) checks release and file identities before opening requested numerical blocks. The framing inputs below also use its saved full-ranking block index.

### Inputs for the other comparisons

- **Clip-level analysis:** the 30-clip row identities and labels must agree across the vector caches, saved retrieval results, and clip metadata. The fusion runner reads `analysis/_emb_cache.npz` for text, MFCC audio, and phone-interval statistics, and `analysis/_q8_cache.npz` for kinematics. Its `INPUTS` list also specifies the original programs and reference files checked during execution. The four blocks have 768, 86, 8, and 18 dimensions, respectively. The historical key `full_fusion_whisper` identifies the saved configuration; its audio block is MFCC, not a Whisper vector.
- **Paiwan text rules:** supply the published XML tree to `study2_rebuild.py`. The coverage-sensitivity program expects that tree at `data/paiwan_texts/xml/`. Both use the published English translation and imperative/hortative glosses; the translation rule and the morphological inventory are separate inputs to the comparison.
- **Surface-form counts:** `symbolic_surface_counts.py` reads the XML files directly inside its first positional directory, using word-level original `PHON`, morpheme `FORM`/glosses, and sentence translations. Its second positional directory must already exist. Counts of target word tokens and translation-rule matches remain separate from unique-sentence counts.
- **Phone-detection support:** prepare `data/ntu_paiwan_asr/clips.json`, `gold_index.json`, the corresponding XML files under `data/ntu_paiwan_asr/xml/`, and saved `data/qc/manifest.json` and `phone_qc.jsonl`. The required `--recording-map` is a JSON object mapping each recording identifier used in the clip metadata's `src` field to its XML basename. The program derives speaker grouping from that XML metadata. Its count references are [`phone_detection_reference.json`](../../analysis/phone_detection_reference.json) and `pipeline/qc/results/armA_q21_reproduction.json`; the source files checked for provenance are named in its `run()` input list. It also records the Git checkout identity.
- **Subtitle overlap:** supply the formal numerical manifest, `index/condition_index.json`, `index/unit_rows.npz`, `supplement/runA/execution/HA_W3_model_data.npz`, and the saved top-ten views named by the condition index. The twelve complete ranking blocks are read from `data/study3_stats/ch5_subtitle_overlap/HA_W3_subtitle_fullblocks_ULFR-v1.zip`. Exact assigned subtitle strings come from `data/study3_runs/subtitles/<clip>.subtitles.json`, checked against `data/study3_runs/index_manifest.json` and the frozen feature-row identities. The existing plan and reader/plan-freeze records accompany these inputs.
- **Utterance-level fusion:** follow [`INVENTORY.json`](../../analysis/study3/ch5_fusion_preparation/INVENTORY.json) for the frozen features and metadata. `--baseline` names the saved same-recipe one-block controls, including `MANIFEST.json`, `FREEZE.json`, `RESULTS.json`, and `cache.npz`. The adopted runner checks those controls before computing combinations on the fixed common population. The controls are a separate result lineage from the formal release's original single-route numbers.
- **Judgment prediction:** the execution receipt binds the frozen item5 text, audio, kinematic, and face inputs, CLM/text metadata, sampling, the three judgment batches, and completion evidence. [`prepare_timing_request.py`](../../analysis/study3/clm_jev_preparation/prepare_timing_request.py) prepares the adopted execution request; [`contracts.py`](../../analysis/study3/clm_jev_preparation/contracts.py) and [`clm_inputs.py`](../../analysis/study3/clm_jev_preparation/clm_inputs.py) define the input checks. Use the matching entry and receipt rather than calling `pipeline/analysis/item5/probe.py` as an alternative execution route.
- **Prosodic description and augmentation:** [`prosody_preparation/INVENTORY.json`](../../analysis/study3/prosody_preparation/INVENTORY.json) binds the scalar features, design metadata, and saved prediction inputs. The augmentation baseline is the saved local text model, including its out-of-fold predictions, fold transformations, manifest, and freeze. It is the paired reference for adding prosody or Whisper features.
- **Framing sensitivity:** use the ULFR path map and the fixed X/Explaining query/candidate bank, frozen kinematic data, and original ranking blocks specified by the framing plan. The map's full-ranking overlay locates the saved blocks under `data/study3_stats/formal_release/v1_full_rankings/rankings/blocks/`; `FULL_RANKING_MANIFEST.json`, `FULL_RANKING_PACKAGES.json`, `FULL_RANKING_VERIFICATION.json`, and `full_ranking_list.tsv` under `analysis/study3/ulfr_local/` identify those blocks and their relationship to the formal condition index. Its full-population and complete-case comparisons retain different denominators.

## Numerical environments

The programs retain the numerical profiles of their analyses. Select the corresponding interpreter before using an execution command.

| Analysis | Recorded environment or explicit program check |
|---|---|
| Paiwan coverage sensitivity | spaCy 3.8.16 and `en_core_web_sm` 3.8.0, with lxml, NumPy, and SciPy. |
| Formal ranking and registered judgment prediction | Python 3.10.22, NumPy 1.26.4, SciPy 1.13.1; the judgment entry additionally checks scikit-learn 1.5.2. The formal ranking manifest records the BLAS libraries and one-thread settings. |
| Formal posterior and focused two-way model | Python 3.12.0, NumPy 2.2.6, SciPy 1.15.3, CmdStanPy 1.3.0, CmdStan 2.37.0, and ArviZ 0.22.0 for posterior reporting. The formal specification records the compiler and sampling settings. |
| Adopted utterance fusion and prosody | Python 3.12.3, NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.6.1, and MKL 2023.1, with one numeric thread; also psutil and threadpoolctl. The shared `execution_support.py` checks this profile. |

For formal ranking and posterior work, set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1`. [`fr_common.py`](../../analysis/study3/formal_release/fr_common.py) accepts `RANKING_PYTHON` and `POSTERIOR_PYTHON` to identify the two interpreters. Its CmdStan path is `data/study3_stats/ch5_reproducibility/toolchain/cmdstan-2.37.0`. The focused two-way runner takes `--cmdstan` explicitly.

The original clip-level programs use NumPy, SciPy, scikit-learn, and threadpoolctl. Their vector/cache definitions and [`PLAN_F0.md`](../../analysis/clip_level_map_audit/PLAN_F0.md) identify the saved numerical lineage. The formal ranking environment and later local fusion environment serve different executions; they are not interchangeable profiles for asserting bitwise equality.

## Original execution entry points

Use these entry points with the prepared inputs and corresponding interpreter. Paths beginning with `/path/to/` stand for existing material locations. Choose new output locations where the program requires them; retain saved results as references.

### 1. Clip-level retrieval and feature combinations

After placing the files required by `complete_fusion_map.py` at their original paths, use a new output directory and a new cache path inside this repository:

```sh
python analysis/clip_level_map_audit/complete_fusion_map.py --output output/clip-fusion --cache output/clip-fusion-rankings.npz
```

The program preserves the 30-query, 29-other-candidate setting and calculates the 15 unweighted feature combinations. Its current AP excludes the query itself; fields naming historical MAP belong to the earlier calculation and remain separately identified.

### 2. Paiwan translation-rule comparison

The first argument is the XML directory; the optional second argument selects `sent`, `whole`, or `quotes`. `sent` is the primary rule.

```sh
python analysis/paiwan_texts/study2_rebuild.py data/paiwan_texts/xml sent
python scripts/study2_recall_check.py --output output/paiwan-coverage.md
```

The second program evaluates the existing rule and its clause-coverage variants, including `voc2` and `voc12`, against the same published material.

#### Surface-form counts

```sh
python analysis/symbolic_surface_counts.py data/paiwan_texts/xml output/surface-counts
```

Create `output/surface-counts` before execution. This standard-library program reads `*.xml` directly in the supplied directory and writes `rule_count_summary.json`. It counts word-final sequences and the stated translation/gloss-rule matches; the latter are target-word counts.

#### Phone-detection support

```sh
python analysis/q21_speaker_recall_audit.py --repo-root . --recording-map /path/to/recording-map.json --out-prefix output/phone-detection
```

The recording map connects saved recording IDs to their source XML files. The standard-library program joins the saved clip, gold, and phone records, pools confusion counts before forming speaker-level ratios, and writes `output/phone-detection.json`. Its per-clip derived cache uses the fixed `output/q21_speaker_recall_audit/clip_counts.json` path. It compares unwindowed detection with the first-25, first-15, and first-10-phone variants using the saved recognizer output.

### 3. Formal utterance-level rankings

```sh
python analysis/study3/formal_release/fr_rank_registered.py --run A
python analysis/study3/formal_release/fr_rank_supplement.py --run A
```

The first command executes the registered retrieval, pair-class, category, and sensitivity programs with recorded ranking outputs. The second supplies the supplementary comparisons and model-data preparation. Run B is the independent repeat mode; `fr_rank_registered.py --verbatim` invokes the original programs for their reference comparison. Output roots are defined by `fr_common.py`, so prepare a separate execution tree when retaining an existing formal numerical release.

#### Subtitle overlap on saved HA W3 rankings

In a new execution tree, use the following stages:

```sh
python analysis/study3/ch5_subtitle_overlap/pipeline.py reader
python analysis/study3/ch5_subtitle_overlap/pipeline.py freeze
python analysis/study3/ch5_subtitle_overlap/pipeline.py analyze
```

These stages read and validate the saved XLM-R and TF-IDF subtitle rankings, bind the plan and inputs, then summarize character-bigram Dice overlap and exact-string matches for rank one and the top three. The two string normalizations retain punctuation or remove Unicode punctuation after NFC and whitespace normalization. The reader writes `reader_cache.npz` in `data/study3_stats/ch5_subtitle_overlap/`; analysis writes `SUMMARY.json` beside the program and a derived cache in the data directory. With the existing reader cache, plan freeze, and saved summary in place, the `reproduce` command writes `summary_reproduced.json` in the data directory and compares it with the saved summary. The program reads saved ranks rather than generating new retrieval rankings.

### 4. Full hierarchical models

Use the posterior interpreter for these stages:

```sh
python analysis/study3/formal_release/fr_posterior.py validate
python analysis/study3/formal_release/fr_posterior.py fit --model H0_W1 --variant base --attempt 1
python analysis/study3/formal_release/fr_posterior.py collect
```

`validate` compiles and checks the implementations. `fit` runs one four-chain fit and its reporting. `queue --jobs 7` is the original scheduled execution mode for the model families, sensitivities, and benchmarks. The model-data inputs, repair rules, and reporting targets are defined in [`FORMAL_RELEASE_SPEC.md`](../../analysis/study3/formal_release/FORMAL_RELEASE_SPEC.md).

### 5. Focused two-way HA W3 model

The formal and frozen roots below are the roots named in `SOURCE_CONTRACT.json`:

```sh
python analysis/study3/ch5_ha_w3_two_way/runner.py preflight --formal-root data/study3_stats/formal_release/v1 --frozen-root data/study3_runs
python analysis/study3/ch5_ha_w3_two_way/runner.py environment --cmdstan /path/to/cmdstan-2.37.0
python analysis/study3/ch5_ha_w3_two_way/runner.py run --formal-root data/study3_stats/formal_release/v1 --frozen-root data/study3_runs --out-root output/ha-w3-two-way --cmdstan /path/to/cmdstan-2.37.0 --execute-server --job base --attempt 1 --reason reproduction
```

The five jobs are `base`, `prior_half`, `prior_double`, `day_weighted`, and `day_holdout`. The run stage requires the recorded Linux execution profile and keeps each attempt separate. The existing code freeze, column mapping, and day plan specify the model and its fixed comparisons.

### 6. Utterance-level feature combinations

```sh
python analysis/study3/ch5_fusion_preparation/run_adopted.py --restore data/study3_runs --baseline /path/to/saved-single-block-controls --out output/utterance-fusion
```

The fixed 384-unit common population, one-block controls, feature normalization, and combination comparisons are specified in the fusion plan and inventory. The runner requires a new output directory.

### 7. Judgment prediction and retest

```sh
python analysis/study3/clm_jev_preparation/registered_entry.py --check-environment
python analysis/study3/clm_jev_preparation/prepare_timing_request.py --bindings /path/to/input-bindings.json --answers-root /path/to/three-batches --evidence-dir analysis/study3/clm_jev_preparation/work/reproduction-evidence --result-dir analysis/study3/clm_jev_preparation/work/reproduction-results
python analysis/study3/clm_jev_preparation/timing_exception_entry.py --receipt analysis/study3/clm_jev_preparation/work/reproduction-evidence/registered_request.json --execute-with-timing-exception
```

The preparation command takes the original input bindings and three recorded batches, and writes `registered_request.json` in a new evidence directory. The execution command uses that receipt. The adopted procedure retained the 40 repeated judgments and their timestamps, with the author's removal of the minimum seven-day retest interval. The original `registered_entry.py` remains a separate source entry. Evidence and results use separate new directories under the existing `work/` root. The published group-result files can be read directly using the result map above.

### 8. Prosody and Whisper augmentation

```sh
python analysis/study3/prosody_preparation/run_augmented.py --restore data/study3_runs --baseline /path/to/saved-local-text-controls --out output/prosody-augmentation
```

The paired reference is the saved local text model. The two additions are acoustic prosody and Whisper-encoder features, respectively; the fixed six-coordinate interval family is specified in the prosody plan.

### 9. Prosodic description

```sh
python analysis/study3/prosody_preparation/runner.py p2 --restore data/study3_runs --out output/prosody-description
```

This task summarizes the fifteen existing scalar features with shared whole-day weighting. Its inventory identifies the frozen inputs and design metadata.

### 10. X/Explaining framing sensitivity

```sh
python analysis/study3/ch5_framing_sensitivity/run_sensitivity.py --root . --path-map analysis/study3/ulfr_local/LOCAL_PATH_MAP.json --out output/framing-sensitivity
```

The plan separates the original representation, centering control, zero- and median-imputed residualization, and complete-case branch. Query/candidate membership, PCA/OLS transformations, and rankings are fixed within its shared day-weight bootstrap. Read the paired changes together with the rank-change and matching-count summaries.

## Reading comparisons correctly

- **Retrieval metrics:** P@1 measures the first returned match, P@3 the top-three proportion under the stated denominator, and MAP the average of defined query AP values. Candidate exclusion, relevance, and undefined-query handling belong to each analysis's contract.
- **Units and labels:** the six subcorpora identify source broadcasts. Machine speech-language judgments and model-assigned activity categories are distinct from human directive judgments. Phone-detector recall uses gold-positive clips as its denominator, not the number of target tokens.
- **Representation definitions:** clip-level phone-interval statistics and utterance-level acoustic prosody measure different properties. The clip-level six-configuration Whisper comparison uses mean plus SD, with mean-only as a separate ablation; the utterance-level Whisper representation uses mean-only.
- **Intervals:** pointwise day-cluster bootstrap intervals, the fixed simultaneous contrast family, and posterior credible intervals have different targets. Use the saved paired contrasts rather than subtracting interval endpoints or judging differences by overlap.
- **Hierarchical predictions:** the response is binary first-neighbor activity-category matching. Conditional predictions retain fitted effects; population-average predictions integrate the specified new effects while retaining the adopted empirical composition. They do not jointly model P@3 and MAP or create a balanced population.
- **Framing and aggregation:** the X/Explaining sensitivity is query-weighted on its fixed bank. The bodily-movement application's ordered-pair and pair-class summaries use their own population and aggregation. Similar category-match summaries can accompany substantial changes in the returned neighbors.
- **Directive retrieval:** finite-bank comparisons describe the directives and cases reached under the specified budgets. Category matching, judgment prediction, and case membership answer different questions about the query layer.
