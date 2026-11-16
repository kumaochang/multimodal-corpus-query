# From Concordance to Similarity

## Multimodal Corpus Query for the Austronesian Languages of Taiwan

## Research focus

This project develops and evaluates multimodal query by example for the Austronesian languages of Taiwan. It examines how wording, sound, and visible bodily activity can organize comparisons beyond shared surface strings or annotation labels.

The utterance-level corpus comprises six television-news subcorpora, named for the designated languages of their source broadcasts: Amis, Atayal, Paiwan, Tao (Yami), Truku, and Tsou. The Paiwan text-corpus analysis examines how imperative and hortative glosses correspond to directive classifications based on published English translations.

## Analysis overview

| Analysis | Main comparison |
|---|---|
| Clip-level retrieval | Single representations and feature combinations, category-specific chance, recording conditions, and shot-composition diagnostics. |
| Paiwan text-corpus analysis | Imperative or hortative glosses compared with a rule over published English translations, including clause-coverage sensitivities, rule-trigger decomposition, and source-gloss profiles. |
| Form and phone diagnostics | Surface-form ambiguity and phonetic detection, with precision and recall tied to their actual evaluation units, plus spelling coverage of saved ASR output. |
| Utterance-level retrieval | Representations compared across candidate ranges, common materials, text sources, and model-assigned activity categories. |
| Hierarchical models | Binary first-neighbor category matching, related source and activity contrasts, conditional and population-average predictions, and model diagnostics. |
| Directive judgments and retrieval cases | Human force judgments, text and non-text prediction, and the directives selected by form-based and example-based retrieval routes. |
| Acoustic and prosodic comparisons | Phone-interval statistics, acoustic-prosodic descriptions, separate prosody and Whisper-encoder prediction comparisons, and diagnostics of saved scores within and between days. |
| Bodily activity and framing | Directed source-pair comparisons and the sensitivity of class matching and retrieved neighbors to representation and composition choices. |

Activity-category retrieval uses model-assigned labels as its reference. Machine speech-language judgments define filtering and comparison groups. The Paiwan news application evaluates predictions and retrieved cases against a consultant’s directive judgments.

## Using the materials

The [materials and analysis guide](docs/materials/README.md) lists the analysis programs and saved results, documents required inputs and software environments, and gives commands for running the analyses.

The [registration version index](docs/registration/index.json) links the original protocol and amendment records. The [file-availability catalogue](docs/materials/file-availability.csv) identifies supplied files and the inputs required for reproduction.

Analysis programs are organized under `analysis/`, `pipeline/`, and `scripts/`. Selected retrieval, hierarchical-model, and bodily-activity results are available under [`materials/releases/`](materials/releases/). Additional saved results are stored with their analysis programs or under `data/study3_stats/`.
