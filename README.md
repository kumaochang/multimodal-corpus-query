# From Concordance to Similarity

## Multimodal Corpus Query for the Austronesian Languages of Taiwan

Research materials accompanying Mao-Chang Ku’s doctoral dissertation at National Taiwan University.

## Research focus

The dissertation develops and evaluates multimodal query by example for the Austronesian languages of Taiwan. It examines how wording, sound, and visible bodily activity can organize comparisons beyond shared surface strings or annotation labels.

The retrieval investigation uses six television-news subcorpora, named for the designated languages of their source broadcasts: Amis, Atayal, Paiwan, Tao (Yami), Truku, and Tsou. Its clip-level and utterance-level comparisons are two stages of the same investigation. A separate Paiwan text-corpus study connects the retrieval problem to imperative forms and directive uses.

The central question is how different representations bring linguistic material together, and how those selections vary with activity category, source, and candidate range. Retrieval matching, prediction of human judgments, and the selection of individual cases provide different kinds of evidence.

## Analysis overview

| Analysis | Main comparison |
|---|---|
| Clip-level retrieval | Single representations and feature combinations, category-specific chance, recording conditions, and shot-composition diagnostics. |
| Paiwan text-corpus analysis | Imperative or hortative glosses compared with a rule over published English translations, including clause-coverage sensitivities. |
| Form and phone diagnostics | Surface-form ambiguity and phonetic detection, with precision and recall tied to their actual evaluation units. |
| Utterance-level retrieval | Representations compared across candidate ranges, common materials, text sources, and model-assigned activity categories. |
| Hierarchical models | Binary first-neighbor category matching, related source and activity contrasts, conditional and population-average predictions, and model diagnostics. |
| Directive judgments and retrieval cases | Human force judgments, text and non-text prediction, and the directives selected by form-based and example-based retrieval routes. |
| Acoustic and prosodic comparisons | Phone-interval statistics, acoustic-prosodic descriptions, and separate prosody and Whisper-encoder contributions to prediction. |
| Bodily activity and framing | Directed source-pair comparisons and the sensitivity of class matching and retrieved neighbors to representation and composition choices. |

Model-assigned activity categories, machine speech-language judgments, and human directive judgments remain distinct. Each comparison retains its own material, outcome, baseline, weighting, and uncertainty definition.

## Using the materials

The [materials and analysis guide](docs/materials/README.md) maps the saved results to their analysis programs, explains the required inputs and numerical environments, and gives the original command-line entry points. Start with its result-file tables to inspect the reported group results, or with its preparation and execution sections to work from the frozen inputs.

Programs retain their `analysis/`, `pipeline/`, and `scripts/` relationships. The two releases under [`materials/releases/`](materials/releases/) provide selected retrieval, hierarchical-model, and bodily-activity results; later comparisons have their own result files beside the corresponding analysis directories or under `data/study3_stats/`.

## Suggested citation

Ku, Mao-Chang. 2027. *From Concordance to Similarity: Multimodal Corpus Query for the Austronesian Languages of Taiwan*. National Taiwan University doctoral dissertation.
