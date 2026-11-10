# Pre-registration — Study 3

**Status: IN FORCE.**
Drafted 2026-10-03. The seven open values listed in §0.3 were confirmed by the author on
2026-10-03, unchanged from the drafted proposals. The file takes effect at the commit that
carries this status line, and that commit precedes any inspection of the Study 3 corpus.
**This file is not edited from here on**; changes go in §7 as dated amendments.

---

## 0. Scope

### 0.1 What this covers

The **Study 3 corpus**: TITV《看見族語》television news, six languages (Amis, Paiwan, Truku,
Atayal, Tsou, Yami), three clips sampled at random from each of twelve broadcast days per
language, 36 clips per language, **216 clips in total**, spread across months. None of this
material has been collected at the time of drafting.

It does not cover Chapters 1 to 3, which depend on no new data, and it does not cover Study 1 or
Study 2, which are complete.

### 0.2 What has already been seen, stated so that nothing here is mistaken for a blind design

The author has worked extensively with **Study 1's 30 clips** (Amis, Paiwan, Truku, ten each)
and has run four families of representation on them. That is precisely why §2 item 6 names the
primary kinematic variant in advance and why **no outcome test in Study 3 is run on Study 1's
material**.

The detector operating point in §4 item 10 was chosen on the **NTU Paiwan ASR corpus**, which is
independent of the Study 3 corpus, and is fixed here before any Study 3 data are touched.

The measurements that preceded this document are listed in the appendix.

### 0.3 The seven values settled before this file took effect

| # | Item | Confirmed 2026-10-03 |
|---|---|---|
| 1 | §1 item 2 | `default_rng(20261005)`, 3 of 10 without replacement, fixed iteration order |
| 2 | §1 item 3 | tested on its own only at **≥15 broadcast days and ≥25 clips** |
| 3 | §1 item 4 | no label-driven top-up; up to **4 extra whole broadcast days** per language |
| 4 | §3 item 8 | **the utterance**, on the principle that the label unit follows the retrieval unit |
| 5 | §3 item 9 | music proportion (Celebrating higher), speaker-change rate (Reporting lower) |
| 6 | §4 item 11 | **25 tokens**, stratified by target and by Urging / not, fixed seed, drawn first |
| 7 | §2 item 5 | monotone non-increasing Short ≥ Middle ≥ Long; **primary test Middle vs Long**, Short descriptive only |

---

## 1. Design and inference

**1. The bootstrap cluster is the broadcast day.** Anchor, studio, lighting and news cycle are
day-level properties; clips within a day and utterances within a clip are not independent.
Permutation tests permute at day level too. Day count, clip count and utterance count are all
reported.

**2. The random-selection rule** for the three clips drawn from each broadcast day:
`numpy.random.default_rng(20261005)`, drawing 3 of the day's 10 clips without replacement,
iterating languages alphabetically and days in date order, so the draw is reproducible from the
seed alone. A broadcast day that yields fewer than 10 clips is sampled 3 of however many exist;
a day yielding fewer than 3 contributes all of them and still counts as one cluster. Both cases
are recorded.

**3. The primary contrast is Explaining against the pooled remainder.** A single category is
tested on its own only if it occurs in **at least 15 broadcast days and at least 25 clips**;
otherwise it is reported descriptively. The threshold is stated in days as well as clips because
the broadcast day is the inference unit, and a category present in only a handful of days has
too few clusters for cluster-robust inference to behave.
Projecting Study 1's label distribution onto 216 clips gives roughly Explaining 101, Urging 58,
Celebrating 36, Reporting 22, Comforting 0, so the threshold is expected to bind on Reporting
and Comforting. That projection comes from three languages and one broadcast day each and may
be wrong; the threshold is fixed here regardless. The primary contrast does not depend on it.
This rule is fixed before the label distribution is seen. Category quotas are **not** fixed,
because the category is unknowable when the video is collected.

**4. The top-up procedure** should a category fall short. **Topping up by searching for the
scarce category is not permitted**, because selecting material by label breaks the unstratified
design. Instead, up to **four additional whole broadcast days per language** may be drawn at
random from the same sampling frame by the rule in item 2, continuing the seeded sequence. Any
day drawn this way **enters every analysis**, not only the one that motivated it. If the
category still falls short, it is reported descriptively.

---

## 2. The gesture section (§6.2)

**5. The genetic-distance gradient prediction and its test.** Cross-lingual retrieval lift is
predicted to be **monotone non-increasing across Short, Middle, Long** as those classes are
defined in `glossary.md`: Short is within Atayalic (Atayal-Truku), Middle is across Formosan
primary branches, Long is Formosan against Malayo-Polynesian, that is Yami against the other
five.

**The primary test is Middle against Long**, 9 pairs against 5. **Short is reported
descriptively and is not tested**, because it contains **exactly one language pair** and a class
of one cannot support a test. This is a property of the six-language design, not of the result,
and it is stated here so that it is not discovered afterwards.

**6. The kinematic variant list, with the primary named in advance.** Four representation
families have already been tried on Study 1's material, so this is where forking paths are
worst. The outcome test runs on the Study 3 corpus only; a further result on Study 1's material
would not be evidence.

**7. Whether the middle distance class is subdivided** once Tsou data are in is declared here as
an **alternative analysis**, not decided after seeing results. Both the undivided and the
subdivided analyses are reported if it is run.

---

## 3. The labels

No new bulk human annotation is done for Study 3. The labels are model output. What follows is
fixed in advance so that none of it is selected afterwards.

**8. The labeling unit, and the dual-unit agreement rate.** The same labeling model is run at
**both** the clip and the finer unit, and the rate at which the finer unit's label differs from
the label it would inherit is reported. This measures model self-consistency, not correctness,
but for the question being asked, how often a clip label misdescribes the units beneath it,
self-consistency is the right quantity. The same measurement supplies the label-inheritance rate
that §6.2's shot-level test requires.

**The labeling unit is the utterance**, on the principle that **the relevance criterion is
labeled at the same unit as the one being retrieved** (`analysis/q15`: the utterance is the
retrieval unit for every modality). Clip labels are produced as well, and retained for describing
the corpus and for the dual-unit rate, but anything evaluated at the utterance uses utterance
labels. No threshold switches between them: a rule that depends on the measured disagreement
rate would be a decision made after seeing data.

**9. What stands in for annotation.** Three measures, declared here:

- **two-model agreement**, between `scripts/label_with_claude.py` (claude-haiku-4-5, five
  keyframes plus an OCR summary) and a second labeling model;
- **self-consistency under perturbation** of keyframe sampling points and of category order in
  the prompt;
- **a criterion check against the audio channel**, which the labeler never receives. The
  on-screen ticker is **not** independent of the labeler: `extract_ocr_summary` filters only
  `label == "logo"`, so ticker, title and subtitle text all reach it (verified 2026-10-03).
  Two measures, each with its direction fixed here:
  **(i) music proportion**, from the `music` segments in `data/output_gemini/`, predicted
  **higher for Celebrating** than for the pooled remainder;
  **(ii) speaker-change rate**, from segment boundaries in the same source, predicted **lower
  for Reporting** than for the pooled remainder, since anchor-read material is monologic.
  Each is a directional prediction, not a search for whichever measure separates best.

These bound **reliability, not validity**, and the write-up says so. Validity rests on Study 1's
human validation of the same taxonomy, model family and genre (κ_inter = 0.747,
κ_avg(VLM) = 0.765), **which covers three of the six languages at the clip level**, and on the
spot check in item 11.

---

## 4. §6.1.3

**10. The detector operating point is unwindowed.** Measured on NTU Paiwan ASR over 102 clips
and three speakers: unwindowed `ari` runs at recall 0.91 and precision 0.43. Restricting the
search to the first 15 phones raises precision to 0.60 but drops recall to 0.68 and does so
unevenly by speaker (0.86 / 0.86 / 0.43); a speaker-varying bias is the one kind this design
cannot afford. The windowed variants (first 25, 15 and 10 phones) are declared here as a
**sensitivity analysis**. **`maya` is never windowed.** `ari` and `maya` are reported
separately throughout; `maya`'s performance is measured on its own and not inherited from
`ari`'s.

**11. The two tests the stage must carry, and the shape of its analyses.**

- **The false-positive test.** Whether the detector's false positives correlate with the label.
  At precision 0.43, with 26 false positives against 20 true ones in the validation material, a
  modest correlation would manufacture an effect. Q7-Q10 is the case in this project where a
  nuisance variable tracked a label.
- **Form to label reports the full label distribution** of the utterances where the detector
  fires, against the corpus-wide distribution. It is **not** a binarized Urging-versus-rest test,
  and it carries no prior that the targets should concentrate in Urging. Reversing the
  conditioning is not an alternative: P(U|F)/P(U) = P(F|U)/P(F), and the independence test is
  symmetric in rows and columns.
- **Form to function** judges force by a rule-based criterion over the published human
  translation, which here is the TITV Chinese subtitle, following Study 2's method. **No LLM
  judges function.**
- **The spot check.** A Paiwan-competent listener checks 20 to 30 retrieved tokens. The sampling
  rule is fixed here so the sample is not chosen after the results are seen: **25 tokens drawn
  by stratified random sampling from the detector's hits in the Paiwan material, stratified by
  target (`ari` / `maya`) and by whether the utterance's label is Urging, allocated
  proportionally to the hit counts, with a fixed seed, drawn before any §6.1.3 result is
  computed.** If fewer than 25 hits exist, all are checked. **Non-Urging hits are included by
  construction**; a sample drawn only from Urging hits would be confirmatory. Form judgement and
  function judgement are recorded separately, and every disagreeing case is listed rather than
  summarized as a rate.

---

## 5. Not pre-registered, deliberately

- **The category distribution**, which cannot be known before collection. This document fixes
  what will be done once it is known, which is what pre-registration is for here.
- **Chapters 1 to 3**, which depend on no new data.
- **Exploratory analyses**, which remain permitted and are labeled as exploratory in the text.

---

## 6. What this document is and is not

It is a repository-internal pre-registration: a file whose commit timestamp precedes the
collection of the Study 3 corpus. It is not registered with an external service. Its force comes
from the commit, so the commit must happen before the first clip is inspected, and this file
must not be edited afterwards.

---

## 7. Amendments

None.

> Format for any amendment: date, the item changed, what it was, what it became, the reason, and
> whether any Study 3 data had been inspected at the time. An amendment made after inspection is
> recorded as such and the affected analysis is reported as exploratory.

---

## Appendix. Measurements that preceded this document

Scripts and results are in `analysis/`. All predate this draft.

| Ref | What it established |
|---|---|
| `q7`-`q10` | The cross-lingual gesture effect is carried by shot framing; the claim is withdrawn |
| `q11` | Power for the shot-level test |
| `q12` | Why the standard representation diagnostic was blind to that confound |
| `q13`-`q15` | Shot segmentation, ICC and design effect; the utterance is the retrieval unit and the shot enters as an inclusion criterion and a covariate |
| `q16` | Label-noise direction of bias |
| `q17` | No countable polar interrogatives in the Study 1 material |
| `q18`-`q19` | The phone layer run without a language inventory emits effectively one inventory across three languages; restricting it recovers the target |
| `q20` | `ari` signal-to-noise 1:2.1 and 1:2.2 across two independent Paiwan corpora |
| `q21` | Detector precision and recall on recognizer output; the operating point in item 10 |
| `q22` | No ready evaluation set for utterance-boundary detection |
| `q23` | The Study 1 prosody result is configuration-dependent |

Claim-level grading for all of the above, including the errors made and corrected along the way,
is in `claim_audit_2026-09.md`.
