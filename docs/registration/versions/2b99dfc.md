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

### Amendment 1, 2026-10-03. The day draw.

**Written before any Study 3 video was downloaded or viewed.** No Study 3 data of any kind had
been inspected when this was written, and none had been collected.

*What was missing.* §1 item 2 fixes the draw of three clips from within a broadcast day. It does
not fix which twelve days are drawn from each language's playlist, and it does not state the
order in which the single random stream is consumed. Both are sampling decisions and both are
fixed here. Nothing in §1 item 2 is changed.

*The day frame.* For each language, every entry of its TITV playlist whose title carries a
parsable eight-digit broadcast date is one day in that language's frame. Entries whose date
cannot be parsed are excluded and listed. The frame is built from playlist metadata alone, with
no video downloaded and no content viewed.

*Exclusion of Study 1's broadcast days.* Study 1 used every one of the ten clips of each of
three broadcast days, and those days have been worked on extensively (§0.2). They are removed
from the day frame in whole, before the draw: **Amis 20260302, Paiwan 20260307, Truku 20260305**
(`data/h264/`, `pilot/labels.json`, and `youtube_download/PROVENANCE.md` for their sources). The
exclusion is by identity, not by content. It is what keeps §2 item 6, which confines the outcome
test to the Study 3 corpus, from requiring an exclusion made after the data have been seen. The
three languages with no Study 1 day, Atayal, Tsou and Yami, lose nothing. If an entry dated
20260302 appears in the Truku playlist, that day is excluded as well and the fact is reported:
one extra excluded day costs nothing, a missed exclusion costs the blindness of the whole Truku
sample.

*The day draw.* If a language's frame spans twelve or more distinct calendar months, twelve
distinct months are drawn uniformly at random without replacement, and one day is drawn
uniformly at random within each drawn month. If the frame spans fewer than twelve months, twelve
days are drawn uniformly at random from the frame without replacement. If the frame holds fewer
than twelve days, all are taken and the shortfall is recorded. The month-stratified branch is
what delivers the spread across months that §0.1 describes. Which branch applies is determined
by the frame, which is metadata, and is recorded with the frame.

*The stream.* One `numpy.random.default_rng(20261005)` instance serves the whole design,
consumed in this order: every day draw first, languages in alphabetical order (Amis, Atayal,
Paiwan, Truku, Tsou, Yami); then every within-day clip draw under §1 item 2, languages in the
same order and days in date order; then any top-up draw under §1 item 4, continuing the same
stream. The draw is reproducible from the seed and the recorded frame alone.

*Records.* The frame and the draw are written to `analysis/study3/frame.json` and
`analysis/study3/draw.json`, which are version controlled. `.gitignore` excludes `data/` whole,
so the media live there and the records do not.

### Amendment 2, 2026-10-03. Which playlist entries form the day frame.

**Written after the playlist metadata was inspected, and before any Study 3 video was
downloaded or viewed and before any draw was run.** What had been inspected is the metadata in
`analysis/study3/frame.json` (commit `4ae0531`): titles, video ids, reported durations, and each
playlist's own title and count. No broadcast content of any kind had been seen. The findings that
prompted this amendment are in `analysis/study3/FRAME_FINDINGS_2026-10-03.md`.

*Item changed.* Amendment 1, *The day frame*.

*What it was.* Every entry of a language's TITV playlist whose title carries a parsable
eight-digit broadcast date is one day in that language's frame.

*What it becomes.* An entry is one day in its language's frame only if all of the following
hold. Its title carries a parsable eight-digit broadcast date, as in Amendment 1. The title's
first 【】 bracket opens with the programme name 看見族語. That bracket names the playlist's own
language, by the Chinese name recorded in `analysis/study3/playlists.json`. An entry whose
bracket names another language is excluded from the frame and listed. A video id listed more than
once in a playlist is one entry. No other exclusion is made. In particular, a special edition
broadcast under the 看見族語 title is in the frame on the same terms as any other 看見族語
broadcast. The exclusion of Study 1's broadcast days in Amendment 1 is unchanged and is applied
after this rule. If, after this rule, any date in a language's frame is carried by more than one
video id, the draw stops and the case is reported rather than resolved.

*Reason.* Section 0.1 defines the corpus as TITV《看見族語》television news, and Study 1's three
broadcasts are 看見族語 broadcasts. The playlists also hold the programmes that preceded 看見族語
(族語午間新聞, 族語晚間新聞) and the one that followed it from September 2026 (族語新聞), along with
one-off specials and entries in other languages. Amendment 1 as written would have admitted all of
these, and its month-stratified branch would have drawn most days from before 看見族語 began. The
restriction keeps the sample within the programme the design describes. Whether 族語新聞 is the
same programme renamed cannot be established from metadata, so it is not admitted.

*Implementation of the draw, fixed before the draw is run.* Amendment 1 fixes the order in which
the stream serves languages; this fixes the calls within a language. Months and days are sorted
chronologically. In the month-stratified branch, `rng.choice(n_months, size=12, replace=False)`
selects twelve months, and then, taking the selected months in chronological order,
`rng.choice(n_days_in_month)` selects one day within each. In the uniform branch,
`rng.choice(n_days, size=12, replace=False)` selects twelve days. The within-day draw under §1
item 2 is `rng.choice(n_clips, size=3, replace=False)` over the day's clip indices in order,
with days taken in date order. The script is `scripts/draw_study3_sample.py`, and its parameters
are read from `analysis/study3/design.json`.

---

### Amendment 3, 2026-10-05. The symbolic detector becomes the baseline, not the subject.

**Written after collection and before any Study 3 content was inspected.** Collection is
complete under Amendments 1 and 2: 72 broadcasts, 720 clips cut, 216 drawn, twelve days in each
of six languages (`analysis/study3/manifest.json`, `draw.json`). `analysis/study3/README.md`
records that no Study 3 content was inspected for any analysis purpose during collection, and
**the author confirmed this on 2026-10-05**. Nothing here changes the corpus: the sample was
drawn for reasons that have nothing to do with §6.1's analysis, so no choice in this amendment
can bias it.

*Item changed.* §4, "§6.1.3", in its entirety. **§4 item 10, the detector operating point, is not
changed.** §4 item 11's sample size is superseded below.

*What it was.* §6.1.3 detected `ari` and `maya` in recognizer output and set the hit set against
the VLM labels, reporting three analyses: form to function, form to label, and retrieval to
function.

*What it becomes.* §6.1.3 is **a comparison of two retrieval routes against one reference
standard.** The two forms remain, inside the symbolic route, as the instrument.

*Why.* Chapter 3 argues that form-based query cannot reach directive force in these languages.
§6.1.3 as written then ran a form-based search, on the two forms §3.4 declares to be the
exceptions, which made the chapter's main application a demonstration of the exception rather
than of the thesis. Nothing in the repository argued the step from "Study 2's function-first
query surfaced these two forms" to "these two forms are the object of study"; the selection was
then reinforced by retrievability, which is a property of the detector and not of the language.
The forms are no longer the research object. The question is the one §7.1 of the outline states:
how much of the directive material a form-based route reaches, and whether a geometric route
reaches more.

**1. The reference standard.** The Study 3 Paiwan utterances whose time-aligned TITV Chinese
subtitle satisfies the rule-based directive criterion of Study 2, unchanged. It is a published
human translation judged by a stated rule, which is why it can serve as a yardstick for both
routes. Source text is `data/ocr_vlm/`; the alignment from the clip-level string to the utterance
is work to be built and its error rate is reported. **No model judges force.** The reference set
is fixed, and its size reported, before either route is scored.

**2. Arm A, symbolic.** The detector of §4 item 10, unchanged: `ari` and `maya` over Allosaurus
`pwn` phone strings, **unwindowed**, windowed variants as sensitivity analysis. **Paiwan only**:
the other five languages have no comparable target and two have no phone inventory in the
recognizer at all (`analysis/q19`), so the route is undefined for them and the asymmetry is
stated rather than worked around.

**3. Arm B, geometric.** Query-by-example at the utterance by multimodal similarity, as in the
retrieval evaluation of Chapter 5.

**4. Seeds and the split.** The twelve Paiwan broadcast days are divided into a **seed half and a
scoring half, six and six**, drawn by continuing the single `default_rng(20261005)` stream after
every draw of Amendment 1, taking 6 of the 12 Paiwan days without replacement in date order.
Arm B's seeds are the reference-set utterances of the seed half. **Both arms are scored on the
scoring half only.** Study 1's and Study 2's material supplies no seeds, so §2 item 6 is not
engaged.

**5. The two routes return the same number of utterances.** `k` for Arm B is set so that
`|B| = |A|` on the scoring half, determined after Arm A has run and before Arm B is scored. This
removes the one free parameter that could otherwise advantage either route, and makes the
comparison one of what each route reaches rather than of how much each returns.

**6. The outcome.** Of the reference set in the scoring half, the fraction reached by A only, by
B only, by both, and by neither. **The primary statistic is the fraction reached by B and not by
A**, with a cluster bootstrap at the broadcast day per §1 item 1. Secondary: the precision of
each route against the reference set, and the size of the reference set as a proportion of all
Paiwan utterances in the scoring half. Study 2's finding that 37.6% of directive sentences carry
no imperative or hortative suffix is from a different corpus and register and **is not a
predicted value here**.

**A null is a result and is reported as one.** If Arm B reaches nothing Arm A does not, that is a
negative finding about the geometric route on this material, stated plainly.

**7. Withdrawn with the superseded design.** The binarized Urging-versus-rest test and the lift
statistic, and with them the question of which direction to condition in. The full label
distribution of each cell is still reported, descriptively, because it is informative about what
the labeler does; **no claim about directive force rests on it.**

**8. The human component, superseding §4 item 11's sample size.** **40 tokens, ten from each of
the four cells of item 6**, stratified by cell, fixed seed, drawn before any outcome is computed.
If a cell holds fewer than ten, all of it is checked and the shortfall recorded. Form judgement
and function judgement are recorded separately, and every disagreeing case is listed rather than
summarized as a rate. The rest of §4 item 11 stands.

**9. Not changed.** §1 items 1 to 4; §2 in full, including the gesture section's distance
gradient and its primary test; §3 items 8 and 9; §4 item 10; §5; and the corpus itself.

---

### Amendment 4, 2026-10-07. Bunun replaces Tsou, and a transcription modality is added.

#### The inspection statement this amendment rests on

**Written after Study 3 collection and before any Study 3 content was inspected.** Collection is
complete under Amendments 1 and 2 (216 clips, twelve days in each of six languages). No Study 3
outcome, label distribution or clip content has been computed or viewed for any analysis
purpose. The author confirms this statement as of 2026-10-07.

What *has* been seen is the result of an ASR pilot run on material **outside** the Study 3
corpus: 族語E樂園「情境族語」season 1 read speech (50 sentences in each of `ami`, `pwn`, `tay`,
`trv`, `tao`, `bnn`) and NTU Paiwan ASR (102 clips). The pilot's decision rule was committed
before the run (`scripts/asr_pilot/README.md`, commit `f6bb612`). The pilot script
(`scripts/asr_pilot/omni_pilot.py`) reads only the 族語E樂園 audio it downloads and
`data/ntu_paiwan_asr/`, and the run was made under written instructions that forbade opening,
reading or listing anything under `data/study3/` or `analysis/study3/`. The pilot is the reason for this amendment, so its
rule, its result and its limits are stated here in full. The rule took effect at commit `f6bb612`,
as its own text provides; the committed file still opens with "Status: DRAFT" and was
deliberately not edited after the result was seen.

One further disclosure. To place the Bunun draw (A4) without disturbing the existing one, the
author authorized reading **metadata only** under `analysis/study3/`. Each JSON file was loaded
by a script that printed only the fields listed:

- `analysis/study3/playlists.json`: the whole file (the six languages' names, their playlist
  URLs and the `source` note).
- `analysis/study3/frame.json`: the top-level `yt_dlp_version`, the command prefix of each
  language's `yt_dlp_command`, and each language's `playlist_url` and `summary.entries` (the
  number of entries the playlist listing returned, before date parsing and the Amendment 2
  filter; between 306 and 365, none equal to 100).
- `analysis/study3/draw.json`: the list of top-level keys; the number of `stream` steps (150);
  the language and purpose of the last step (Yami, clips of the 2026-07-28 broadcast); and the
  set of purpose types (`clips`, `day`, `months`).
- `analysis/study3/sources.json`: a count of the yt-dlp version and command prefix across the
  72 records.
- `analysis/study3/design.json`: existence checked only; not opened.

Nothing under `data/study3/` was opened, listed or checked by name. One repository-wide search
for `*.py` files (ripgrep, looking for the code that computes the text modality) also covered
both directories; it returned no match from either and printed nothing from them. No manifest,
clip, transcript, OCR output, label or other content was opened. One day identifier (the Yami date above) was seen as
part of the draw record.

#### Part A. Bunun replaces Tsou

##### A1. What is changed

*Items changed.* §0.1, the list of six languages. Amendment 1, *The stream*, only insofar as it
names Tsou. §2 item 7.

*What it was.* Six languages: Amis, Paiwan, Truku, Atayal, Tsou, Yami.

*What it becomes.* Six languages: Amis, Paiwan, Truku, Atayal, **Bunun**, Yami. Tsou leaves the
Study 3 analyses.

##### A2. Why

The transcription modality of Part B needs an ASR system for each language in the design. Meta
Omnilingual ASR supports `ami`, `pwn`, `tay`, `trv`, `bnn` and `tao`, and does not support `tsu`.
With Tsou in the design, the transcription modality would run under one model for five languages
and a different, zero-shot model for the sixth, so transcript quality would vary with language by
construction.

Supported is not the same as usable, so the substitution was made conditional on a pilot with a
rule fixed in advance (`scripts/asr_pilot/README.md`, commit `f6bb612`): switch only if Bunun's
median CER is at most 0.25 and at least four of the five retained languages also reach it. The
result (`scripts/asr_pilot/results/summary.json`):

| Language | Median CER (n = 50) |
| --- | --- |
| `tay` | 0.081 |
| `trv` | 0.089 |
| `bnn` | 0.095 |
| `ami` | 0.116 |
| `pwn` | 0.116 |
| `tao` | 0.182 |

All six pass; the rule's outcome is to switch. The reason for the change is the instrument's
coverage, a property of the tool and not of any Study 3 content.

##### A3. What the substitution does not change

The distance classes keep their sizes. Bunun belongs to a Formosan primary branch outside
Atayalic, so it takes Tsou's place in the Middle class: **Short 1 pair (Atayal and Truku), Middle
9 pairs, Long 5 pairs (Yami against the other five)**, exactly as §2 item 5 counts them. The
primary test, Middle against Long, is unchanged.

§1 items 1 to 4, §3, §4 and Amendment 3 are unchanged. Amendment 3 concerns Paiwan only and does
not involve Tsou.

##### A4. Drawing the Bunun sample without disturbing the existing draw

Amendment 1 serves every draw from one `default_rng(20261005)` stream with languages in
alphabetical order, and Amendment 3 item 4 continues that stream. Inserting Bunun into that order
would change every draw after it, including Paiwan's, Truku's and Yami's days and the Paiwan seed
split. **The existing draw is therefore left exactly as it is**, and Bunun is drawn from a
separate stream:

- **Frame.** Built by the rules of Amendments 1 and 2 from the Bunun playlist: an entry is a day
  only if its first 【】 bracket opens with 看見族語 and names 布農族語. No Study 1 day exists for
  Bunun, so nothing is excluded by identity. The playlist is added to
  `analysis/study3/playlists.json`: `PLtE4cgfvTnt6sYPyiGk9iwz6uf-qetSH5`, titled
  【族語新聞】布農族語 on the TITV News channel (`UCkceO_uT0eWlMhX-04rxAMQ`), the only Bunun
  playlist among the channel's 39 (listed 2026-10-07, metadata only). Confirmed by the author.
- **State of the existing stream.** At the time of writing the `default_rng(20261005)` stream
  stands at step 150, and the Paiwan seed split of Amendment 3 item 4 has not yet been drawn. It
  will continue from step 150 as Amendment 3 provides. Nothing in this amendment consumes a value
  from it.
- **Stream.** `numpy.random.default_rng(202610074)` (the amendment date followed by the
  amendment number), used only for Bunun: the day draw first, by the branch rule of Amendment 1
  with the calls of Amendment 2, then the within-day draw of §1 item 2.
- **Top-up.** §1 item 4 applies to Bunun from the same separate stream.
- **Records.** Appended to `analysis/study3/frame.json` and `draw.json` under a Bunun key, with
  the seed recorded beside it. Media go to `data/study3/` as for the other languages. Existing
  entries in these files are not edited.
- **Tooling.** The Bunun frame is built with the same yt-dlp version and settings that built the
  existing frames, recorded here. If the playlist returns a truncated listing (yt-dlp
  2026.03.03 returns only the first 100 entries), the frame step stops and nothing is drawn.
  The existing frames and all 72 broadcasts were retrieved with yt-dlp 2026.08.19, run as
  `YTDLP="C:/Users/user/miniconda3/python.exe -m yt_dlp"`, as recorded in
  `analysis/study3/frame.json` and `sources.json`; Bunun uses the same.

##### A5. The Tsou material already collected

The Tsou clips (twelve days, 36 drawn clips) were collected under Amendments 1 and 2 and have not
been inspected. They are **retained on disk and not opened**. They take no part in any Study 3
analysis, primary or exploratory. The dissertation reports that they were collected and set
aside, and why. Leaving an exploratory use open would leave room to decide after the primary
results whether Tsou is worth looking at, which this amendment exists to prevent.

##### A6. §2 item 7

§2 item 7 declared an alternative analysis subdividing the Middle class "once Tsou data are in".
Its rationale was specific to Tsou. **It is withdrawn.** A subdivision written for Bunun now,
after the pilot, would be a new hypothesis rather than the one registered, and is not added.

#### Part B. A transcription modality is added

##### B1. What is added

*Item added.* A fifth modality for the retrieval evaluation: **transcription**, an embedding of
the Omnilingual ASR transcript of each utterance in its own language.

*Why.* The existing text modality is the OCR of the TITV Chinese subtitle. It is a translation,
not the language itself, and it is **not independent of the labels**: §3 item 9 records that
`extract_ocr_summary` passes subtitle text to the labeling model. A modality built from the
speech itself does not reach the labeler, and it is the strongest available form-based
representation of the language, which is what the umbrella thesis needs as its comparison.

The existing text modality is renamed **translation (subtitle)** in the write-up and its
dependence on the labeler's input is stated wherever its result is reported. Nothing about how
it is computed changes.

##### B2. How it is produced

- **Model.** `omniASR_LLM_7B_v2` with the language code given, the model of the pilot. The
  environment is `scripts/asr_pilot/environment.txt`.
- **Segmentation.** The existing utterance boundaries, unchanged. Each utterance is passed to the
  model as a separate segment; an utterance longer than 40 s is split at its longest internal
  pause and the transcripts joined. The number of such splits is reported.
- **Normalization.** As in the pilot: NFC, lowercase, apostrophe variants unified, punctuation
  removed except `'` and `^`.
- **Embedding.** No Study 3 code for either text modality exists yet. Both are computed by one
  program with identical settings, so that they differ only in their input, and the settings
  are those of the Study 1 subtitle modality (`pilot/experiment.py`), in keeping with B1's "nothing
  about how it is computed changes":
  - `FacebookAI/xlm-roberta-base`, `max_length` 512, mean of `last_hidden_state` masked by the
    attention mask. The Hugging Face revision hash is recorded at the first run and used for
    both modalities thereafter. The number of truncated inputs is reported.
  - No centering. Cosine similarity on the pooled vectors.
  - An empty input (no transcript, or no subtitle) gives a zero vector, as in Study 1; the number
    of empty inputs per language and modality is reported.
  - The input unit is the retrieval unit of §3 item 8 for both modalities.
- **Non-neural reference.** As in Study 1: TF-IDF with `analyzer="char_wb"`, n-grams 1 to 3,
  `sublinear_tf=True`, `norm="l2"`, fitted once on all Study 3 inputs of that modality together.
- **Sensitivity analysis, both modalities.** The same vectors centered on the mean of each
  language's valid rows, then L2-normalized (the procedure of `corpus/build_similarity.py`,
  scoped per language). Reported beside the main result, never in its place. XLM-R was not
  trained on these languages, so the neural result is read against the n-gram reference;
  tuning or fine-tuning an encoder is outside this amendment.

##### B3. What the pilot cannot tell us, and what is reported instead

The pilot's CERs are an optimistic bound. 族語E樂園 and NTU Paiwan ASR are public, and
Omnilingual's training data may contain either; some per-dialect medians (0.02, 0.04) are low
enough to suggest it. The pilot also used clean read speech, not broadcast news.

§3 rules out new bulk human annotation for Study 3, and that applies here: **no news gold set is
transcribed, and transcript quality on Study 3 material is not measured.** What the write-up
reports instead, with the limits stated:

- **Read speech, all six languages.** The pilot's median CER per language
  (`scripts/asr_pilot/results/summary.json`), described as an optimistic bound.
- **Spontaneous speech, Paiwan only.** The secondary result on NTU Paiwan ASR (102 clips, median
  CER 0.173, against 0.116 on read speech), the only existing human transcription of spontaneous
  speech available for any of the six languages. It is reported as one language's indication of
  how far quality falls off read speech, not as an estimate for the others.
- **Glottal and epiglottal marks.** The pilot shows these (`'`, `^`) are systematically dropped
  (`taisic’an` → `taisicaan`, `ina^` → `ina`). The rate at which they are lost is computed on the
  pilot data and reported separately from CER, because in most of these languages they are
  phonemic.

Because news-speech quality is unknown for five of the six languages, B4 confines the modality
more tightly than it would with a news gold set.

##### B4. Where the transcription modality may and may not enter

Transcript quality differs by language, it is not measured on news speech, and the language with
the worst pilot CER, Yami, is the **only** member of the Long class. Lower retrieval for Yami
under this modality would point in the predicted direction of the distance gradient whether or
not the gradient exists.

- **Not in the primary test of §2 item 5.** The primary distance-gradient test is run on the
  modalities already declared, exactly as before.
- **Secondary only.** Results for the transcription modality are reported as secondary
  analyses, each with the language's pilot CER beside it.
- **No cross-language inference under this modality.** Per-language results are reported side
  by side, descriptively. No test across languages or distance classes is run on the
  transcription modality, and no difference between languages under it is read as a linguistic
  difference, because the quality of the input is not known to be comparable.
- **The comparison the modality exists for** is within language: does the transcription
  modality retrieve directive utterances as well as the geometric modalities do on the same
  language's material.

#### Consequences outside this document

These are not governed by the pre-registration (§5 excludes Chapters 1 to 3) but follow from it:

- Chapter 3 replaces Tsou with Bunun, including §3.4's discussion of the Tsou auxiliary and the
  absence of a dedicated imperative focus form, which is lost as a case.
- `reference_grammars/` now holds 布農語語法概論 (黃慧娟、施朝凱; 郡群 dialect), and
  `祈使標記跨語言對照.md` has a Bunun row. Its implications section was written with Tsou and is
  rewritten before any Study 3 data are seen. The grammar describes 郡群 only, which Chapter 3
  states wherever it describes Bunun imperatives.
- `analysis/study3/README.md` records the Bunun collection and the status of the Tsou clips.

---

### Amendment 5, 2026-10-07. Tsou is kept, and Amendment 4 Part A is withdrawn.

#### Inspection statement

**Written after Amendment 4 and before any Study 3 content was inspected.** Nothing has been
inspected beyond what Amendment 4 discloses. Since Amendment 4 was committed (`0e61e39`), no
Bunun frame has been built, no Bunun draw made, no Bunun video downloaded and no file written
under `analysis/study3/` or `data/study3/`. The only file read under `analysis/study3/` was
`playlists.json` (already read in full and disclosed in Amendment 4), read again to check its
formatting before the Bunun entry was added; the entry was not added. Nothing under
`data/study3/` was read. The four collection scripts were not re-read after the commit; what was
said about them rested on reading them as code earlier the same day. The author confirms this
statement as of 2026-10-07.

#### 1. What is changed

*Items changed.* Amendment 4 Part A (A1 to A6) is withdrawn in whole. Amendment 4 Part B is
narrowed to five languages (item 3 below).

*What it was.* Bunun replaces Tsou; Bunun is drawn from a separate `default_rng(202610074)`
stream; the Tsou clips are set aside; §2 item 7 is withdrawn.

*What it becomes.* The six languages of §0.1 as first registered: Amis, Atayal, Paiwan, Truku,
Tsou, Yami. Everything Amendment 4 Part A changed returns to its state before Amendment 4:

- §0.1 and Amendment 1, *The stream*, stand as written. The existing draw (`analysis/study3/draw.json`,
  150 steps of `default_rng(20261005)`) and the existing frame, sources and clips stand as they
  are, and Amendment 3 item 4 continues that stream from step 150 as it provides.
- The Tsou clips are part of the Study 3 corpus on the same terms as the other five languages.
- §2 item 7 stands as registered.
- The distance classes are those of §2 item 5: Short 1 pair, Middle 9, Long 5.
- Nothing for Bunun is built, drawn or downloaded. The separate stream `default_rng(202610074)`
  is never used.

#### 2. Why

Amendment 4's pilot rule established that substituting Bunun was **admissible**: Omnilingual ASR
reaches the threshold on Bunun read speech. Implementing it showed its cost. Amendment 1 consumes
one stream in alphabetical language order, day draws for every language before any clip draw, so
Bunun can enter the design only by a second stream or by redrawing every language's clips and
the days of Paiwan, Truku and Yami. Either departs from the single-stream design that Amendment 1
fixed.

The benefit, meanwhile, was reduced by Amendment 4 itself. A2 justified the substitution by the
transcription modality: with Tsou unsupported, transcript quality would vary with language by
construction. B4 then confined the transcription modality to **within-language** comparison and
ruled out any test across languages or distance classes under it. Under that restriction, a
language without ASR loses only its own within-language comparison and biases nothing else.
Keeping the registered design intact is worth more than one further within-language comparison.

This reasoning uses no Study 3 data and does not depend on any pilot CER. The pilot's result
stands on record as Amendment 4 reports it; the decision not to act on it is made here.

#### 3. Part B, narrowed to five languages

- **Coverage.** The transcription modality is computed for Amis, Atayal, Paiwan, Truku and Yami
  (`ami`, `tay`, `pwn`, `trv`, `tao`). Omnilingual ASR does not support `tsu`, and **no Tsou
  transcription modality is computed**, zero-shot or otherwise; the write-up states this as a
  limitation. Tsou takes part in every other modality and in every primary analysis.
- **B3.** "Read speech, all six languages" reads as the pilot's median CER for the five covered
  languages; the Bunun result is reported only as part of the pilot record.
- **B2 and B4** apply unchanged to the five covered languages. The TF-IDF reference is fitted on
  the transcription inputs of those five.
- **Disclosure and pilot record.** Amendment 4's inspection statement, its disclosure of the
  metadata read and its account of the pilot remain in force as written.

#### Consequences outside this document

- Chapter 3 keeps Tsou, including §3.4's discussion of the Tsou auxiliary.
- `reference_grammars/` keeps 布農語語法概論 and the Bunun row of `祈使標記跨語言對照.md` as
  reference material, marked as not a Study 3 language. The table's implications section stands
  as written for the six registered languages.
- `analysis/study3/README.md` records that no Bunun collection was made.

---

### Amendment 6, 2026-10-07. The subtitle modality leaves Arm B; Arm T and Arm C are added; the form judgement is extended.

#### Inspection statement

**Written after Amendment 5 and before any Study 3 content was inspected.** Since Amendment 5
was committed (`5f40b6c`), and through the commit of this amendment's draft (`5d35551`) to the
commit that appends it here, nothing under `data/study3/` has been read, listed or processed,
and no Study 3 outcome, label distribution, transcript, subtitle or clip content has been
computed or viewed. One file under `analysis/study3/` was read, in part: `README.md`, for its
description of the cut rule and its per-language counts table (Paiwan: 12 broadcast days, 120
clips cut, 36 drawn, 84 not drawn). That file is documentation written during collection and
contains no content, outcome, label or transcript. A one-paragraph edit to it was made and
reverted the same session, before the draft commit, so it is unchanged at `5d35551`. The author
confirms this statement, and the values in items 4 and 5 (25 candidate tokens, 4 broadcast days;
10 further listener tokens), as of 2026-10-07.

---

#### 1. The translation (subtitle) modality leaves Arm B

*Item changed.* Amendment 3 item 3.

*What it was.* Arm B is "query-by-example at the utterance by multimodal similarity, as in the
retrieval evaluation of Chapter 5", which leaves the modality set to Chapter 5, and Chapter 5
includes the translation (subtitle) modality.

*What it becomes.* Arm B's representation is the fusion of the **non-text** modalities only: the
audio representations, prosody and kinematics, as computed for Chapter 5. It excludes the
translation (subtitle) modality and the transcription modality of Amendment 4 Part B. Fusion is
an **equal-weight sum of per-modality similarities, each standardized per query** (z-score over
that query's candidate row) before summing, so that no modality dominates through the spread of
its cosine distribution. The weights are fixed here and are not tuned.

*Why.* The reference standard of Amendment 3 item 1 is defined on the TITV Chinese subtitle, and
Arm B's seeds are chosen by membership in it. With the subtitle modality in the fusion, a seed
would retrieve its neighbours partly by subtitle similarity, so part of Arm B's reach would be
the reference standard retrieving itself. That inflates exactly the primary statistic, the
fraction reached by B and not A. The transcription modality is excluded for a different reason:
it is form-based, and it is reported as a route of its own (item 2), so that the comparison
keeps form-based and geometric evidence apart.

The with-subtitle variant is computed once and reported as a **diagnostic of the size of the
circularity**, never in place of the main result.

#### 2. Arm T, the text route (secondary)

*Item added.* Query-by-example on the **transcription modality alone** (Amendment 4 Part B,
Paiwan), with Arm B's seeds, scored on the scoring half, returning the same number of
utterances as Arm A and Arm B (`|T| = |A|`). It is the comparison Amendment 4 B4 names as the
reason the transcription modality exists: within one language, does a representation of the
transcribed words retrieve directive utterances as well as the non-text modalities do.

It is **secondary** under B4, reported with the Paiwan pilot CER beside it (read speech 0.116,
NTU Paiwan ASR 0.173), descriptively, and enters no test. Its place in the write-up is the third
column of the route comparison: keyword (A), transcribed text (T), non-text multimodal (B).

#### 3. Arm C, form-filtered multimodal re-ranking

*Item added.* Arm B asks whether the geometric route reaches directive utterances the form-based
route misses (a gain in **recall**). Arm C asks the complementary question: among the
utterances a form-based search over-retrieves, can multimodal similarity separate the directive
ones (a gain in **precision**). It is defined for the imperative suffix `-u` only, in Paiwan
only. `-i` is not attempted: in 100 Paiwan Texts only 103 of 1,113 `-i` tokens carry `hort`, so
a candidate set cannot be defined on form with any useful concentration.

- **Candidate set.** Utterances in the scoring half whose transcription (normalized as in
  Amendment 4 B2) contains at least one whitespace-delimited token of two or more characters
  ending in `u`. The rule is fixed here and no word is excluded from it afterwards.
- **Seeds.** The reference-set utterances of the seed half whose transcription contains a
  candidate token by the same rule. If fewer than five exist, all reference-set utterances of
  the seed half are used, and which rule applied is reported.
- **Score.** For each candidate, the mean of its standardized fused similarity (item 1's
  representation) to the seeds.
- **Framing-only baseline.** The same procedure with the representation replaced by the shot
  framing measures of §4.3 alone. This is the control for the confound of `analysis/q7`–`q10`:
  in television news, an anchor addressing the camera is the most directive-looking visual
  configuration, and it is a property of the shot. If framing alone ranks the candidates about
  as well, the multimodal contribution is not established.
- **Primary statistic.** Average precision of the Arm C ranking against the reference standard,
  **minus** the average precision of the framing-only ranking, with a 95% interval from the
  cluster bootstrap at the broadcast day (§1 item 1). Secondary: Arm C's average precision
  against the base rate (the proportion of reference-set utterances among the candidates), and
  precision in the top decile, descriptive.
- **Units.** Utterances lying within one shot are the primary analysis; utterances straddling a
  shot boundary are the declared secondary analysis, as in `analysis/q15`.
- **What the reference standard does and does not establish here.** It judges whether an
  utterance is directive, not whether the word-final `u` in it is the imperative suffix; an
  utterance can be directive and contain an unrelated final `u`. The second question is answered
  only by the human check of item 5, and the write-up keeps the two apart.
- **Test eligibility.** A test is run only if the candidate set contains reference-set
  utterances in at least four broadcast days of the scoring half. Otherwise Arm C is reported
  descriptively. Counting reference-set membership is permitted before scoring, since Amendment 3
  item 1 fixes and reports the reference set before either route is scored.

A null is a result: if Arm C does not beat the framing-only ranking, that is reported as a
negative finding about multimodal disambiguation on this material.

#### 4. The conditional whole-day extension, for Arm C only

*Item added.* Television news may contain few imperatives, so Arm C's candidate set may be small
on the registered sample. Whether to extend is decided **now**, by a rule on counts, not later
by inspection of results.

- **Trigger.** After the transcription modality has been computed on the registered sample and
  **before any Arm C score is computed**, count the candidate tokens of item 3 in the scoring
  half and the broadcast days they span. If there are **fewer than 25 candidate tokens, or they
  span fewer than 4 broadcast days**, the extension runs. Otherwise it does not. The two counts
  and the decision are recorded before Arm C is scored. The trigger looks only at form-level
  counts; it never looks at reference-set membership, Arm C scores or any other outcome.
- **The extension.** Arm C is run on the **whole broadcasts** of the twelve Paiwan days, all ten
  clips of each (120 clips: the 36 drawn and the 84 not drawn), keeping the seed/scoring split of
  Amendment 3 item 4 at the day level. Using whole days involves no selection, so it introduces
  no sampling bias. It adds utterances, not clusters: inference remains over the same six
  scoring days, and the write-up says so.
- **Scope.** The 84 undrawn clips enter **Arm C and nothing else**. Every other analysis,
  Arms A, B and T included, remains on the registered 216 clips. This amends the statement in
  `analysis/study3/README.md` that the undrawn clips enter no analysis, for this one arm.
  The pipeline steps Arm C needs (utterance boundaries, transcription, subtitle OCR for the
  reference standard, shot segmentation and framing measures, the non-text features) are run on
  them; labeling is not, since Arm C does not use labels.
- **No further extension is pre-registered.** Drawing new Paiwan broadcast days for Arm C is not
  provided for here. If it is done later, the added analysis is reported as exploratory.

#### 5. The form judgement, extended

*Item changed.* Amendment 3 item 8.

*What it was.* Form judgement and function judgement are recorded separately on 40 tokens, ten
per cell of the A/B table; the form judgement concerns the detector's targets.

*What it becomes.* For every checked token the listener records **every** directive form the
utterance contains, from: `ari`; `maya`; the imperative suffix `-u`; the imperative or hortative
suffix `-i`; another directive form (written down); none. Several may be recorded together. The
function judgement is recorded separately, as before.

**Ten further tokens** are drawn at random from the top decile of the Arm C ranking, with
`default_rng(20261007)`, before any Arm C outcome is examined, and checked the same way. They are
the only evidence on whether the `-u` that Arm C ranks highly is the imperative suffix.

*Why.* It lets the result be reported against three levels of visibility of directive form:
forms a search can key on (`ari`, `maya`), forms present but too small to key on (`-u`, `-i`),
and no form at all. The cells reached by B and not A are the ones the argument turns on, and the
listener's form judgement is what says which level they fall on.

*How it is reported.* Descriptively only, with Wilson intervals. At ten tokens per cell the
intervals are wide (five of ten gives roughly 0.24 to 0.76), and no proportion from this check
is tested or compared across cells.

#### 6. How §6.1.3 is reported

Not a change to any analysis; recorded so that the layout is fixed before results exist.

1. The route comparison: the reference set divided among A only, B only, both and neither
   (primary, Amendment 3 item 6), with Arm T's reach given beside it (secondary).
2. The human-check table: for each cell, the forms recorded by the listener, grouped into the
   three levels of item 5.
3. Arm C: average precision against the framing-only baseline and the base rate, with whether
   the item 4 extension ran.

#### 6a. A text-only structured classifier, exploratory

Declared so that its place is fixed in advance, though §5 already permits exploratory analyses.
A structured classifier with confidence scores (TypeSafe AI's Jev, or another) may be run on
Study 3 Paiwan transcripts **only after every registered §6.1.3 outcome has been computed and
recorded**. Its output never enters the reference standard, the definition or scoring of any
arm, or the human-check draws, and every result from it is labeled exploratory. Its calibration
is assessed first on Study 2 material (`00_dissertation_outline.md` §7.5), not on Study 3.

#### 7. Not changed

§1 to §5 except as stated; Amendment 3 items 1, 2, 4, 5, 6 and 7; the reference standard; the
detector operating point (§4 item 10); Amendments 4 and 5; the corpus.

---

#### Consequences outside this document

- `analysis/study3/README.md` states that the undrawn clips enter no analysis. That paragraph
  gains the Arm C exception of item 4, in a follow-up commit after this one.
- `00_dissertation_outline.md` §4.5, §6.1 and §7.5, `CLAUDE.md`/`AGENTS.md` §4 and `glossary.md`
  describe this amendment as drafted; they are updated to "in force" in the same follow-up commit.

---

### Amendment 7, 2026-10-11. The reference standard's subtitle text is read line by line, and the phone layer is recorded.

#### Inspection statement

**Written after Amendment 6 and before any Study 3 content was inspected.** Since Amendment 6 was
appended, no Study 3 clip, transcript, subtitle, OCR output, phone string, label or outcome has
been computed or viewed, and nothing under `data/study3/` has been read or processed. Two
listings were made on 2026-10-11, both disclosed here: `ls data/study3` (which printed the two
directory names `clips` and `source`) and `du -sh data/study3/clips` (which printed one total
size, 40 GB). Neither printed a file name or any content. Under `analysis/study3/`, three files
were read in part: `README.md` and `FRAME_FINDINGS_2026-10-03.md` (collection documentation), and
`manifest.json`, whose clip records (file path, language, broadcast date, clip index, times,
`drawn`, SHA-256) were used to write the upload checksum lists `pipeline/server/*.sha256`. The
manifest holds no content, label or outcome. The measurements behind this amendment use only
Study 1's 30 clips, the NTU Paiwan ASR corpus and the 語法概論 volumes (`analysis/q24`). The draft was committed as `26f1901`; between that commit and
the commit that appends it here nothing under `data/study3/` was read, listed or processed. **The
author confirms this statement, including that no OCR, transcription or labeling has been run on
the Study 3 clips, as of 2026-10-11.**

---

#### 1. How the reference standard's subtitle text is produced

*Item changed.* Amendment 3 item 1, the sentence "Source text is `data/ocr_vlm/`; the alignment
from the clip-level string to the utterance is work to be built and its error rate is reported."
The reference standard itself (the Paiwan utterances whose subtitle satisfies Study 2's rule-based
directive criterion) is not changed.

*What it was.* `data/ocr_vlm/`, produced by `scripts/ocr_vlm_reextract.py`: Claude Haiku reads
one whole frame, scaled to 640 px, every 10 s; the readings are de-duplicated and joined into one
string per clip, with no time stamps.

*What it becomes.* Subtitle text is produced per **subtitle line**, with a time span, and then
assigned to utterances:

1. **Per-frame OCR.** PaddleOCR as in MOCO stage 5.1 (`new_make_ocr_raw_results.py`), about one
   frame per second, uncorrected (`ocr_raw_results`). The majority-vote correction step is not
   used for this purpose.
2. **Line segmentation.** `pipeline/stages/subtitle_lines.py` as committed at `b218de1`: boxes lying wholly at x ≤ 300 (the programme watermark and language label) are
   excluded; the subtitle band is a box vertical centre in [555, 625) of the 1280×720 frame and
   the banner band [625, 700); consecutive frames whose subtitle-band text is within a
   normalized edit distance of 0.5 form one line, with start and end times.
3. **Reading.** One keyframe per line, at the midpoint of its span, cropped to x 300 to 1280 and
   y 540 to 640 at native resolution, read by the model of the existing script
   (`claude-haiku-4-5-20251001`, temperature 0) with this prompt, verbatim:
   "This image is the subtitle area of a frame from a Taiwanese Indigenous Television news
   broadcast. Transcribe the subtitle text exactly as shown, in the characters shown. Do not
   translate, correct or complete it. Output only the text. If there is no subtitle text,
   output exactly: (none)". If the reading is `(none)` or empty while the line exists, the
   line's most frequent OCR reading (step 2) is used and the line is flagged. The number of
   flagged lines is reported.
4. **Assignment to utterances.** Each line is assigned to the utterance with which its time span
   overlaps most; a line overlapping no utterance is unassigned. An utterance's subtitle text is
   its assigned lines joined in time order; an utterance with none has empty subtitle text. No
   time offset between subtitle and speech is applied or tuned. Reported: the number of lines,
   of unassigned lines, and of utterances with empty subtitle text, per language.
5. **The banner band** (headlines, speaker name tags) is kept as a separate field and is **not**
   part of the subtitle text the directive criterion is applied to.

*Why.* Measured on Study 1 (`analysis/q24` §2; 60 frames, 52 with a subtitle line): the
registered layer recovers only 15 of 52 on-screen lines (29%, 95% CI by clip bootstrap 17 to
42%), because a line stays on screen for a few seconds and the layer reads one frame in ten
seconds; and it has no time stamps, so the alignment Amendment 3 called for could not be built
from it. A reference standard drawn from it would contain at most about a third of the directive
utterances it defines, selected by where the ten-second grid falls. The watermark exclusion
removes the 見族X fragments that the whole-frame reading carries. Line segmentation alone
recovered 52 of 52 lines (line-text CER 0.142 with the OCR majority reading); the VLM reading of
each line is kept because OCR alone has CER 0.20 per frame.

*Check before Study 3.* The procedure is first run on Study 1's 30 clips and scored against the
frame reading of `analysis/q24` §2 (60 frames; read by a model, not by a person): coverage of the
52 lines and CER of the assigned text. The result is reported whatever it is and does not change
the procedure. No new human annotation is done.

#### 2. The translation (subtitle) modality takes the same text

*Item clarified.* Amendment 4 Part B2 fixes how the translation (subtitle) modality is embedded
and that its input unit is the utterance, but not where its utterance-level text comes from.

*What it becomes.* The utterance subtitle text of item 1 step 4, the same text the reference
standard is judged on. Embedding settings are unchanged (Amendment 4 B2).

*Why.* One subtitle text per utterance everywhere. In particular, the with-subtitle variant of
Arm B (Amendment 6 item 1), which measures how far the reference standard retrieves itself, is
only a measure of that circularity if the modality and the reference standard read the same text.

#### 3. The phone layer, recorded

*Item recorded, not changed.* The pre-registration does not specify the phone layer outside Arm A.
This records the choice made before any Study 3 output exists.

*What it is.* Allosaurus 1.0.2, model `uni2005`, decoded with the clip's own inventory file
(`pipeline/inventories/<iso>.txt`, derived from the phonology chapters of the 語法概論 volumes by
`build_inventories.py`, segments the model lacks mapped by Allosaurus' articulatory similarity),
`emit` 1.0, `approximate` false; `pipeline/stages/phone_layer.py` writes the inventory's SHA-256
with every output. ZIPA (`anyspeech/zipa-large-crctc-500k`) is reported as a comparison only.

*Why.* On NTU Paiwan ASR (102 clips, 8,442 gold phones) the grammar-derived inventory gives PER
0.411, against 0.445 for Allosaurus' shipped Paiwan inventory, 0.496 for the Mandarin inventory
and 0.722 unrestricted. ZIPA gives 0.312 but emits `ʔ` 4 times against 259 in the gold and `ə`
33 times against 387. Neither recognizer emits `q`, `c` or `ʎ`.

*Arm A is not changed.* It remains `ari` and `maya` over Allosaurus `pwn` phone strings (shipped
inventory, `emit` 1.0), the configuration whose operating point §4 item 10 fixed;
`phone_layer.py` writes that tier separately for Paiwan clips.

*Correction to the evidence table.* The row for `q18`-`q19` in the Appendix says the phone layer
was "run without a language inventory". `analysis/q24` §1.1 shows the Study 1 layer was decoded
under the Mandarin inventory (`lang_id="cmn"`, `emit` 2). The finding the row reports is
unchanged; its stated cause is corrected here, and the Appendix is not edited.

#### 4. Not changed

The reference standard's definition; Study 2's directive criterion; the existing utterance
boundaries; the labeling model and its input (`ocr_corrected` summary and five keyframes, the
input with which Study 1's human validation was obtained); Arms A, B, C and T except as stated in
item 2; the listener check (Amendment 3 item 8, Amendment 6 item 5); the detector operating
point (§4 item 10); the corpus.

---

#### Consequences outside this document

- `pipeline/server/RUNBOOK.md` step 9 follows item 1; the line-reading and assignment programs are
  written, and run on Study 1 as item 1 requires, before that step runs on Study 3.
- `pipeline/README.md` and `analysis/q24` note that Amendment 7 adopted the line-based source.
- `00_dissertation_outline.md` §6.1.3 (reference standard) and §4.2 describe the source as in
  item 1.

---

### Amendment 8, 2026-10-12. What "the existing utterance boundaries" means for Study 3, and how they are fixed.

#### Inspection statement

**Written after Amendment 7 and before any Study 3 content was inspected.** Since Amendment 7 was
appended, the following was done with the Study 3 clips, and nothing else.

- **Transfer, 2026-10-11, all after Amendment 7 was appended (`6244f32`, 19:07 UTC+8).** The
  author ran the following commands, supplied in writing in the working conversation, by hand.
  On the author's Windows machine (PowerShell): the 216 file paths were read from
  `pipeline/server/clips_216_drawn.sha256` into a list file; `tar -cf study3_216.tar -T <list>`
  (no compression, 20:35 UTC+8); `tar -czf moco2026.tar.gz moco2026`; `scp` of both archives to
  the processing server. On the server (SSH): `tar -xf ~/study3_216.tar` and
  `tar -xzf ~/moco2026.tar.gz` (no `-v`, no output); `sha256sum -c
  pipeline/server/clips_216_drawn.sha256 | grep -v ': OK$'` (no output, i.e. all 216 OK);
  `mkdir -p work/clips work/data data/study3_runs`; `ln -s` of the 216 clips into `work/clips/`
  (20:56 UTC+8); `ls work/clips | wc -l` (216); removal of the archives. The only other checks
  suggested were counts (`(Get-Content <list>).Count`, `tar -tf study3_216.tar | Measure-Object`),
  which print a number. The author ran no other command on these files. The file times on the
  author's machine and on the server agree with these steps.
- **Checks on the server, 2026-10-12.** `sha256sum -c pipeline/server/clips_216_drawn.sha256`
  (216 OK; its output, the 216 file names and OK, was written to a file in the session's scratch
  folder and only counts were displayed); `du -sh` on `data/study3`, `data/study3/clips` and
  `work` (totals only: 12G, 12G, 884K); file counts with `ls -A | wc -l` and
  `ls data/study3/clips/*.mp4 | wc -l` (1, 216, 216); `readlink -f` on the 216 links piped into
  `grep -c`, and `find -L work/clips -type l | wc -l`, which printed counts only (216, 0);
  `git check-ignore -v work data/study3`, which printed only ignore rules; and a Python import
  probe run with `work/` as the working directory, which read nothing under `work/clips/`.
- The file names are those of `analysis/study3/manifest.json`, already disclosed in Amendment 7.

No Study 3 clip has been decoded, transcribed, segmented, read by OCR or a model, or played, and no
Study 3 output of any kind exists. Every measurement behind this amendment was made on Study 1's
30 clips, on three public files from the `pyannote-audio` and `whisper` repositories
(`tutorials/assets/sample.wav` and its reference annotation `sample.rttm`; `tests/jfk.flac`) and
on synthetic audio. The draft was committed as `1163acf`; between that commit and the commit that
appends it here nothing under `data/study3/` or `work/` on the server was processed.
**The author confirms this statement as of 2026-10-12.**

---

#### 1. The boundaries are those of one fixed run

*Item clarified.* Amendment 4 Part B2 ("The existing utterance boundaries, unchanged"), Amendment 7
item 4 ("the existing utterance boundaries"), and every other use of the phrase. For Study 1 the
phrase names files that exist. For Study 3 no boundaries exist; the phrase can only mean the
procedure that produced Study 1's.

*What it becomes.* Study 3's utterance boundaries are the `utterances` tier of the aligned trees
produced by **one run** of MOCO stages 2 to 4 (`moco2026/`, byte-identical to the copy used for
Study 1's files as far as they survive; Whisper `large-v2` with the decoding settings written in
`make_modern_pipeline.py`, unchanged), in the environment recorded in `analysis/study3/run_log.md`.
Two things are added to make that run reproducible and to stop it from being chosen:

1. **Seed.** The PyTorch random seed is set to `20261012` immediately before Whisper decodes each
   clip, by a wrapper that calls the unmodified `make_modern_pipeline.py`. Whisper's temperature
   fallback samples; without a seed, two runs on the same machine differ (item 3). The seed
   selects one draw from the same procedure and does not change the procedure.
2. **Freeze.** The first complete run is the result. It is not repeated in order to obtain
   different boundaries. An interrupted run is resumed (finished clips are skipped, and the
   per-clip seed makes the resumed clips identical to an uninterrupted run). A run that fails for a
   technical reason is repeated from the start with the same seed, and the failure is reported.

*Check before Study 3.* Two seeded runs are made on Study 1's 30 clips on the server and
compared. The result is reported whatever it is. If they are not identical, the seed is still
used and the remaining non-identity is reported as the run-to-run variation of the Study 3
boundaries.

*Result of the check (2026-10-12).* Two seeded runs (S1, S2) on Study 1's 30 clips gave
byte-identical `_whisper.json` files for all 30 clips and identical boundaries (1,938 utterances in
each; 30 of 30 clips identical; 412 segments with temperature above 0 in each). Against Study 1,
S1 agrees on 38.0% of boundaries within ±0.1 s and 60.2% within ±0.5 s, within the range of the
unseeded reruns in item 3. The seed removes run-to-run variation on this server; it does not
remove the difference from Study 1.

#### 2. Utterances that are one whole Whisper window

*Item added.* Whisper sometimes returns a whole 30-second decoding window as one segment (Study 1:
2 such segments in 30 clips; two reruns on the server: 6 and 7). They are not boundaries found in
the speech. Whether alignment keeps one varies with the environment and with the segment: Study 1's files
dropped both of its own, which the server's alignment keeps; of the server's own segments, the
alignment kept 4 of 6 in one rerun and 4 of 7 in the other.

*What it becomes.* They are kept in the primary analysis, so that the procedure is unchanged. A
**sensitivity analysis** repeats every Arm comparison and every utterance-level estimate after
removing utterances whose duration lies in [29.9 s, 30.1 s]. Their number per language is
reported. Both results are reported; the primary analysis is the one with them.

#### 3. Reproducibility of the boundaries, as measured before Study 3

*Item recorded.* Measured on Study 1's 30 clips (Study 1: 2,140 utterances, 4,280 boundaries):

| Comparison | Boundaries within ±0.1 s | within ±0.5 s | Identical clips |
|---|---|---|---|
| Alignment only: Study 1's Whisper output realigned on the server | 99.98% | 100% | 21 / 30 |
| Two unseeded Whisper runs on the server | 60.2% | 75.5% | 6 / 30 |
| Server rerun against Study 1 | 33.7% | 59.4% | 0 / 30 |

The six clips that are identical between the two server runs are exactly those in which Whisper
used no temperature fallback in either run. Four clips used no fallback in any of the three runs
and their Whisper segments still differ from Study 1's (segment counts 104 against 109, 141 against
108, 60 against 71 and 114 against 112; segment-boundary agreement within ±0.5 s 56.6% to 100%,
a segment-level figure, unlike the utterance-level table above), so part of the difference from
Study 1 is not sampling; Study 1's software versions and hardware are not recorded and the cause
is not established.

*Consequences for reporting.* (a) Study 1's boundaries are one draw of the same procedure, and so
are Study 3's. (b) Comparisons between Study 1 and Study 3 that depend on utterance counts or
boundaries (utterances per clip, utterance duration, the share of clip time inside an utterance,
the subtitle assignment error of Amendment 7) are reported with the figures above. (c) Settings
calibrated on Study 1 utterances (the pause detector of `pipeline/stages/audio_utils.py`) were
calibrated on one draw; this is stated where they are used. (d) Within Study 3, all Arms are scored
on the same boundaries, so the variation does not favour one Arm over another.

#### 4. The rebuilt environment, recorded

*Item recorded, not changed.* MOCO's setup script cannot be reproduced from today's package
indexes, and Study 1's environment was not recorded. The departures from the setup script needed
to run it, each with its reason and a check, are listed in `analysis/study3/run_log.md`
(deviations (a) to (i)). Three bear on the boundaries and are stated here:

- `alignphones/data/allosaurus_ipas.txt` is regenerated by `moco2026/build-ipa.py` (230 entries,
  `<blk>` first), the script written for Study 1's machine. The copy shipped with `moco2026/`,
  which is the one `new_make_allosaurus.py` generates from `phone.txt` when the file is missing,
  has 229 entries without `<blk>`, and with it every alignment fails silently.
- `make_modern_pipeline.py` line 77 raises a TypeError under pyannote.audio 3.3.2 after Whisper
  has finished. It is not modified: diarization feeds only speaker labels, which are empty in all
  2,145 of Study 1's utterance records.
- The path mismatch between `new_make_align_phones.py` and `make_modern_pipeline.py` is not
  repaired, for the same reason.

#### 5. Not changed

The definition of the utterance as the retrieval and labeling unit; Whisper's model and decoding
settings; the alignment programs; everything Amendments 1 to 7 fix.

---

#### Consequences outside this document

- `pipeline/server/RUNBOOK.md` step 6 runs stage 4.2 through the seeding wrapper and records the
  seed in the provenance file.
- `00_dissertation_outline.md` §4.3 and §7.4 report item 3.

---

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
