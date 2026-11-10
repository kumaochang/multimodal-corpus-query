# Independent numerical reproduction



## Deterministic computation

Rank all 175 historical supplement conditions and all 210 adopted exploratory conditions,
without selecting by agreement. Do not deduplicate in this implementation: every ID gets
its own new computation. Complete condition IDs are frozen in CONDITIONS.json before rank.
Two fresh processes, independent output directories run1/run2, each start from frozen
features; neither reads previous/new ranking results as computational inputs.

Use retained item5 Python 3.10.22 / NumPy 1.26.4 / SciPy 1.13.1, dense NumPy OpenBLAS
0.3.23.dev and SciPy OpenBLAS 0.3.27, one thread for BLAS/OMP. This is a new named Linux
baseline, not a Windows MKL match. CPU dispatch and binary hashes are recorded. No backend
search. Original l2rows, float32 cosine, CHUNK=512 and stable descending argsort; preserve
full-reference normalization, language-centered sensitivities, all masks, source-sorted
queries and feature-row candidates. Frozen TF-IDF only. No new transcription X/common or Face.

Retain per-query metrics/counts/AP masks, ordered IDs/banks, post-exclusion order and complete
ranking fingerprints, top-3 IDs/scores. Save full scores/ranks and normalized matrices for
subtitle kept/own W3 and kin14 kept/own X; also save full Whisper kept/own W3 for independent
Python stable-sort and scalar-metric checks on actual produced scores. Other full similarities
are not retained. The 30 GiB initial storage budget comfortably exceeds this bounded output;
stop a stage rather than exhaust disk. No unbounded similarity-matrix accumulation.

Historical supplement: PCG64 seed 20261020, 10,000 within-source day draws, exact existing
group definitions and query-weighted ratio of day sums. Adopted A/B: PCG64 seed 20261024,
10,000 shared source/day weights, fixed adopted overlap cells/weights, all 24 coordinates,
ddof=1 and 95% max-standardized-deviation quantile; joint-valid threshold 99%.
Weights are newly generated, then compared with available reference multiplicities.
Undefined values stay undefined; candidates/ranks fixed during bootstrap. No MAP chance.

## Posterior scope and resources

Only H0_W3 attempt_1 (2000/2000), HA_W3 attempt_5 (4000/4000), HB_W3 attempt_1 (2000/2000)
are references. Canonical frozen NPZ and converted JSON, exact recorded Stan source,
priors/offsets/structural rows/encoding/random effects are retained. HA keeps its supplied
warm-start states, inverse metrics, per-chain stepsizes, adaptation and chain IDs.
New independent seeds are respectively 202610311, 202610312, 202610313.

Isolated posterior venv: Python 3.12.0, CmdStanPy 1.3.0, CmdStan 2.37.0, NumPy 2.2.6,
SciPy 1.15.3, ArviZ 0.22.0, pandas 2.3.3, psutil 7.1.3. Exact installed transitive lock and
compiler/build flags are retained; original item5 and all other environments are untouched.
Compile -O3/stanc --O1. Original and native-events implementations are validated at nonzero
parameters, likelihood/prior/Jacobian and all gradients, before adopting either for sampling.

First run H0_W3 with four independent chains, one CPU thread each, and measure wall/RSS/disk.
Only after this benchmark may two model fits run concurrently (maximum eight sampler CPUs).
No queue for the historical 68 fits. Do not reuse local posterior as new draws, append partial
chains, or mix attempts. If actual reranked model-input arrays differ, at most one additional
fit per affected selected model, seeds 202610411/202610412/202610413, retaining all original
settings except the independent seed and explicit new data version.

At most one repair per model name across this task, only for sampler/estimand diagnostics or
MCSE consistency warnings: double warmup/retained, adapt_delta .99 (HA .995), treedepth 14
(HA 15), same data/model/prior/support; record independent repair seed base+1000. Retain failures.
Do not keep sampling until agreement. Infrastructure/compilation errors are not posterior
attempts; diagnose and retain their logs before retrying unchanged scientific settings.

## Fixed posterior comparisons

POSTERIOR_COORDINATES.json fixes all eligible source-pooled margins (including speech-pooled
M/I margins) and existing contrasts from the selected attempts' metadata before new draws.
Exclude only recorded sparse-O7 or no-likelihood prediction coordinates, never by effects.
Compare conditional and population-average probability/lift, and fixed paired/DID contrasts;
means and 2.5/97.5 percentiles. Keep empirical support/weights and GH40 with GH80 check.
Do not use logistic(mean logit). Raw chains are verified; reconstruct local estimand series
from those selected chains and check supplied summaries/draw caches where available.

For each fixed coordinate/statistic estimate mean/quantile MCSE independently on both chains,
chain-aware ArviZ MCSE. Define D=abs(local-server)/sqrt(MCSE_local^2+MCSE_server^2), warning
threshold normal quantile 1-.01/(2*m), where m is frozen across all three same-input models.
This is approximate multi-coordinate computation consistency, not an effect test or the
frequentist family of 24. Constants compare exactly; unavailable MCSE means indeterminate.
Check every parameter and all reporting estimands for Rhat<=1.01, bulk/tail ESS>=400,
zero unresolved divergences/treedepth saturation, E-BFMI>.3 and chain mixing.

Retain PPC misfit, high Pareto k, rank deficiency, missing cells, prior dependence,
shared-subtitle evidence, source/speech/framing confounds and unvalidated model functionality.
Convergence does not resolve these limitations. Six separate status layers, never one global PASS.
