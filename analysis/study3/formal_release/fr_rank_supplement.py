"""ULFR-v1 stage R2: the 175 metric-supplement and 210 adopted exploratory conditions, all 11 model data.

Reuses the reproduction's frozen condition definitions (CONDITIONS.json order/hashes) and the supplied
statistics/hierarchy formulas unchanged. Differences from analysis/study3/ch5_reproducibility/rank_pipeline.py:
every per-source ranking block is recorded (complete ranks/scores, content-addressed), outputs go to the
release root, and hierarchy.prefit builds all eleven model data sets (the reproduction built three).
  --run A  save blocks;  --run B  independent repeat, block hashes only.
"""
import os
for _v in ('MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse, hashlib, json, sys, time, resource, shutil
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fr_common as fc  # noqa: E402
sys.path.insert(0, str(fc.REPRO_REPORT))
import rank_pipeline as rp  # noqa: E402  (imports frozen CORE/scripts modules; no analysis at import)
base, ex, hierarchy = rp.base, rp.ex, rp.hierarchy

RULE = {'W1': 'exclude_self', 'W2': 'exclude_self_and_same_clip', 'W3': 'exclude_self_and_same_day',
        'X': 'exclude_self_other_language_bank'}
FULL = ['main|xlmr_sub|kept|own|W3', 'main|kin14|kept|own|X', 'main|whisper|kept|own|W3']


def score(X, U, idx, q, banks, pool, full, store):
    """rank_pipeline.score verbatim, plus per-source block recording."""
    keys = ['hit', 'chance', 'p3', 'ap', 'n_cand', 'n_rel']; out = {k: [] for k in keys}
    topidx = []; tops = []; fingerprints = []; maskprints = []; scoreprints = []; full_idx = []; full_scores = []; offsets = [0]
    blocks = []
    for l, c in banks.items():
        qq = q[U['lang'][q] == l]; cn = X[c]
        b_rank, b_score, b_off = [], [], [0]
        for start in range(0, len(qq), 512):
            qs = qq[start:start + 512]; S = base.cr.sims(X[qs], cn)
            assert S.dtype == np.float32
            for row, i in enumerate(qs):
                keep = c != i
                if pool == 'W2': keep &= idx['clip'][c] != idx['clip'][i]
                elif pool == 'W3': keep &= idx['day'][c] != idx['day'][i]
                ci = np.flatnonzero(keep); candidates = c[ci]
                order = ci[np.argsort(-S[row, ci], kind='stable')]
                ranked = c[order]; scores = S[row, order]; n = len(order)
                rel = idx['lab'][ranked] == idx['lab'][i]; nr = int(rel.sum())
                out['hit'].append(float(rel[0]) if n else np.nan)
                out['chance'].append(nr / n if n else np.nan)
                out['p3'].append(float(rel[:min(3, n)].mean()) if n else np.nan)
                out['ap'].append(float((np.cumsum(rel) / np.arange(1, n + 1))[rel].mean()) if nr else np.nan)
                out['n_cand'].append(n); out['n_rel'].append(nr)
                ti = np.full(3, -1, np.int64); ts = np.full(3, np.nan, np.float32)
                ti[:min(3, n)] = ranked[:3]; ts[:min(3, n)] = scores[:3]
                topidx.append(ti); tops.append(ts); fingerprints.append(fc.arrhash(ranked)); scoreprints.append(fc.arrhash(scores)); maskprints.append(fc.arrhash(candidates))
                if full: full_idx.append(ranked); full_scores.append(scores); offsets.append(offsets[-1] + n)
                b_rank.append(ranked); b_score.append(scores); b_off.append(b_off[-1] + n)
        rr = np.concatenate(b_rank) if b_rank else np.zeros(0, np.int64)
        ss = np.concatenate(b_score) if b_score else np.zeros(0, np.float32)
        h = store.put(qq, c, 'registered_float32_cosine_stable_desc:' + RULE[pool], np.array(b_off), rr, ss)
        blocks.append(dict(source=l, block=h, queries=int(len(qq)), entries=int(len(rr))))
    r = {k: np.array(v, np.int64 if k in ['n_cand', 'n_rel'] else float) for k, v in out.items()}
    r['lift1'] = r['hit'] - r['chance']
    proof = {'top3_indices': np.array(topidx), 'top3_scores': np.array(tops), 'rank_sha256': np.array(fingerprints),
             'score_sha256': np.array(scoreprints), 'post_exclusion_sha256': np.array(maskprints), 'ap_defined': np.isfinite(r['ap'])}
    if full: proof.update(full_rank=np.concatenate(full_idx), full_scores=np.concatenate(full_scores), full_offsets=np.array(offsets))
    return r, proof, blocks


def run(name):
    fc.check_frozen()
    dest = fc.DATA / 'supplement' / f'run{name}'; dest.mkdir(parents=True, exist_ok=False)
    cache = dest / 'execution'; cache.mkdir(); out = dest / 'execution_reports'; out.mkdir()
    store = fc.BlockStore(fc.DATA / 'rankings', 'write' if name == 'A' else 'hash')
    ids, U, idx, reps, old, new, hashes = rp.inputs()
    ex.REPORT = fc.REF; ex.OUT = out; ex.CACHE = cache
    spec = fc.read(fc.REPRO_REPORT / 'CONDITIONS.json'); assert spec['inputs'] == hashes
    pools = fc.threads_one()
    norm = {}; normhash = {}; manifest = {'cells': {}}; old_manifest = {'cells': {}}; oldrows = []; weights = {}; indchecks = []
    conditions = []
    desc = rp.descriptors(old, new, U); start = time.perf_counter()
    for ii, (suite, key, rep, center, pool, q, banks) in enumerate(desc):
        planned = spec['conditions'][ii]
        assert planned['condition'] == key and planned['suite'] == suite
        assert fc.arrhash(q) == planned['ordered_query_sha256']
        assert {l: fc.arrhash(c) for l, c in banks.items()} == planned['candidate_sha256']
        nk = (rep, center)
        if nk not in norm:
            X, ok, _ = reps[rep]
            if center:
                Y = np.zeros_like(X, dtype=np.float64)
                for l in ex.SOURCES:
                    m = (U['lang'] == l) & ok; Y[m] = X[m] - X[m].mean(0, keepdims=True)
                X = Y.astype(np.float32)
            norm[nk] = base.cr.l2rows(X); normhash[f'{rep}|centered={center}'] = fc.matrix_hash(norm[nk])
        full = suite == 'old' and key in FULL
        r, proof, blocks = score(norm[nk], U, idx, q, banks, pool, full, store)
        filename = hashlib.sha256(key.encode()).hexdigest()[:20] + '.npz'
        folder = dest / 'old' if suite == 'old' else cache; folder.mkdir(exist_ok=True)
        path = folder / filename
        np.savez_compressed(path, q=q, **r, **proof)
        if full:
            rp.independent(q, r, proof, U, idx, pool, banks); indchecks.append(key)
            if rep != 'whisper': np.savez_compressed(folder / (filename[:-4] + '_normalized.npz'), normalized=norm[nk], ids=ids)
        item = {'file': filename, 'sha256': fc.sha(path), 'query_sha256': fc.arrhash(q), 'candidate_sha256': {l: fc.arrhash(c) for l, c in banks.items()},
                'rank_fingerprints_sha256': fc.arrhash(proof['rank_sha256']), 'score_fingerprints_sha256': fc.arrhash(proof['score_sha256'])}
        conditions.append(dict(suite=suite, condition=key, representation=rep, centered=center, pool=pool,
                               normalization=f'{rep}|centered={center}', rule=RULE[pool], queries=int(len(q)),
                               metrics_file=str(path.relative_to(dest)), blocks=blocks,
                               rank_fingerprints_sha256=item['rank_fingerprints_sha256'],
                               score_fingerprints_sha256=item['score_fingerprints_sha256'],
                               perquery_sha256={k: fc.arrhash(v) for k, v in r.items()}))
        if suite == 'new': manifest['cells'][key] = item
        else:
            old_manifest['cells'][key] = item
            task = old[ii]; universe = tuple(sorted(set(zip(U['lang'][q], U['day'][q]))))
            if universe not in weights:
                weights[universe] = base.cluster_weights(U['lang'][q], U['day'][q])[0]
                np.savez_compressed(folder / ('weights_' + hashlib.sha256(str(universe).encode()).hexdigest()[:16] + '.npz'), weights=weights[universe], day_keys=np.array(universe))
            dp = {c: j for j, c in enumerate(universe)}; di = np.array([dp[c] for c in zip(U['lang'][q], U['day'][q])])
            oldrows.extend(base.aggregate(q, r, U, task, weights[universe], di))
        print('RANKED', name, ii + 1, len(desc), key, flush=True)
    fc.dump(dest / 'old_manifest.json', old_manifest); fc.dump(cache / 'ranking_manifest.json', manifest)
    fc.dump(dest / 'old_metrics.json', {'seed': 20261020, 'rows': oldrows})
    pf = fc.read(fc.REF / 'execution_20261020/preflight.json')
    fc.dump(out / 'preflight.json', {'day_keys': pf['day_keys'], 'seed': 20261024})
    shutil.copyfile(fc.REF / 'execution_20261020/adopted_overlap_weights.csv', out / 'adopted_overlap_weights.csv')
    t0 = time.perf_counter(); ex.statistics(fc.INPUT); stats_seconds = time.perf_counter() - t0
    t0 = time.perf_counter(); hierarchy.prefit(fc.INPUT); prefit_seconds = time.perf_counter() - t0
    fc.dump(dest / 'conditions.json', conditions)
    fc.dump(dest / 'blocks.json', store.index)
    fc.dump(dest / 'run_complete.json', dict(release=fc.RELEASE_ID, run=name, conditions=len(desc),
        ranking_seconds_including_old_metrics=t0 - start - stats_seconds, statistics_seconds=stats_seconds, prefit_seconds=prefit_seconds,
        unique_blocks=len(store.index), normalization_sha256=normhash, independent_checks=indchecks,
        loaded_source_sha256={str(m.__file__): fc.sha(m.__file__) for m in (rp, base, ex, hierarchy, base.cr)},
        runner_sha256=fc.sha(__file__), common_sha256=fc.sha(HERE / 'fr_common.py'),
        conditions_spec_sha256=fc.sha(fc.REPRO_REPORT / 'CONDITIONS.json'), spec_sha256=fc.sha(fc.REPORT / 'FORMAL_RELEASE_SPEC.md'),
        threadpools=pools, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, environment=fc.environment_record()))
    print('SUPPLEMENT COMPLETE', name, flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--run', choices=['A', 'B'], required=True)
    run(a.parse_args().run)
