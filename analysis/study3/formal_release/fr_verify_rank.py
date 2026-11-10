"""ULFR-v1 stage V1: verify formal rankings and deterministic products (read-only; no reranking).

1. Registered programs: run A JSON == unmodified-program JSON == run B JSON (bytes); same calls,
   block hashes and per-query hashes in A and B.
2. Supplement/exploratory: run A == run B (conditions, block hashes, all NPZ arrays, JSON outputs);
   run A == reproduction run1 for every shared NPZ array and JSON output; model data vs Windows.
3. Cross-program identity: the supplement's main/sensitivity conditions use exactly the registered
   program's ranking blocks.
4. Every stored block, read back from disk: payload hash, exclusion rule and candidate set,
   non-increasing scores with stable bank-order ties, top-10 view; per-query hit/chance/P@3/AP/n
   recomputed from the stored ranking and compared with each program's own per-query output.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import csv, json, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fr_common as fc  # noqa: E402

REG = fc.DATA / 'registered'
SUP = fc.DATA / 'supplement'
STORE = fc.DATA / 'rankings'
LANGS = ['ami', 'pwn', 'tao', 'tay', 'trv', 'tsu']


def meta():
    ids = np.load(fc.INPUT / 'features_full/features.npz', allow_pickle=False)['ids']
    with (fc.INPUT / 'label_analysis/units.tsv').open(encoding='utf-8') as f:
        rows = {r['unit_id']: r for r in csv.DictReader(f, delimiter='\t')}
    col = lambda k: np.array([rows[u][k] for u in ids])
    return dict(ids=ids, lang=col('lang'), day=col('day'), clip=col('clip'), haiku=col('haiku'), gemini=col('gemini'))


M = None


def init():
    global M
    M = meta()


def check_block(args):
    """Read one stored block; verify structure; return per-query recomputation keyed by block."""
    h, info = args
    z = np.load(STORE / info['file'], allow_pickle=False)
    q, bank, off, rr, ss = (z[k] for k in ('query_rows', 'bank_rows', 'offsets', 'rank_rows', 'rank_scores'))
    assert fc.BlockStore.digest(q, bank, info['rule'], off, rr, ss) == h
    rule = info['rule']
    out = dict(n=np.diff(off), problems=0)
    pos = {int(c): j for j, c in enumerate(bank)}
    for j, i in enumerate(q):
        r = rr[off[j]:off[j + 1]]; s = ss[off[j]:off[j + 1]]
        if 's62' in rule:
            expected = set(map(int, bank))
            if 'exclude_own_clip' in rule:
                expected = {c for c in expected if M['clip'][c] != M['clip'][i]}
        else:
            expected = {int(c) for c in bank if c != i}
            if rule.endswith('same_clip'):
                expected = {c for c in expected if M['clip'][c] != M['clip'][i]}
            elif 'same_day' in rule:
                expected = {c for c in expected if M['day'][c] != M['day'][i]}
        ok = set(map(int, r)) == expected and len(r) == len(expected)
        ok &= bool(np.all(np.diff(s.astype(np.float64)) <= 0))
        if len(r) > 1:  # stable ties: equal scores keep candidate-bank order
            eq = np.flatnonzero(s[1:] == s[:-1])
            ok &= all(pos[int(r[k])] < pos[int(r[k + 1])] for k in eq)
        if not ok:
            out['problems'] += 1
    t = np.load(STORE / info['topk_file'], allow_pickle=False)
    for j in range(len(q)):
        m = min(10, int(off[j + 1] - off[j]))
        assert np.array_equal(t['top_rows'][j, :m], rr[off[j]:off[j] + m])
        assert np.array_equal(t['top_scores'][j, :m], ss[off[j]:off[j] + m])
    return h, out


def metrics_from_block(h, info, labels, q_expected=None):
    z = np.load(STORE / info['file'], allow_pickle=False)
    q, off, rr = z['query_rows'], z['offsets'], z['rank_rows']
    if q_expected is not None:
        assert np.array_equal(q, q_expected)
    n = len(q); res = {k: np.full(n, np.nan) for k in ('hit', 'chance', 'p3', 'ap')}
    res['n_cand'] = np.diff(off); res['n_rel'] = np.zeros(n, np.int64)
    for j, i in enumerate(q):
        r = rr[off[j]:off[j + 1]]
        if not len(r):
            continue
        rel = labels[r] == labels[i]; nr = int(rel.sum()); k = len(r)
        res['n_rel'][j] = nr; res['hit'][j] = float(rel[0]); res['chance'][j] = nr / k
        res['p3'][j] = rel[:min(3, k)].mean()
        if nr:
            res['ap'][j] = float((np.cumsum(rel) / np.arange(1, k + 1))[rel].mean())
    return q, res


def same(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def npz_equal(x, y):
    a, b = np.load(x, allow_pickle=False), np.load(y, allow_pickle=False)
    if sorted(a.files) != sorted(b.files):
        return False
    for k in a.files:
        u, v = a[k], b[k]
        if u.dtype != v.dtype or u.shape != v.shape:
            return False
        if u.dtype.kind in 'US' or not np.array_equal(u, v, equal_nan=u.dtype.kind == 'f'):
            if not np.array_equal(u, v):
                return False
    return True


def main():
    start = time.perf_counter(); fc.check_frozen(); init()
    report = dict(release=fc.RELEASE_ID)
    # 1. registered programs
    ja, jb, jv = (REG / x / 'json' for x in ('runA', 'runB', 'verbatim'))
    names = sorted(p.name for p in ja.glob('*.json'))
    assert names == sorted(p.name for p in jv.glob('*.json')) == sorted(p.name for p in jb.glob('*.json'))
    report['registered_json'] = {n: dict(sha256=fc.sha(ja / n), verbatim_equal=fc.sha(ja / n) == fc.sha(jv / n),
                                         runB_equal=fc.sha(ja / n) == fc.sha(jb / n)) for n in names}
    ca, cb = fc.read(REG / 'runA/calls.json'), fc.read(REG / 'runB/calls.json')
    strip = lambda c: {k: c[k] for k in ('seq', 'program', 'condition', 'rule', 'block', 'queries', 'entries', 'representation_sha256', 'perquery_sha256')}
    report['registered_calls'] = dict(calls=len(ca), runB_identical=[strip(c) for c in ca] == [strip(c) for c in cb],
                                      unique_blocks=len({c['block'] for c in ca}))
    # perquery files in A must match their recorded hashes
    bad = 0
    for c in ca:
        z = np.load(fc.DATA / 'rankings' / c['perquery_file'], allow_pickle=False)
        bad += any(fc.arrhash(z[k]) != v for k, v in c['perquery_sha256'].items())
    report['registered_perquery_files_hash_mismatch'] = bad
    # 2. supplement
    sa, sb = SUP / 'runA', SUP / 'runB'
    conds_a, conds_b = fc.read(sa / 'conditions.json'), fc.read(sb / 'conditions.json')
    report['supplement_conditions'] = dict(conditions=len(conds_a), runB_identical=conds_a == conds_b)
    npz_a = sorted(p.relative_to(sa) for p in sa.rglob('*.npz'))
    npz_b = sorted(p.relative_to(sb) for p in sb.rglob('*.npz'))
    assert npz_a == npz_b
    report['supplement_npz_runA_runB'] = dict(files=len(npz_a), all_arrays_equal=all(npz_equal(sa / r, sb / r) for r in npz_a))
    run1 = fc.REPRO_DATA / 'run1'
    shared = [r for r in npz_a if (run1 / r).exists()]
    report['supplement_npz_vs_reproduction_run1'] = dict(shared_files=len(shared), formal_only=[str(r) for r in npz_a if not (run1 / r).exists()],
                                                         run1_only=[str(p.relative_to(run1)) for p in run1.rglob('*.npz') if not (sa / p.relative_to(run1)).exists()],
                                                         all_arrays_equal=all(npz_equal(sa / r, run1 / r) for r in shared),
                                                         unequal=[str(r) for r in shared if not npz_equal(sa / r, run1 / r)])
    js = {}
    for rel in ['old_metrics.json', 'execution_reports/metrics.json', 'execution_reports/paired.json', 'execution_reports/joint_family.json',
                'old_manifest.json', 'execution/ranking_manifest.json']:
        a, b, c = fc.read(sa / rel), fc.read(sb / rel), fc.read(run1 / rel)
        if rel.endswith('joint_family.json'):
            for x in (a, b, c):
                x.pop('elapsed_statistics_seconds', None)
        js[rel] = dict(runB_equal=a == b, run1_equal=a == c)
    report['supplement_json'] = js
    # model data vs Windows reference and vs reproduction W3
    md = []
    for f in sorted((sa / 'execution').glob('*_model_data.npz')):
        a = np.load(f, allow_pickle=False); w = np.load(fc.REFDATA / 'execution_20261020' / f.name, allow_pickle=False)
        row = dict(model=f.name.replace('_model_data.npz', ''), formal_sha256=fc.sha(f), windows_sha256=fc.sha(fc.REFDATA / 'execution_20261020' / f.name))
        assert sorted(a.files) == sorted(w.files)
        diffs = {}
        for k in a.files:
            u, v = a[k], w[k]
            if u.shape != v.shape:
                diffs[k] = 'shape'; continue
            if u.dtype.kind in 'iub' and v.dtype.kind in 'iub':
                if not np.array_equal(u.astype(np.int64), v.astype(np.int64)):
                    diffs[k] = int(np.sum(u.astype(np.int64) != v.astype(np.int64)))
            elif not np.array_equal(u, v, equal_nan=True):
                d = np.abs(np.asarray(u, float) - np.asarray(v, float)); d = d[np.isfinite(d)]
                diffs[k] = dict(differing=int(np.sum(~np.isclose(u, v, rtol=0, atol=0, equal_nan=True))), max_abs=float(d.max()) if len(d) else None)
        row['array_differences_vs_windows'] = diffs
        row['integer_dtype_only_storage_differences'] = sorted(k for k in a.files if a[k].dtype != w[k].dtype and k not in diffs)
        r1 = run1 / 'execution' / f.name
        row['vs_reproduction_run1'] = npz_equal(f, r1) if r1.exists() else 'not built in reproduction'
        md.append(row)
    report['model_data'] = md
    # 3. cross-program identity
    reg_blocks = defaultdict(dict)
    for c in ca:
        if c['program'] in ('ch5_retrieval', 'ch5_sensitivity') and c.get('kind') != 'diagnostic':
            prog, kind, rep, ver, sub, mode, lang = c['condition'].split('|')
            scope = {'analysis': 'main', 'B2_centered': 'centered', 'no_duration_band': 'no_duration_band', 'no_input_gap': 'no_input_gap'}[kind]
            reg_blocks[f'{scope}|{rep}|{ver}|{sub}|{mode}'][lang] = c['block']
    cross = []
    for c in conds_a:
        if c['suite'] != 'old':
            continue
        mine = {b['source']: b['block'] for b in c['blocks']}
        theirs = reg_blocks.get(c['condition'])
        cross.append(dict(condition=c['condition'], registered_found=theirs is not None, identical=theirs == mine))
    report['cross_program_identity'] = dict(conditions=len(cross), identical=sum(x['identical'] for x in cross),
                                            not_identical=[x for x in cross if not x['identical']])
    # 4. every stored block
    index = {}
    for f in [REG / 'runA/blocks.json', SUP / 'runA/blocks.json']:
        for h, v in fc.read(f).items():
            if h in index:
                assert index[h]['sha256'] == v['sha256']
            index[h] = v
    for h, v in index.items():
        assert fc.sha(STORE / v['file']) == v['sha256']
    with ProcessPoolExecutor(16, initializer=init) as pool:
        structural = dict(pool.map(check_block, list(index.items()), chunksize=4))
    report['blocks'] = dict(unique_blocks=len(index), entries=sum(v['entries'] for v in index.values()),
                            payload_bytes=sum(v['bytes'] for v in index.values()),
                            queries_with_structure_problems=sum(v['problems'] for v in structural.values()))
    # per-query recomputation from stored rankings
    lab = {'haiku': M['haiku'], 'gemini': M['gemini']}
    mism = defaultdict(int); checked = 0
    for c in ca:
        z = np.load(fc.DATA / 'rankings' / c['perquery_file'], allow_pickle=False)
        labels = lab['gemini'] if 'gemini' in c['condition'] else lab['haiku']
        q, res = metrics_from_block(c['block'], index[c['block']], labels, z['query_rows'])
        if c['program'] in ('ch5_retrieval', 'ch5_sensitivity'):
            res['n_cand'] = np.where(res['n_cand'] == 0, np.nan, res['n_cand'])  # program stores NaN when empty
            for k in ('hit', 'chance', 'p3', 'ap', 'n_cand'):
                mism[k] += not same(res[k], z[k])
        elif 'item8' not in c['condition']:
            # pair_hits output: only queries with candidates, in query order
            valid = res['n_cand'] > 0
            mism['s62_hit'] += not same(res['hit'][valid], z['hit'])
            mism['s62_chance'] += not same(res['chance'][valid], z['chance'])
            mism['s62_lift'] += not same((res['hit'] - res['chance'])[valid], z['lift'])
        else:
            mism['item8_lift'] += not same(res['hit'] - res['chance'], z['lift'])
        checked += 1
    for c in conds_a:
        z = np.load(sa / c['metrics_file'], allow_pickle=False)
        parts = [metrics_from_block(b['block'], index[b['block']], lab['haiku']) for b in c['blocks']]
        q = np.concatenate([x[0] for x in parts]); assert np.array_equal(q, z['q'])
        for k in ('hit', 'chance', 'p3', 'ap', 'n_cand', 'n_rel'):
            mism['supplement_' + k] += not same(np.concatenate([x[1][k] for x in parts]), z[k])
        checked += 1
    report['perquery_recomputation'] = dict(program_calls_and_conditions=checked, mismatching_arrays=dict(mism))
    report['seconds'] = time.perf_counter() - start
    report['verifier_sha256'] = fc.sha(__file__)
    passed = (all(v['verbatim_equal'] and v['runB_equal'] for v in report['registered_json'].values())
              and report['registered_calls']['runB_identical'] and bad == 0
              and report['supplement_conditions']['runB_identical'] and report['supplement_npz_runA_runB']['all_arrays_equal']
              and report['supplement_npz_vs_reproduction_run1']['all_arrays_equal']
              and all(v['runB_equal'] and v['run1_equal'] for v in js.values())
              and report['blocks']['queries_with_structure_problems'] == 0 and sum(mism.values()) == 0)
    report['passed'] = passed
    fc.dump(fc.REPORT / 'verification/RANKING_VERIFICATION.json', report, exclusive=False)
    print(json.dumps({k: v for k, v in report.items() if k not in ('model_data',)}, indent=1, default=str)[:6000])
    return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
