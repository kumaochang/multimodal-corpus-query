"""ULFR-v1 stage R1: the four registered utterance-level programs in the fixed ranking environment.

  --run A     execute ch5_retrieval -> s62_lift -> s62_by_label -> ch5_sensitivity in one process with
              recording wrappers; save every ranking block, top-10 view and per-query outputs.
  --run B     independent fresh process, same computation; hashes only (streaming repeat check).
  --verbatim  the unmodified original scripts as subprocesses; their JSON must equal run A byte for byte.

Wrappers are verbatim copies of Retriever.run, pair_hits and item8 plus recording lines; recording
never feeds back into the computation. Inputs are the frozen files; nothing upstream is rerun.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse, contextlib, json, subprocess, sys, time, resource
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_common import *  # noqa: E402

PROGRAMS = ['ch5_retrieval', 's62_lift', 's62_by_label', 'ch5_sensitivity']
LANGS_SORTED = ['ami', 'pwn', 'tao', 'tay', 'trv', 'tsu']
S62_LIFT_SEQ = ['primary_14d_tight', 'secondary_18d_tight', 'secondary_14d_resid', 'secondary_18d_resid',
                'secondary_9d_gesture', 'secondary_18d_full', 'sensitivity_kept_mixed',
                'sensitivity_no_mandarin_exclusion', 'sensitivity_no_input_gap', 'sensitivity_gemini_labels_exploratory']
S62_ITEM8_SEQ = ['item8', 'item8_same_clip_included']
S62_BYLABEL_SEQ = ['by_label_14d_tight', 'by_label_14d_resid']


def argv_for(program, out):
    F, Un, K = INPUT / 'features_full', INPUT / 'label_analysis/units.tsv', INPUT / 'unit_table/units_k.npz'
    if program == 'ch5_retrieval':
        return ['--features', F, '--units', Un, '--units-k', K, '--out', out / 'ch5_retrieval.json']
    if program == 's62_lift':
        return ['--units', Un, '--units-k', K, '--trees', TREES, '--pose-dirs', POSE, '--out', out]
    if program == 's62_by_label':
        return ['--units', Un, '--units-k', K, '--trees', TREES, '--pose-dirs', POSE, '--out', out / 's62_by_label.json']
    return ['--features', F, '--units', Un, '--units-k', K, '--trees', TREES, '--s62', out / 's62_results.json',
            '--main', out / 'ch5_retrieval.json', '--out', out / 'ch5_sensitivity.json']


class Recorder:
    def __init__(self, store, root, save):
        self.store, self.root, self.save = store, Path(root), save
        self.calls, self.names = [], set()
        ids = np.load(INPUT / 'features_full/features.npz', allow_pickle=False)['ids']
        self.global_row = {u: i for i, u in enumerate(ids)}
        self.program = None
        self.seq = {'lift': 0, 'item8': 0, 'bylabel': 0}
        self.context = None

    def record(self, condition, rule, query_rows, bank_rows, ranked, scores, offsets, perquery, rep_hash, extra):
        assert condition not in self.names, condition
        self.names.add(condition)
        rr = np.concatenate(ranked).astype(np.int32) if ranked else np.zeros(0, np.int32)
        dtype = scores[0].dtype if scores else np.float32
        ss = np.concatenate(scores) if scores else np.zeros(0, dtype)
        h = self.store.put(np.asarray(query_rows), np.asarray(bank_rows), rule, np.asarray(offsets), rr, ss)
        seq = len(self.calls)
        pq = {k: np.asarray(v) for k, v in perquery.items()}
        if self.save:
            folder = self.root / 'perquery' / self.program
            folder.mkdir(parents=True, exist_ok=True)
            save_npz(folder / f'{seq:05d}.npz', pq)
        self.calls.append(dict(seq=seq, program=self.program, condition=condition, rule=rule, block=h,
                               queries=int(len(query_rows)), entries=int(len(rr)), representation_sha256=rep_hash,
                               perquery_file=f'perquery/{self.program}/{seq:05d}.npz' if self.save else None,
                               perquery_sha256={k: arrhash(v) for k, v in pq.items()}, **extra))


REC = None


def install(cr, S, SBL):
    """Recording copies of the three ranking functions; original arithmetic retained line for line."""
    original_lift, original_bylabel = S.lift_analysis, SBL.analyse

    def run(self, q, cand, mode):
        out = {k: [] for k in ("hit", "chance", "p3", "p3c", "ap", "has_rel", "top_clip", "top_day",
                               "top_lang", "c_clip", "c_day", "c_lang", "n_cand")}
        ranked, scored, offsets = [], [], [0]
        Cn = self.X[cand]
        for s in range(0, len(q), cr.CHUNK):
            qs = q[s:s + cr.CHUNK]
            S_ = cr.sims(self.X[qs], Cn)
            for r, i in enumerate(qs):
                m = cand != i
                if mode == "W2":
                    m &= self.clip[cand] != self.clip[i]
                elif mode in ("W3", "DIAG_LANG"):
                    m &= self.day[cand] != self.day[i]
                idx = np.nonzero(m)[0]
                if len(idx) == 0:
                    for k in out: out[k].append(np.nan)
                    offsets.append(offsets[-1])
                    continue
                sc = S_[r, idx]
                order = idx[np.argsort(-sc, kind="stable")]
                ranked.append(cand[order]); scored.append(S_[r, order]); offsets.append(offsets[-1] + len(order))
                rel = (self.lab[cand[order]] == self.lab[i])
                nrel = rel.sum(); n = len(order)
                out["hit"].append(float(rel[0])); out["chance"].append(nrel / n)
                k3 = min(3, n); out["p3"].append(rel[:k3].mean()); out["p3c"].append(nrel / n)
                if nrel:
                    prec = np.cumsum(rel) / np.arange(1, n + 1)
                    out["ap"].append(float(prec[rel].mean()))
                else:
                    out["ap"].append(np.nan)
                out["has_rel"].append(float(nrel > 0))
                t = cand[order[0]]; pool = cand[idx]
                out["top_clip"].append(float(self.clip[t] == self.clip[i]))
                out["top_day"].append(float(self.day[t] == self.day[i]))
                out["top_lang"].append(float(self.lang[t] == self.lang[i]))
                out["c_clip"].append(float((self.clip[pool] == self.clip[i]).mean()))
                out["c_day"].append(float((self.day[pool] == self.day[i]).mean()))
                out["c_lang"].append(float((self.lang[pool] == self.lang[i]).mean()))
                out["n_cand"].append(n)
        result = {k: np.array(v, float) for k, v in out.items()}
        # ---- recording only ----
        if not hasattr(self, '_fr_hash'):
            self._fr_hash = matrix_hash(self.X)
        frames, f = [], sys._getframe(1)
        while f is not None:
            frames.append(f); f = f.f_back
        caller = frames[0].f_code.co_name
        main = next(fr for fr in frames if fr.f_code.co_name == 'main'
                    and Path(fr.f_code.co_filename).name in ('ch5_retrieval.py', 'ch5_sensitivity.py'))
        L = main.f_locals
        langs = sorted(set(self.lang[q].tolist()))
        lang = 'ALL' if mode == 'DIAG_LANG' else LANGS_SORTED[langs[0]]
        assert mode == 'DIAG_LANG' or len(langs) == 1
        if Path(main.f_code.co_filename).name == 'ch5_retrieval.py':
            kind = 'analysis' if caller == 'main' else 'diagnostic'
            name, version = L['name'], L['version']
            subset = L['subset'] if caller == 'main' else 'own'
        else:
            res = L['res']
            kind = ('B2_centered' if '1_am4_b2_centered_xlmr' not in res else
                    'no_duration_band' if 'results' not in res.get('2_am8_item2', {}) else 'no_input_gap')
            name, version, subset = L['name'], 'kept', 'own'
        rule = {'W1': 'exclude_self', 'W2': 'exclude_self_and_same_clip', 'W3': 'exclude_self_and_same_day',
                'X': 'exclude_self_other_language_bank', 'DIAG_LANG': 'exclude_self_and_same_day_all_language_bank'}[mode]
        REC.record(f'{REC.program}|{kind}|{name}|{version}|{subset}|{mode}|{lang}', 'registered_float32_cosine_stable_desc:' + rule,
                   q, cand, ranked, scored, offsets, dict(query_rows=q, **result), self._fr_hash,
                   dict(kind=kind, representation=name, version=version, subset=subset, mode=mode, source=lang,
                        score_dtype=str(scored[0].dtype) if scored else None, chunk=cr.CHUNK))
        return result

    def pair_hits(E, rows, q, t, exclude_own_clip=False, targets=None):
        qi = [i for i, r in enumerate(rows) if r["lang"] == q]
        ti = targets if targets is not None else [i for i, r in enumerate(rows) if r["lang"] == t]
        En = S.unit_norm(E)
        sim = En[qi] @ En[ti].T
        out = []
        tl = [rows[j]["label"] for j in ti]; tc = np.array([rows[j]["clip"] for j in ti])
        tg = np.array([REC.global_row[rows[j]['unit_id']] for j in ti], np.int64)
        ranked, scored, offsets, qrows = [], [], [0], []
        for a, i in enumerate(qi):
            s = sim[a].copy()
            mask = np.ones(len(ti), bool)
            if exclude_own_clip:
                mask = tc != rows[i]["clip"]
                s[~mask] = -np.inf
            n = mask.sum()
            qrows.append(REC.global_row[rows[i]['unit_id']])
            if n == 0:
                offsets.append(offsets[-1])
                continue
            j = int(np.argmax(s))
            valid = np.flatnonzero(mask); order = valid[np.argsort(-s[valid], kind='stable')]
            assert order[0] == j, 'stable descending first must equal argmax'
            ranked.append(tg[order]); scored.append(s[order]); offsets.append(offsets[-1] + len(order))
            lab = rows[i]["label"]
            chance = sum(1 for k in range(len(ti)) if mask[k] and tl[k] == lab) / n
            out.append((i, float(tl[j] == lab) - chance, float(tl[j] == lab), chance))
        context = REC.context
        REC.record(f'{REC.program}|{context}|{q}->{t}', 's62_float64_cosine_argmax_first' + ('_exclude_own_clip' if exclude_own_clip else ''),
                   qrows, tg, ranked, scored, offsets,
                   dict(query_rows=np.array(qrows, np.int64),
                        out_rows=np.array([REC.global_row[rows[o[0]]['unit_id']] for o in out], np.int64),
                        lift=np.array([o[1] for o in out]), hit=np.array([o[2] for o in out]), chance=np.array([o[3] for o in out])),
                   arrhash(E), dict(kind=context, source=q, target=t, score_dtype=str(sim.dtype),
                                    rows_order_sha256=arrhash(np.array([REC.global_row[r['unit_id']] for r in rows], np.int64))))
        return out

    def item8(E, rows, rng, include_same_clip=False):
        tl = {t: i for i, t in enumerate(S.LANGS)}
        tgt = {c: [i for i, r in enumerate(rows) if r["role"] == "t" and r["lang"] == c] for c in S.LANGS}
        En = S.unit_norm(E)
        per_lang, n_q = {}, {}
        context = S62_ITEM8_SEQ[REC.seq['item8']]; REC.seq['item8'] += 1
        for c in S.LANGS:
            qi = [i for i, r in enumerate(rows) if r["role"] == "q" and r["lang"] == c]
            n_q[c] = len(qi)
            if len(qi) < S.MIN_MANDARIN:
                continue
            M = np.full((len(qi), len(S.LANGS)), np.nan)
            blocks = {t: dict(ranked=[], scored=[], offsets=[0], qrows=[], lift=np.full(len(qi), np.nan)) for t in S.LANGS}
            for a, i in enumerate(qi):
                for t in S.LANGS:
                    ti = tgt[t]
                    if not ti:
                        continue
                    blk = blocks[t]; blk['qrows'].append(REC.global_row[rows[i]['unit_id']])
                    s = En[i] @ En[ti].T
                    mask = np.ones(len(ti), bool)
                    if t == c and not include_same_clip:
                        mask = np.array([rows[j]["clip"] != rows[i]["clip"] for j in ti])
                    if mask.sum() == 0:
                        blk['offsets'].append(blk['offsets'][-1])
                        continue
                    s = np.where(mask, s, -np.inf)
                    j = ti[int(np.argmax(s))]
                    valid = np.flatnonzero(mask); order = valid[np.argsort(-s[valid], kind='stable')]
                    assert ti[order[0]] == j, 'stable descending first must equal argmax'
                    tg = np.array([REC.global_row[rows[x]['unit_id']] for x in ti], np.int64)
                    blk['ranked'].append(tg[order]); blk['scored'].append(s[order])
                    blk['offsets'].append(blk['offsets'][-1] + len(order))
                    lab = rows[i]["label"]
                    chance = sum(1 for k, jj in enumerate(ti) if mask[k] and rows[jj]["label"] == lab) / mask.sum()
                    M[a, tl[t]] = float(rows[j]["label"] == lab) - chance
                    blk['lift'][a] = M[a, tl[t]]
            days = sorted({rows[i]["day"] for i in qi})
            per_lang[c] = (np.array([days.index(rows[i]["day"]) for i in qi]), M)
            for t, blk in blocks.items():
                if not tgt[t]:
                    continue
                tg = np.array([REC.global_row[rows[x]['unit_id']] for x in tgt[t]], np.int64)
                REC.record(f'{REC.program}|{context}|{c}->{t}',
                           's62_item8_float64_cosine_argmax_first' + ('' if t != c or include_same_clip else '_exclude_own_clip'),
                           blk['qrows'], tg, blk['ranked'], blk['scored'], blk['offsets'],
                           dict(query_rows=np.array(blk['qrows'], np.int64), lift=blk['lift']), arrhash(E),
                           dict(kind=context, source=c, target=t, score_dtype='float64'))
        comm = sorted(per_lang)

        def D(means):
            return np.mean([means[c][tl[c]] - np.mean([means[c][tl[t]] for t in S.LANGS if t != c]) for c in comm])

        o, b = S.day_boot(per_lang, D, rng)
        p2 = 2 * min(np.mean(b <= 0), np.mean(b >= 0))
        per_c = {}
        for c in comm:
            di, M = per_lang[c]
            m = np.nanmean(M, 0)
            per_c[c] = {"n_queries": n_q[c], "lift_same": round(float(m[tl[c]]), 4),
                        "lift_other": round(float(np.mean([m[tl[t]] for t in S.LANGS if t != c])), 4)}
        return {"D": round(float(o), 4), "p_two_sided": round(float(min(1.0, p2)), 4),
                "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
                "communities": per_c, "left_out": [c for c in S.LANGS if c not in comm]}

    def lift_analysis(E, rows, rng):
        REC.context = S62_LIFT_SEQ[REC.seq['lift']]; REC.seq['lift'] += 1
        return original_lift(E, rows, rng)

    def analyse(E, rows, rng):
        REC.context = S62_BYLABEL_SEQ[REC.seq['bylabel']]; REC.seq['bylabel'] += 1
        return original_bylabel(E, rows, rng)

    cr.Retriever.run = run
    S.pair_hits, S.item8, S.lift_analysis = pair_hits, item8, lift_analysis
    SBL.analyse = analyse


def recorded(run_name):
    global REC
    check_frozen()
    dest = DATA / 'registered' / f'run{run_name}'
    dest.mkdir(parents=True, exist_ok=False)
    out = dest / 'json'; out.mkdir()
    store = BlockStore(DATA / 'rankings', 'write' if run_name == 'A' else 'hash')
    REC = Recorder(store, DATA / 'rankings' if run_name == 'A' else dest, save=run_name == 'A')
    sys.path.insert(0, str(PIPE / 'item5')); sys.path.insert(0, str(PIPE))
    import ch5_retrieval as cr, s62_lift as S, s62_by_label as SBL, ch5_sensitivity as CS
    sources = {m.__name__: (m.__file__, sha(m.__file__)) for m in (cr, S, SBL, CS)}
    install(cr, S, SBL)
    pools = threads_one()
    timing = {}
    for program, module in zip(PROGRAMS, (cr, S, SBL, CS)):
        REC.program = program
        sys.argv = [module.__file__] + [str(x) for x in argv_for(program, out)]
        start = time.perf_counter()
        with (dest / f'{program}.stdout.txt').open('x', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            module.main()
        timing[program] = time.perf_counter() - start
        print('DONE', run_name, program, round(timing[program], 1), 's', len(REC.calls), 'calls', flush=True)
    assert REC.seq == {'lift': 10, 'item8': 2, 'bylabel': 2}, REC.seq
    dump(dest / 'calls.json', REC.calls)
    dump(dest / 'blocks.json', store.index)
    dump(dest / 'run_complete.json', dict(release=RELEASE_ID, run=run_name, programs=PROGRAMS, seconds=timing,
         calls=len(REC.calls), unique_blocks=len(store.index), entries=sum(v['entries'] for v in store.index.values()),
         json_sha256={p.name: sha(p) for p in sorted(out.glob('*.json'))}, sources=sources,
         runner_sha256=sha(__file__), common_sha256=sha(Path(__file__).with_name('fr_common.py')),
         spec_sha256=sha(REPORT / 'FORMAL_RELEASE_SPEC.md'), threadpools=pools,
         max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, environment=environment_record()))


def verbatim():
    check_frozen()
    dest = DATA / 'registered' / 'verbatim'
    dest.mkdir(parents=True, exist_ok=False)
    out = dest / 'json'; out.mkdir()
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    record = {}
    for program in PROGRAMS:
        script = PIPE / f'{program}.py'
        cmd = [str(ITEM5), str(script)] + [str(x) for x in argv_for(program, out)]
        start = time.perf_counter()
        with (dest / f'{program}.stdout.txt').open('x') as o, (dest / f'{program}.stderr.txt').open('x') as e:
            code = subprocess.call(cmd, stdout=o, stderr=e, env=env, cwd=str(ROOT))
        record[program] = dict(exit=code, seconds=time.perf_counter() - start, script_sha256=sha(script), command=cmd)
        print('VERBATIM', program, code, round(record[program]['seconds'], 1), flush=True)
        assert code == 0
    dump(dest / 'run_complete.json', dict(release=RELEASE_ID, programs=record,
         json_sha256={p.name: sha(p) for p in sorted(out.glob('*.json'))}, environment_variables={k: env[k] for k in
         ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--run', choices=['A', 'B'])
    a.add_argument('--verbatim', action='store_true')
    x = a.parse_args()
    verbatim() if x.verbatim else recorded(x.run)
