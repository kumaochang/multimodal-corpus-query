"""ULFR-v1 supplementary sampler-consistency check (defined after the formal H0_W3 fit was inspected).

For H0_W3, HA_W3 and HB_W3, the formal base fit and an earlier independent server fit have value-identical
model data (H0: reproduction same_input; HA/HB: reproduction server_data, i.e. the Linux ranking). Different
seeds/tuning, same target posterior. For every pooled/speech margin (conditional and population-average,
probability and lift) and every non-overlap fixed contrast, compare mean and 2.5/97.5% quantiles with
D = |a-b| / sqrt(MCSE_a^2 + MCSE_b^2) (ArviZ chain-aware MCSE) against z = Phi^-1(1 - .01/(2m)).
This is a Monte Carlo consistency diagnostic, not an effect test.
"""
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
import json, sys
from pathlib import Path
import numpy as np
import arviz as az
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fr_common as fc  # noqa: E402

PAIRS = {'H0_W3': fc.REPRO_DATA / 'posterior/same_input/H0_W3/initial',
         'HA_W3': fc.REPRO_DATA / 'posterior/server_data/HA_W3/initial',
         'HB_W3': fc.REPRO_DATA / 'posterior/server_data/HB_W3/initial'}
FORMAL = fc.DATA / 'posterior/work/fits'


def series(folder, desc_path):
    z = np.load(folder / 'posterior_cell_draws.npz')
    W = z['group_weights']; desc = fc.read(desc_path)
    rec = fc.read(folder / 'sampling.json'); n = rec['retained_per_chain']
    out = {}
    rows = [r for r in desc if r['prediction'] == 'conditional']
    chance = np.array([r['chance'] for r in rows])
    for label in ('conditional', 'population_average'):
        v = z[label] @ W.T
        for j, r in enumerate(rows):
            if r['scope'] not in ('pooled_margin', 'speech_margin'):
                continue
            k = (label, r['scope'], r['route'], r['model_class'], r['speech_group'])
            out[k + ('probability',)] = v[:, j]
        routes = list(dict.fromkeys(r['route'] for r in rows))
        idx = lambda route, klass='all', speech='all': next((j for j, r in enumerate(rows) if r['route'] == route and r['model_class'] == klass
                                                             and r['speech_group'] == speech and r['scope'] == ('pooled_margin' if speech == 'all' else 'speech_margin')), None)
        vv = v - chance
        pairs = [('ASR_xlmr', 'subtitle_xlmr'), ('ASR_tfidf', 'subtitle_tfidf')] if 'ASR_xlmr' in routes else [('whisper', 'prosody'), ('xlmr_sub', 'tfidf_sub'), ('kin14', 'whisper')]
        for a, b in pairs:
            for g in ('all', 'Explaining', 'Reporting'):
                out[(label, 'contrast', f'{a}-{b}', g, 'all', 'lift')] = vv[:, idx(a, g)] - vv[:, idx(b, g)]
            out[(label, 'contrast', f'{a}-{b}', 'E_minus_R_DID', 'all', 'lift')] = (vv[:, idx(a, 'Explaining')] - vv[:, idx(b, 'Explaining')]
                                                                                   - vv[:, idx(a, 'Reporting')] + vv[:, idx(b, 'Reporting')])
            if idx(a, 'all', 'M') is not None:
                out[(label, 'contrast', f'{a}-{b}', 'M_minus_I_DID', 'all', 'lift')] = (vv[:, idx(a, 'all', 'M')] - vv[:, idx(b, 'all', 'M')]
                                                                                       - vv[:, idx(a, 'all', 'I')] + vv[:, idx(b, 'all', 'I')])
    return out, n


def stats(x, n):
    data = az.from_dict(posterior={'v': x.reshape(4, n)})
    est = [x.mean(), np.quantile(x, .025), np.quantile(x, .975)]
    mc = [float(az.mcse(data, method='mean')['v']), float(az.mcse(data, method='quantile', prob=.025)['v']),
          float(az.mcse(data, method='quantile', prob=.975)['v'])]
    return est, mc


def main():
    results, total = {}, 0
    for name, other in PAIRS.items():
        fdir = FORMAL / name / 'base/attempt_1'
        a, na = series(fdir, fc.DATA / f'posterior/out/summaries/{name}_base_attempt_1/summaries.json')
        b, nb = series(other, other / 'report/summaries/summaries.json')
        assert set(a) == set(b)
        rows = []
        for k in a:
            if np.ptp(a[k]) == 0 and np.ptp(b[k]) == 0:
                continue
            ea, ma = stats(a[k], na); eb, mb = stats(b[k], nb)
            for s, x, y, u, w in zip(('mean', 'q025', 'q975'), ea, eb, ma, mb):
                rows.append(dict(coordinate='|'.join(k), statistic=s, formal=x, independent=y, formal_MCSE=u, independent_MCSE=w,
                                 D=abs(x - y) / np.hypot(u, w)))
        results[name] = dict(formal_fit=str(fdir), independent_fit=str(other), independent_sampling=fc.read(other / 'sampling.json')['settings'],
                             formal_seed=fc.read(fdir / 'sampling.json')['settings']['seed'], rows=rows)
        total += len(rows)
    z = float(norm.ppf(1 - .01 / (2 * total)))
    summary = {}
    for name, r in results.items():
        D = [x['D'] for x in r['rows']]
        r['warnings'] = [x for x in r['rows'] if x['D'] > z]
        summary[name] = dict(coordinates=len(r['rows']), max_D=max(D), warnings=len(r['warnings']))
    fc.dump(fc.REPORT / 'verification/POSTERIOR_MC_CONSISTENCY.json', dict(release=fc.RELEASE_ID, definition=__doc__, m=total, threshold=z,
            summary=summary, models=results), exclusive=False)
    print(json.dumps(dict(m=total, threshold=z, summary=summary), indent=1))


if __name__ == '__main__':
    main()
