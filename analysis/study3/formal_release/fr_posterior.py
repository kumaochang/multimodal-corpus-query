"""ULFR-v1 stage P: hierarchical fits, diagnostics and postprocessing on the eleven formal model data.

Run with the isolated posterior interpreter (fr_common.POSTERIOR_PY). The original modules
ch5_posterior / ch5_posterior_report / ch5_posterior_collect / ch5_warm_start /
ch5_refit_support are imported unchanged from the verified core; only paths, the model-data
hash source (the formal prefit) and the CmdStan location are redirected, plus the same
assertion-only chance-logit tolerance the reproduction documented (stored logits are used).

Stages:
  validate                     compile both implementations; likelihood/gradient checks for all 11
  fit --model M --variant V --attempt A    one four-chain fit, diagnostics, postprocessing
  queue [--jobs 7]             fixed schedule: H0_W1 benchmark, bases, one repair, variants, benchmarks
  collect                      statuses, comparisons, PPC/influence/refit-support, identification
Never reranks, never touches chains of another attempt, never mixes attempts.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'STAN_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, csv, inspect, json, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fr_common as fc  # noqa: E402
sys.path.insert(0, str(fc.CORE / 'scripts'))
_path, _make = os.environ['PATH'], os.environ.get('MAKE')
import ch5_posterior as p  # noqa: E402  (module import sets Windows PATH/MAKE; restored below)
os.environ['PATH'] = _path
if _make is None:
    os.environ.pop('MAKE', None)
else:
    os.environ['MAKE'] = _make
os.environ['CMDSTAN'] = str(fc.CMDSTAN)
import numpy as np  # noqa: E402
import ch5_posterior_report as pr  # noqa: E402
import ch5_posterior_collect as pc  # noqa: E402
import ch5_warm_start as ws  # noqa: E402,F401  (used by p.sample for variant initialization)

SUP = fc.DATA / 'supplement/runA'
POST = fc.DATA / 'posterior'
STATUS = POST / 'job_status'
VARIANTS = ['base', 'prior_half', 'prior_double', 'flexible_baseline', 'add_clip', 'no_day_slopes']
BENCH = ['day_holdout', 'day_weighted']
ORIGINAL_ADAPTER_SHA = None
COMPAT_ADAPTER_SHA = None


def metadata():
    for key in ['features_full/features.npz', 'label_analysis/units.tsv']:
        assert fc.sha(fc.INPUT / key) == fc.FROZEN[key]
    ids = np.load(fc.INPUT / 'features_full/features.npz', allow_pickle=False)['ids']
    with (fc.INPUT / 'label_analysis/units.tsv').open(encoding='utf-8') as f:
        rows = {r['unit_id']: {k: r[k] for k in ['lang', 'day', 'clip', 'haiku', 'lid_answer', 'mf_class']}
                for r in csv.DictReader(f, delimiter='\t')}
    return {k: np.array([rows[u][k] for u in ids]) for k in ['lang', 'day', 'clip', 'haiku', 'lid_answer', 'mf_class']}


def configure():
    global ORIGINAL_ADAPTER_SHA, COMPAT_ADAPTER_SHA
    prefit = fc.read(SUP / 'execution_reports/hierarchy_prefit.json')
    p.PREFIT = prefit
    p.MODELS = {x['model'] + '_' + x['pool']: x for x in prefit['models']}
    assert len(p.MODELS) == 11
    p.OLD = SUP / 'execution'
    p.WORK = POST / 'work'; p.OUT = POST / 'out'; p.REPORT = POST / 'report_inputs'
    p.RUNTIME = fc.CMDSTAN.parent
    p.metadata = metadata
    inputs = p.REPORT / 'execution_20261020'
    inputs.mkdir(parents=True, exist_ok=True)
    for f in ['preflight.json', 'adopted_overlap_weights.csv']:
        if not (inputs / f).exists():
            shutil.copyfile(SUP / 'execution_reports' / f, inputs / f)
        assert fc.sha(inputs / f) == fc.sha(SUP / 'execution_reports' / f)
    assert fc.sha(inputs / 'adopted_overlap_weights.csv') == fc.sha(fc.REF / 'execution_20261020/adopted_overlap_weights.csv')
    assert fc.read(inputs / 'preflight.json')['day_keys'] == fc.read(fc.REF / 'execution_20261020/preflight.json')['day_keys']
    for name, m in p.MODELS.items():
        assert fc.sha(p.OLD / (name + '_model_data.npz')) == m['model_data_sha256']
    # Same documented compatibility as the reproduction: only the bit-exact re-evaluation of
    # logit(chance) in the assertion uses the frozen floating tolerance; stored logits are used.
    if COMPAT_ADAPTER_SHA is None:
        source = inspect.getsource(p.adapter)
        ORIGINAL_ADAPTER_SHA = fc.hashlib.sha256(source.encode()).hexdigest()
        old = "np.testing.assert_allclose(z['chance_logit'],np.log(chance[inside]/(1-chance[inside])),rtol=0,atol=0)"
        assert source.count(old) == 1
        source = source.replace(old, old.replace('rtol=0,atol=0', 'rtol=1e-10,atol=1e-12'))
        COMPAT_ADAPTER_SHA = fc.hashlib.sha256(source.encode()).hexdigest()
        exec(compile(source, '<ULFR_v1_logit_assertion_compatibility>', 'exec'), p.__dict__)
    return prefit


def implementation(name, variant):
    return 'original' if name.startswith('H0') and variant == 'base' else 'native_events'


def validate():
    configure()
    import cmdstanpy
    cmdstanpy.set_cmdstan_path(str(fc.CMDSTAN))
    records = {}
    for impl in ['original', 'native_events']:
        os.environ['CH5_STAN_IMPLEMENTATION'] = impl
        model = p.compiled()
        records[impl] = dict(exe=str(model.exe_file), exe_sha256=fc.sha(model.exe_file), generated_stan_sha256=fc.sha(model.stan_file))
    p.validate_stan()  # writes stan_validation_native.json (last implementation) under p.OUT
    os.environ['CH5_STAN_IMPLEMENTATION'] = 'original'
    p.validate_stan()
    original = cmdstanpy.CmdStanModel(exe_file=records['original']['exe'], stan_file=str(p.WORK / 'compiled/ch5_posterior.stan'))
    native = cmdstanpy.CmdStanModel(exe_file=records['native_events']['exe'], stan_file=str(p.WORK / 'compiled_native_events/ch5_posterior.stan'))
    checks = []
    for k, name in enumerate(p.MODELS):
        for variant in VARIANTS + (BENCH if name == 'H0_W1' else []):
            os.environ['CH5_STAN_IMPLEMENTATION'] = 'native_events'
            d, _, _, _ = p.adapter(name, variant)
            data = p.clean(d)
            rng = np.random.Generator(np.random.PCG64(202610250 + 97 * k + VARIANTS.index(variant) if variant in VARIANTS else 202610250 + 97 * k + 50))
            pars = dict(alpha=.17, z_beta=rng.normal(0, .2, d['J']), tau=np.full(d['F'], .3), z_day=rng.normal(0, .2, d['D']),
                        z_query=rng.normal(0, .2, d['Q']), z_slope=rng.normal(0, .2, (d['D'], d['H'])), sigma_day=.6, sigma_query=.7,
                        sigma_slope=np.full(d['H'], .4), z_clip=rng.normal(0, .2, d['CL']), sigma_clip=.5, baseline_delta=np.array([.12, -.08]))
            order = np.asarray(d['interaction_order'])[np.asarray(d['family']) - 1]
            beta = pars['z_beta'] * np.where(order == 1, .75, pars['tau'][np.asarray(d['family']) - 1]) * d['prior_scale']
            eta = d['chance_logit'] + .17 + (d['X_cell'] @ beta)[d['cell'] - 1] + .6 * pars['z_day'][d['day'] - 1] + .7 * pars['z_query'][d['query'] - 1]
            if d['H']:
                slope = np.sum(d['slope_design'] * (pars['z_slope'][d['slope_day'] - 1] * pars['sigma_slope']), axis=1)
                eta = eta + slope[d['slope_index'] - 1]
            if d['CL']:
                eta = eta + .5 * pars['z_clip'][d['clip'] - 1]
            if d['flexible_baseline']:
                eta = eta + d['baseline_basis'] @ pars['baseline_delta']
            ll = float(np.sum(d['likelihood_weight'] * (d['hit'] * eta - np.logaddexp(0, eta))))
            zero = dict(data); zero['likelihood_weight'] = [0.] * d['N']
            row = dict(model=name, variant=variant)
            for impl, model in [('original', original), ('native_events', native)]:
                a = model.log_prob(params=p.clean(pars), data=data, jacobian=False, sig_figs=18)
                b = model.log_prob(params=p.clean(pars), data=zero, jacobian=False, sig_figs=18)
                row[impl + '_independent_loglik_error'] = abs(float(a['lp__'].iloc[0] - b['lp__'].iloc[0]) - ll)
            deltas = []
            for jac in [False, True]:
                for dd in [data, zero]:
                    aa = original.log_prob(params=p.clean(pars), data=dd, jacobian=jac, sig_figs=18)
                    bb = native.log_prob(params=p.clean(pars), data=dd, jacobian=jac, sig_figs=18)
                    deltas.append(float(np.max(np.abs(aa.values - bb.values))))
            row['max_lp_and_gradient_delta_original_vs_native'] = max(deltas)
            row['passed'] = bool(row['original_independent_loglik_error'] < 2e-8 and row['native_events_independent_loglik_error'] < 2e-8 and max(deltas) < 2e-8)
            checks.append(row)
            print('VALIDATED', name, variant, row['passed'], flush=True)
    fc.dump(POST / 'validation.json', dict(release=fc.RELEASE_ID, binaries=records, checks=checks, all_passed=all(r['passed'] for r in checks),
                                          original_adapter_source_sha256=ORIGINAL_ADAPTER_SHA, compatibility_adapter_source_sha256=COMPAT_ADAPTER_SHA,
                                          environment=fc.environment_record(), cmdstan=str(fc.CMDSTAN)), exclusive=False)
    assert all(r['passed'] for r in checks)


def fit(name, variant, attempt):
    configure()
    os.environ['CH5_STAN_IMPLEMENTATION'] = implementation(name, variant)
    start = time.time()
    status = dict(release=fc.RELEASE_ID, model=name, variant=variant, attempt=attempt, implementation=implementation(name, variant),
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), runner_sha256=fc.sha(__file__),
                  compatibility_adapter_source_sha256=COMPAT_ADAPTER_SHA)
    STATUS.mkdir(parents=True, exist_ok=True)
    path = STATUS / f'{name}_{variant}_attempt_{attempt}.json'
    try:
        record = p.sample(name, variant, attempt)
        diag = p.diagnose(name, variant, attempt)
        post = pr.summarize(name, variant, attempt)
        status.update(sampled=True, parameters_pass=diag['parameters_pass'], summaries_diagnostics_pass=post['summaries_diagnostics_pass'],
                      qualified=bool(diag['parameters_pass'] and post['summaries_diagnostics_pass']),
                      settings=record['settings'], sampling_seconds=record['elapsed_seconds'], divergences=diag['divergences'],
                      max_Rhat=diag['max_Rhat'], min_bulk_ESS=diag['min_bulk_ESS'], max_Rhat_reported=post['max_Rhat_reported'],
                      min_bulk_ESS_reported=post['min_bulk_ESS_reported'], min_tail_ESS_reported=post['min_tail_ESS_reported'])
    except Exception as err:  # retained as a failed attempt; never converted into a posterior
        status.update(sampled=False, qualified=False, error=f'{type(err).__name__}: {err}')
        raise
    finally:
        status['wall_seconds'] = time.time() - start
        fc.dump(path, status, exclusive=False)
    return 0 if status['qualified'] else 2


def queue(jobs):
    configure()
    log = POST / 'queue_log.jsonl'
    POST.mkdir(exist_ok=True)

    def note(**k):
        k['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        with log.open('a') as f:
            f.write(json.dumps(k) + '\n')
        print(json.dumps(k), flush=True)

    def done(name, variant, attempt):
        s = STATUS / f'{name}_{variant}_attempt_{attempt}.json'
        return fc.read(s) if s.exists() else None

    running = {}

    def launch(name, variant, attempt):
        logs = POST / 'worker_logs'; logs.mkdir(parents=True, exist_ok=True)
        out = (logs / f'{name}_{variant}_attempt_{attempt}.log').open('x')
        cmd = [sys.executable, __file__, 'fit', '--model', name, '--variant', variant, '--attempt', str(attempt)]
        running[(name, variant, attempt)] = (subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), out)
        note(event='launch', model=name, variant=variant, attempt=attempt)

    def reap():
        finished = []
        for key, (proc, out) in list(running.items()):
            if proc.poll() is not None:
                out.close(); del running[key]
                s = done(*key) or {}
                note(event='finished', model=key[0], variant=key[1], attempt=key[2], exit=proc.returncode, qualified=s.get('qualified'))
                finished.append((key, s))
        return finished

    # Fixed order: benchmark first, then bases by observed historical cost (longest first).
    order = ['HA_W2', 'HA_W3', 'HA_W1', 'HA_X', 'HB_W1', 'HB_W2', 'HB_W3', 'H0_W2', 'H0_W3', 'H0_X']
    launch('H0_W1', 'base', 1)
    while running:
        reap(); time.sleep(5)
    pending = [(n, 'base', 1) for n in order]
    final_base = {}
    s = done('H0_W1', 'base', 1)
    if s and s.get('qualified'):
        final_base['H0_W1'] = 1
    else:
        pending.insert(0, ('H0_W1', 'base', 2))
    variants_queued = set()
    while pending or running:
        for (name, variant, attempt), s in reap():
            if not s.get('sampled'):
                # Infrastructure/compilation failure is not a posterior attempt: stop scheduling it and
                # leave it for diagnosis (RUN_SPEC); never auto-convert it into a repair.
                note(event='infrastructure_failure_not_rescheduled', model=name, variant=variant, attempt=attempt)
                continue
            if not s.get('qualified') and attempt == 1:
                pending.insert(0, (name, variant, 2))  # the one fixed repair, same seed
                note(event='repair_scheduled', model=name, variant=variant)
            elif variant == 'base':
                final_base[name] = attempt
        for name in list(final_base):
            if name not in variants_queued:
                variants_queued.add(name)
                pending += [(name, v, 1) for v in VARIANTS[1:] + (BENCH if name == 'H0_W1' else [])]
        while pending and len(running) < jobs:
            launch(*pending.pop(0))
        time.sleep(10)
    note(event='queue_complete')


def collect():
    configure()
    os.environ['CH5_STAN_IMPLEMENTATION'] = 'native_events'
    # verify_completed hashes the Windows binary name 'ch5_posterior.exe'; on Linux CmdStan writes
    # 'ch5_posterior'. Only that file name is adapted; every other check is the original code.
    source = inspect.getsource(p.verify_completed)
    assert source.count("'ch5_posterior.exe'") == 1
    exec(compile(source.replace("'ch5_posterior.exe'", "'ch5_posterior'"), '<ULFR_v1_linux_executable_name>', 'exec'), p.__dict__)
    statuses = pc.summarize_all()
    # PPC calibration and interaction-component aggregation (ch5_posterior_deliver logic without its
    # Windows package dependency).
    from collections import Counter
    diagnostics, components = [], []
    for model in p.MODELS:
        chosen = pc.select(model, 'base')
        if chosen is None or not chosen[0]:
            continue
        path = pc.summary_path(model, 'base', chosen[1]); ppc = p.loadjson(path / 'PPC.json')['calibration']
        for scope, n in Counter(r['scope'] for r in ppc).items():
            rr = [r for r in ppc if r['scope'] == scope]
            diagnostics.append(dict(model=model, scope=scope, calibration_coordinates=n,
                                    observed_outside_replicated95=sum(not r['observed_in_replicated95'] for r in rr),
                                    note='Posterior predictive calibration, not an independent significance/validation test.'))
        components += p.loadjson(path / 'interaction_components.json')
    p.dump(p.OUT / 'PPC_calibration_summary.json', diagnostics); pr.csv_save(p.OUT / 'PPC_calibration_summary.csv', diagnostics)
    p.dump(p.OUT / 'interaction_components.json', components); pr.csv_save(p.OUT / 'interaction_components.csv', components)
    # Finalize logic: verification, refit support, influence and paired-hit PPC summaries.
    import ch5_refit_support as pf
    verified, influence, pair_ppc, refit = [], [], [], []
    for row in statuses:
        name, variant = row['model'], row['variant']; chosen = pc.select(name, variant)
        if not chosen:
            continue
        good, attempt, folder, sampling, diag, post = chosen
        os.environ['CH5_STAN_IMPLEMENTATION'] = implementation(name, variant)
        v = dict(p.verify_completed(name, variant, attempt), diagnostics_pass=good)
        if post:
            assert p.sha(folder / 'loaded_postprocessor.py') == post['postprocessor_sha256']
            manifest = p.loadjson(p.OUT / 'summary_manifests' / f'{name}_{variant}_attempt_{attempt}.json')
            for file, digest in manifest['sha256'].items():
                assert p.sha(Path(manifest['absolute_path']) / file) == digest
            cache = folder / 'posterior_cell_draws.npz'
            v.update(postprocessor_and_summary_manifest_verified=True, posterior_cell_cache=dict(path=str(cache), sha256=p.sha(cache), bytes=cache.stat().st_size))
        verified.append(v)
        if good and variant.startswith('day_'):
            refit += pf.refit_support(name, variant)
        if variant != 'base' or not good:
            continue
        target = pc.summary_path(name, variant, attempt)
        days = p.loadjson(target / 'influential_days.json'); active = [r for r in days if r['likelihood_day_active']]
        ks = [r['conditional_Pareto_k'] for r in active if r['conditional_Pareto_k'] is not None]
        influence.append(dict(model=name, days=len(days), active_likelihood_days=len(active), conditional_k_over_0_7=sum(k > .7 for k in ks),
                              max_conditional_k=max(ks), maximum_deleted_day_target_probability_shift=max(r['deleted_day_target_max_absolute_probability_mean_shift'] for r in days),
                              note='Conditional latent-effect diagnostic and target deletion, not reliable heldout-day refitted validation.'))
        routes = list(dict.fromkeys(r['route'] for r in p.loadjson(target / 'summaries.json') if r['scope'] == 'pooled_margin' and r['model_class'] == 'all' and r['prediction'] == 'conditional'))
        for r in p.loadjson(target / 'PPC.json')['paired_hit_correlations']:
            pair_ppc.append(dict(model=name, route_a_name=routes[r['route_a']], route_b_name=routes[r['route_b']], **r))
    p.dump(p.OUT / 'selected_fit_verification.json', verified)
    p.dump(p.OUT / 'day_influence_summary.json', influence); pr.csv_save(p.OUT / 'day_influence_summary.csv', influence)
    p.dump(p.OUT / 'paired_hit_PPC_summary.json', pair_ppc); pr.csv_save(p.OUT / 'paired_hit_PPC_summary.csv', pair_ppc)
    p.dump(p.OUT / 'refit_likelihood_support.json', refit); pr.csv_save(p.OUT / 'refit_likelihood_support.csv', refit)
    # Identification audit and attempt index: the original modules, unchanged, on the redirected p.
    import ch5_identification_audit as ia
    ia.main()
    import ch5_posterior_attempts as pa
    sys.argv = ['ch5_posterior_attempts.py', '--verify-csv-hashes']
    pa.main()
    print('COLLECTED', sum(s['status'] == 'completed' for s in statuses), '/', len(statuses), flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('stage', choices=['validate', 'fit', 'queue', 'collect'])
    a.add_argument('--model'); a.add_argument('--variant', default='base'); a.add_argument('--attempt', type=int, default=1)
    a.add_argument('--jobs', type=int, default=7)
    x = a.parse_args()
    if x.stage == 'validate': validate()
    elif x.stage == 'fit': sys.exit(fit(x.model, x.variant, x.attempt))
    elif x.stage == 'queue': queue(x.jobs)
    else: collect()
