"""Index every local fit attempt, including excluded partial chains; no fitting."""
import argparse
from datetime import datetime
import ch5_posterior as p
from ch5_posterior_collect import select
from ch5_posterior_report import csv_save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-csv-hashes',action='store_true');args=parser.parse_args()
    rows=[]
    for name in p.MODELS:
        for variant_folder in sorted((p.WORK/'fits'/name).glob('*')):
            variant=variant_folder.name;chosen=select(name,variant)
            adopted=variant in p.VARIANTS or (name=='H0_W1' and variant in ['day_holdout','day_weighted'])
            for folder in sorted(variant_folder.glob('attempt_*'),key=lambda f:int(f.name.split('_')[1])):
                def record(file):return p.loadjson(folder/file) if (folder/file).exists() else {}
                sampling=record('sampling.json');running=record('running.json');interrupted=record('interrupted.json')
                failed=record('failed.json');diag=record('diagnostics.json');post=record('postprocessed.json');target=record('target.json')
                attempt=int(folder.name.split('_')[1]);qualified=bool(sampling and diag.get('parameters_pass') and post.get('summaries_diagnostics_pass'))
                selected=bool(adopted and qualified and chosen and chosen[0] and chosen[1]==attempt)
                status='selected_qualified' if selected else 'qualified_superseded' if qualified else 'sampled_not_qualified' if sampling else 'interrupted_excluded' if interrupted else 'execution_failed' if failed else 'running_record_no_complete_manifest' if running else 'prepared_only_no_sampling_manifest'
                settings=sampling.get('settings',running.get('settings',target.get('settings',{})))
                elapsed_proxy=None
                if running.get('started_at') and interrupted.get('recorded_at'):
                    elapsed_proxy=(datetime.fromisoformat(interrupted['recorded_at'])-datetime.fromisoformat(running['started_at'])).total_seconds()
                digests=dict(sampling.get('csv_sha256',{}));verified=False
                if args.verify_csv_hashes and (sampling or interrupted):
                    actual={str(f):p.sha(f) for f in sorted(folder.glob('ch5_posterior-*.csv'))}
                    assert all(actual.get(file)==digest for file,digest in digests.items())
                    digests=actual;verified=True
                rows.append(dict(model=name,variant=variant,attempt=attempt,status=status,adopted_logical_target=adopted,
                    implementation=target.get('equivalent_implementation','original'),
                    complete_chain_manifest=bool(sampling),accepted_complete_chains=4 if selected else 0,
                    completed_chains=sampling.get('chains'),retained_per_chain=sampling.get('retained_per_chain'),
                    requested_warmup_per_chain=settings.get('iter_warmup'),requested_retained_per_chain=settings.get('iter_sampling'),
                    seed=settings.get('seed'),chain_ids=settings.get('chain_ids'),
                    sampling_seconds=sampling.get('elapsed_seconds'),sampling_peak_RSS_bytes=sampling.get('peak_RSS_process_tree_bytes'),
                    incomplete_elapsed_proxy_from_recorded_timestamps=elapsed_proxy,
                    excluded_partial_rows_as_recorded=interrupted.get('retained_rows_in_partial_files'),
                    excluded_partial_draws_used_in_posterior_summaries=False if interrupted else None,
                    parameters_pass=diag.get('parameters_pass'),summary_diagnostics_pass=post.get('summaries_diagnostics_pass'),
                    max_Rhat=diag.get('max_Rhat'),divergences=diag.get('divergences'),
                    CSV_sha256=digests,CSV_hashes_freshly_verified=verified,
                    adapter_sha256=target.get('adapter_sha256'),canonical_Stan_sha256=target.get('stan_sha256'),
                    target_manifest_sha256=p.sha(folder/'target.json') if target else None,
                    absolute_path=str(folder)))
    p.dump(p.OUT/'attempt_history.json',rows);csv_save(p.OUT/'attempt_history.csv',rows)
    p.dump(p.OUT/'attempt_history_notes.json',dict(
        recorded_at=p.time.strftime('%Y-%m-%dT%H:%M:%S%z'),attempts=len(rows),selected_qualified=sum(r['status']=='selected_qualified' for r in rows),
        note='Only selected qualified attempts contribute posterior results. Incomplete actual draw counts are unknown unless recorded at interruption, never fabricated as zero. Timestamp differences are explicitly inferred proxies, not measured sampler time. Per-attempt wall times overlap under concurrency; do not sum them as elapsed host time. RSS covers sampling only. Historical partial draws stay excluded from posterior summaries; recorded states/metrics may provide proposal initialization for fresh complete chains.'))
    print('Indexed',len(rows),'actual attempt folders;',sum(r['status']=='selected_qualified' for r in rows),'selected qualified fits',flush=True)


if __name__=='__main__':main()
