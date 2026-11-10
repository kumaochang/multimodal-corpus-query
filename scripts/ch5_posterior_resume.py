"""Idempotent base-fit entry point for the historical hierarchy fit() dispatcher."""
import argparse
import subprocess
import sys
import ch5_posterior as p


def main():
    p.os.environ.setdefault('CH5_STAN_IMPLEMENTATION','native_events')
    statuses=[]
    queue=p.OUT/'future_queue_status.json'
    queued=[]
    if queue.exists():
        state=p.loadjson(queue)
        # Restricted Windows process views may hide the authorized runtime PID.
        # Nonempty queue claims remain reservations until reconciled; absence of
        # a visible PID must never start a duplicate sampler.
        queued=[x['model'] for x in state.get('pending',[])+state.get('running',[]) if x['variant']=='base']
    for name in p.MODELS:
        # Prefer completed verified attempts, never concatenate incomplete chains.
        records=[]
        for f in (p.WORK/'fits'/name/'base').glob('attempt_*/sampling.json'):
            r=p.loadjson(f)
            if r['status']=='sampled':records.append((r['attempt'],f,r))
        accepted=[]
        for a,f,r in records:
            diagnostic=f.parent/'diagnostics.json'
            if diagnostic.exists() and p.loadjson(diagnostic).get('parameters_pass',False):accepted.append((a,f,r))
        if accepted:
            a,f,r=max(accepted)
            p.verify_completed(name,'base',a)
            statuses.append(dict(model=name,attempt=a,status='completed'))
            continue
        folders=list((p.WORK/'fits'/name/'base').glob('attempt_*'))
        running=False
        for d in folders:
            if (d/'running.json').exists() and not any((d/x).exists() for x in ['sampling.json','interrupted.json','failed.json']):
                record=p.loadjson(d/'running.json')
                if p.psutil.pid_exists(record['pid']):running=True
        if running or name in queued:
            statuses.append(dict(model=name,status='running_or_queued'));continue
        # New work uses the validated equivalent implementation and full four chains.
        attempt=max([int(d.name.split('_')[1]) for d in folders],default=0)+1
        # A completed diagnostic failure needs the declared 4000/.99 repair.
        # An interrupted initial attempt instead gets a fresh initial schedule.
        result=p.run_one(name,'base',attempt,numerical_restart=attempt>1 and not records)
        statuses.append(dict(model=name,attempt=attempt,status='completed' if result['parameters_pass'] else 'diagnostic_failure'))
    p.dump(p.OUT/'resume_status.json',statuses)
    print('Base fit status:',{s:sum(r['status']==s for r in statuses) for s in set(r['status'] for r in statuses)},flush=True)
    return 0 if all(r['status']=='completed' for r in statuses) else 2


if __name__=='__main__':sys.exit(main())
