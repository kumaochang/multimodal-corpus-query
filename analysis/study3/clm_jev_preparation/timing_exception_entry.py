"""Author removed the minimum retest interval; strict runner is retained.

No date substitution and no change to models/metrics/RNG. Final artifacts use
research_protocol_deviation, which the existing Jev research gate rejects.
"""
import argparse
import csv
from datetime import timedelta
import json
from pathlib import Path
import shutil
import sys

import evaluation as e
import registered_entry as retained
from clm_inputs import verify
from contracts import (Blocked, HOME, ROOT, artifact, authorize, create_json,
                       json_load, ref, require, require_environment, safe_work,
                       sha_file, verified_path)

OPERATION = 'registered_clm_evaluation_timing_exception'
EXCEPTION_SCOPE = 's61_3_retest_minimum_seven_days_only'
MODE = 'research_protocol_deviation'
AUTHORIZATION_MESSAGE_ID = 'retest_interval_removed_2026_11_01'
PROGRAMS = ('timing_exception_entry.py', 'prepare_timing_request.py', 'registered_entry.py',
            'evaluation.py', 'contracts.py', 'clm_inputs.py', 'material_audit.py')


def validate_exception(exception, receipt):
    require(exception.get('scope') == EXCEPTION_SCOPE, 'Wrong exception scope')
    require(exception.get('waive_minimum_seven_days') is True, 'No explicit interval exception')
    require(exception.get('preserve_original_timestamps') is True, 'Original timestamps must be preserved')
    require(exception.get('items_sha256') == receipt.get('items_sha256'), 'Exception bound to other items')
    require(exception.get('answer_files') == receipt.get('answer_files'), 'Exception bound to other answers')
    require(bool(exception.get('author_instruction')), 'Missing direct author instruction')
    require(exception.get('authorization_message_id') == AUTHORIZATION_MESSAGE_ID,
            'Missing verified direct author instruction locator')
    require(exception.get('batch2_completed_at') == receipt.get('batch2_completed_at')
            and exception.get('batch3_started_at') == receipt.get('batch3_started_at'),
            'Exception timestamps differ from the bound answer files')


def completion_with_exception(items, answers, receipt, exception):
    validate_exception(exception, receipt)
    # The retained gate performs all checks up to interval/chronology. Only its
    # specifically identified interval failure can be handled here.
    strict_receipt = dict(receipt, authorization='author-approved-registered-evaluation',
        scope_decision_adopted=True, minimum_positive_scope='all_batches_1_2',
        frozen_at='2026-10-17T11:29:13+08:00')
    try:
        e.completion_gate(items, answers, strict_receipt)
    except e.NotReady as exc:
        if str(exc) != 'Retest interval or completion chronology invalid':
            raise
    else:
        raise Blocked('Strict interval already passes; use the retained strict entry')
    end = e.instant(receipt['batch2_completed_at'])
    start = e.instant(receipt['batch3_started_at'])
    b2times = [e.instant(r['time']) for r in answers[2]]
    b3times = [e.instant(r['time']) for r in answers[3]]
    require(end == max(b2times), 'Completion evidence is not the actual last batch 2 answer')
    require(start == min(b3times), 'First-answer upper bound is not the actual first recorded batch 3 answer')
    require(start >= end, 'Retest chronology invalid independently of seven-day rule')
    require(start - end < timedelta(days=7), 'This interval-removal entry is not needed')
    # Repeat checks follow the retained interval check and must still run.
    main = [r for r in items if r['batch'] in (1, 2)]
    first = e.unique_index(main, 'utt_id')
    repeat = [r for r in items if r['batch'] == 3]
    e.unique_index(repeat, 'utt_id')
    for r in repeat:
        require(r['utt_id'] in first and all(r[k] == first[r['utt_id']][k]
            for k in ('day', 'clip', 'stratum', 'incl_prob')), 'Repeat does not match frozen primary item')
    answer = {r['item_id']: int(r['answer']) for group in answers.values() for r in group}
    positives = sum(answer[r['item_id']] == 1 for r in main)
    return {'answers': answer, 'positives_all_batches': positives,
        'descriptive_only': positives < 15,
        'answer3_count': sum(answer[r['item_id']] == 3 for r in main), 'batch3_n': 40}


def annotation(exception_ref, receipt):
    return {'mode': MODE, 'timing_status': 'authorized_timing_requirement_removed',
        'registration_minimum_retest_interval_met': False,
        'author_approved_timing_exception': exception_ref,
        'batch2_completed_at': receipt['batch2_completed_at'],
        'batch3_started_at': receipt['batch3_started_at'],
        'batch3_start_time_basis': receipt['batch3_start_time_basis'],
        'retest_interpretation': 'Agreement on 40 repeated items under the author-adopted removal of the minimum interval.',
        'primary_and_clm_use_batches': [1, 2],
        'automatic_jev_prerequisite_eligible': False}


def execute(receipt_path, enabled=False):
    require(enabled, 'Timing-exception execution disabled; no answers read')
    receipt = json_load(receipt_path)
    authorize(receipt, OPERATION)
    require(set(receipt.get('programs', {})) == set(PROGRAMS), 'Incomplete program fingerprint binding')
    for name in PROGRAMS:
        require(verified_path(receipt['programs'][name]).resolve() == (HOME / name).resolve(),
                'Request belongs to another program installation')
    exception = artifact(receipt['timing_exception'], 'author_approved_protocol_exception', EXCEPTION_SCOPE)
    validate_exception(exception, receipt)
    proof = artifact(receipt['completion'], 'consultant_completion_with_timing_exception', 'batches_1_2_3')
    require(proof.get('batch_counts') == {'1': 50, '2': 190, '3': 40}, 'Incomplete batches')
    for key in ('answer_files', 'items_sha256', 'batch2_completed_at', 'batch3_started_at',
                'batch3_start_time_basis', 'completion_time_basis', 'timing_exception'):
        require(proof.get(key) == receipt.get(key) and receipt.get(key) is not None, 'Completion evidence differs: ' + key)
    require(proof.get('go_confirmed') is True, 'Missing batch 1 go evidence')
    require(receipt['batch3_start_time_basis'] == 'first_recorded_answer_proxy_not_verified_actual_start',
            'Unrecorded actual-start evidence treatment')
    inputs = verify(json_load(verified_path(receipt['bindings'])))
    require(sha_file(inputs['annotation/items.json']) == receipt['items_sha256'], 'Wrong frozen items')
    require_environment(retained.EXPECTED_ENV)  # Never replace by local profile.
    items = json_load(inputs['annotation/items.json'])['items']
    require(set(receipt['answer_files']) == {'1', '2', '3'}, 'Missing answer files')
    answers = {}
    for batch, value in receipt['answer_files'].items():
        with verified_path(value).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == ['item_id', 'answer', 'seconds', 'time'], 'Wrong answer schema')
            answers[int(batch)] = list(reader)
    gate = completion_with_exception(items, answers, receipt, exception)
    out = safe_work(receipt['out'])
    require(not out.exists(), 'Preserve previous output')
    # Retain the exact compute adapter output as an intermediate. Publication uses
    # separately tagged evidence; no intermediate is claimed as strict completion.
    staging = out / '_unreleased_adapter_output'
    retained.compute(items, gate, inputs, staging)
    context = annotation(receipt['timing_exception'], receipt)
    metrics = json_load(staging / 'registered_metrics.json')
    metrics.update(context)
    metrics['completion_verified_under_author_timing_exception'] = True
    metrics['completion_verified'] = False
    create_json(out / 'registered_metrics.json', metrics)
    for name in ('paired_day_draws.json', 'oof_scores.npz'):
        shutil.copyfile(staging / name, out / name)
        require(sha_file(staging / name) == sha_file(out / name), 'Numerical copy changed')
    for outcome in ('judgment_primary', 'registered_secondary_models', 'clm_baselines', 'retest_kappa'):
        obj = json_load(staging / (outcome + '.evidence.json'))
        obj.update(context)
        obj['payloads'] = [ref(out / name) for name in ('registered_metrics.json', 'paired_day_draws.json', 'oof_scores.npz')]
        create_json(out / (outcome + '.evidence.json'), obj)
    create_json(out / 'execution_provenance.json', {
        **context, 'status': 'completed_with_author_removed_timing_requirement',
        'receipt': ref(receipt_path), 'authorization': receipt['authorization'],
        'completion': receipt['completion'], 'inputs': {k: ref(p) for k, p in inputs.items()},
        'programs': {p.name: ref(p) for p in (Path(__file__), HOME / 'registered_entry.py', HOME / 'evaluation.py', HOME / 'contracts.py')},
        'unreleased_intermediate': str(staging),
        'unreleased_intermediate_is_not_strict_completion_evidence': True,
    })
    print(json.dumps({'status': 'completed_with_author_removed_timing_requirement', 'out': str(out)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt')
    parser.add_argument('--execute-with-timing-exception', action='store_true')
    args = parser.parse_args()
    try:
        execute(args.receipt, args.execute_with_timing_exception)
    except (Blocked, e.NotReady) as exc:
        print(json.dumps({'status': 'blocked', 'reason': str(exc)}))
        sys.exit(2)
