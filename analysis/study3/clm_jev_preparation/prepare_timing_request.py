"""Bind the author's actual interval-removal decision to explicit frozen inputs.

Creates new local evidence only. Does not fit, bootstrap, infer, connect, or alter
the original strict completion record. Run on the eventual execution host so
paths and the request fingerprint bind that host's verified files.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import evaluation as e
from clm_inputs import verify
from contracts import (HOME, create_json, fingerprint, json_load, ref, require,
                       safe_work, sha_file)
from timing_exception_entry import (AUTHORIZATION_MESSAGE_ID, EXCEPTION_SCOPE,
                                    OPERATION, PROGRAMS, completion_with_exception)

FINAL_AUTHOR_MESSAGE_ID = '01a196ca-4833-7740-ba52-3c126acc6505'


def prepare(bindings_path, answers_root, evidence_dir, result_dir):
    inputs = verify(json_load(bindings_path))
    items = json_load(inputs['annotation/items.json'])['items']
    folder, target = safe_work(evidence_dir), safe_work(result_dir)
    require(not folder.exists() and not target.exists(), 'Preserve existing evidence/results')
    require(not folder.is_relative_to(target) and not target.is_relative_to(folder), 'Separate evidence and output')
    answers, answer_files = {}, {}
    for b in (1, 2, 3):
        p = Path(answers_root) / f'results_batch_{b}.csv'
        answer_files[str(b)] = ref(p)
        with p.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            require(reader.fieldnames == ['item_id', 'answer', 'seconds', 'time'], 'Wrong CSV schema')
            answers[b] = list(reader)
    # These are preserved recorded-answer bounds, not a claim about actual
    # browser-session start. Removal of the minimum interval makes that unknown
    # start irrelevant to the adopted gate.
    b2 = max(e.instant(r['time']) for r in answers[2]).isoformat()
    b3 = min(e.instant(r['time']) for r in answers[3]).isoformat()
    exception = {
        'artifact_kind': 'author_approved_protocol_exception', 'scope': EXCEPTION_SCOPE,
        'mode': 'research', 'status': 'completed',
        'decision_status': 'authorized_timing_requirement_removed',
        'waive_minimum_seven_days': True, 'preserve_original_timestamps': True,
        'authorization_message_id': AUTHORIZATION_MESSAGE_ID,
        'author_instruction': '第三批是今日收到，沒關係，不用理會日期',
        'author_followup_message_id': FINAL_AUTHOR_MESSAGE_ID,
        'author_followup': '不以七日間隔阻擋、取消時間也無須揭露偏離',
        'presentation': 'Describe 40 repeated judgments; omit dates/deviation terminology from dissertation prose; do not assert a seven-day interval.',
        'actual_batch3_start': None,
        'batch2_completed_at': b2, 'batch3_started_at': b3,
        'answer_files': answer_files, 'items_sha256': sha_file(inputs['annotation/items.json']),
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    request = {
        'operation': OPERATION, 'bindings': ref(bindings_path),
        'items_sha256': exception['items_sha256'], 'answer_files': answer_files,
        'batch2_completed_at': b2, 'batch3_started_at': b3,
        'batch3_start_time_basis': 'first_recorded_answer_proxy_not_verified_actual_start',
        'completion_time_basis': 'verified_csv_utc', 'out': str(target),
        'programs': {name: ref(HOME / name) for name in PROGRAMS},
    }
    # All retained integrity/go/repeat checks must pass before evidence is made.
    completion_with_exception(items, answers, request, exception)
    request['timing_exception'] = create_json(folder / 'author_timing_decision.json', exception)
    completion = {k: request[k] for k in ('answer_files', 'items_sha256', 'batch2_completed_at',
        'batch3_started_at', 'batch3_start_time_basis', 'completion_time_basis', 'timing_exception')}
    completion.update(artifact_kind='consultant_completion_with_timing_exception',
        scope='batches_1_2_3', mode='research', status='completed',
        batch_counts={'1': 50, '2': 190, '3': 40}, go_confirmed=True,
        strict_registered_interval_passed=False)
    request['completion'] = create_json(folder / 'completion_with_timing_removal.json', completion)
    approval = {
        'artifact_kind': 'execution_authorization', 'scope': OPERATION,
        'mode': 'research', 'status': 'completed', 'approved': True,
        'approved_at': datetime.now(timezone.utc).isoformat(),
        'approval_timestamp_basis': 'Evidence recording time, not a claim about the precision of the source-message time.',
        'request_sha256': fingerprint(request),
        'authorization_basis': {'phase1': 'Researcher delegation relaying the author-approved execution of the existing registered outcomes.',
            'minimum_interval_removal': AUTHORIZATION_MESSAGE_ID,
            'presentation_followup': FINAL_AUTHOR_MESSAGE_ID,
            'excluded': ['Jev', 'new symbolic-versus-similarity comparisons', 'CLM inference', 'remote agent execution']},
    }
    request['authorization'] = create_json(folder / 'execution_authorization.json', approval)
    reference = create_json(folder / 'registered_request.json', request)
    print(json.dumps({'request': reference, 'request_fingerprint': approval['request_sha256'],
        'status': 'prepared_only', 'execution_started': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', required=True)
    parser.add_argument('--answers-root', required=True)
    parser.add_argument('--evidence-dir', required=True)
    parser.add_argument('--result-dir', required=True)
    a = parser.parse_args()
    prepare(a.bindings, a.answers_root, a.evidence_dir, a.result_dir)
