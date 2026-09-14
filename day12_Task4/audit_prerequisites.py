#!/usr/bin/env python3
"""Read-only research prerequisite audit. This program cannot train or score images."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

DISTANCES = {'D0': 'D0', 'KL': 'KL', 'W2': 'W2_squared', 'BHATT': 'Bhattacharyya'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def audit(config):
    checks = []

    def check(name, passed, detail):
        checks.append({'check': name, 'passed': bool(passed), 'detail': detail})

    def read_json(path, name):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError) as exc:
            check(name, False, str(exc))
            return {}

    hash_checks = []
    for filename, expected in config['expected_hashes'].items():
        try:
            actual = sha256(filename)
            item = {'path': filename, 'expected_sha256': expected,
                    'actual_sha256': actual, 'passed': actual == expected}
        except OSError as exc:
            item = {'path': filename, 'expected_sha256': expected,
                    'actual_sha256': None, 'passed': False, 'error': str(exc)}
        hash_checks.append(item)
    check('frozen_artifact_hashes', all(row['passed'] for row in hash_checks),
          {'checked': len(hash_checks), 'failed': sum(not r['passed'] for r in hash_checks)})

    task3 = Path(config['task3_directory'])
    status = read_json(task3 / 'preflight_status.json', 'task3_status_readable')
    check('task3_complete', status.get('task_complete') is True and
          status.get('ready_for_matched_distance_training') is True,
          status.get('status', 'MISSING'))
    normalization_records = []
    for pole in ('EA', 'EH'):
        pole_status = read_json(task3 / pole / 'status.json', f'{pole}_status_readable')
        check(f'{pole}_task3_complete', pole_status.get('complete') is True,
              pole_status.get('status', 'MISSING'))
        path = task3 / pole / 'frozen_training_normalization.json'
        stats = read_json(path, f'{pole}_training_stats_readable')
        check(f'{pole}_training_stats_protocol',
              stats.get('ddof') == 0 and stats.get('epsilon') == 1e-8 and
              stats.get('fit_split') == 'unique severity-training images' and
              stats.get('n_unique_images') == config['training']['unique_images'],
              {'ddof': stats.get('ddof'), 'epsilon': stats.get('epsilon'),
               'fit_split': stats.get('fit_split'), 'N': stats.get('n_unique_images')})
        for distance, key in DISTANCES.items():
            record = stats.get('distances', {}).get(key)
            if record is None:
                check(f'{pole}_{distance}_frozen_training_stats', False,
                      'Required Task 3 normalization record is missing; not recomputed in Task 4.')
                continue
            mean, std = record.get('mean'), record.get('std_ddof0')
            valid = (isinstance(mean, (int, float)) and isinstance(std, (int, float)) and
                     math.isfinite(mean) and math.isfinite(std) and std > 1e-8 and
                     record.get('nonconstant') is True and record.get('nonfinite_count') == 0 and
                     record.get('n') == config['training']['unique_images'] and
                     record.get('finite_count') == record.get('n'))
            check(f'{pole}_{distance}_frozen_training_stats', valid, record)
            normalization_records.append({'pole': pole, 'distance': distance,
                'checkpoint': config['checkpoints'][pole],
                'checkpoint_sha256': config['expected_hashes'].get(config['checkpoints'][pole]),
                'train_split': str(task3 / 'inputs/train_images.csv'),
                'train_split_sha256': config['expected_hashes'].get(str(task3 / 'inputs/train_images.csv')),
                'N': record.get('n'), 'mean': mean, 'std_population': std,
                'epsilon': stats.get('epsilon'), 'ddof': stats.get('ddof'),
                'source': str(path), 'source_sha256': config['expected_hashes'].get(str(path)),
                'source_distance_name': key, 'values_copied_without_recomputation': True,
                'valid': valid})

        aggpath = Path(config['aggregate_references'][pole])
        try:
            with np.load(aggpath, allow_pickle=False) as data:
                required = ('mu_agg', 'v_between', 'v_within', 'v_aggregate_raw', 'v_agg')
                arrays = {key: data[key] for key in required}
                valid = (all(a.shape == (100,) and np.isfinite(a).all() for a in arrays.values()) and
                         np.all(arrays['v_between'] >= 0) and np.all(arrays['v_within'] > 0) and
                         np.all(arrays['v_agg'] > 0) and
                         np.allclose(arrays['v_aggregate_raw'], arrays['v_between'] + arrays['v_within'], rtol=1e-12, atol=1e-15) and
                         np.array_equal(arrays['v_agg'], np.maximum(arrays['v_aggregate_raw'], 1e-8)) and
                         float(data['variance_epsilon']) == 1e-8 and int(data['ddof']) == 0)
                check(f'{pole}_aggregate_reference', valid,
                      {'path': str(aggpath), 'N': int(data['n_images']),
                       'hash_is_previously_frozen': str(aggpath) in config['expected_hashes']})
                check(f'{pole}_aggregate_frozen_hash_available',
                      str(aggpath) in config['expected_hashes'], str(aggpath))
        except (OSError, ValueError, KeyError) as exc:
            check(f'{pole}_aggregate_reference', False, {'path': str(aggpath), 'error': str(exc)})
        legacy_path = task3 / pole / 'legacy_reference.npz'
        try:
            with np.load(legacy_path, allow_pickle=False) as data:
                arrays = [data[key] for key in data.files]
                check(f'{pole}_legacy_reference_finite',
                      bool(arrays) and all(np.isfinite(a).all() for a in arrays),
                      {'path': str(legacy_path), 'keys': data.files})
        except (OSError, ValueError, KeyError) as exc:
            check(f'{pole}_legacy_reference_finite', False, str(exc))

    protocol = read_json(task3 / 'inputs/protocol.json', 'protocol_readable')
    required_protocol = {'margin': 0.1, 'ddof': 0, 'normalization_epsilon': 1e-8,
        'variance_clamp_min': 1e-8, 'variance_upper_cap': None,
        'KL_direction': 'image posterior || aggregate reference', 'W2': 'squared',
        'EA_gap': 'normalized mild - normalized severe',
        'EH_gap': 'normalized severe - normalized mild', 'MOS_used': False}
    for key, expected in required_protocol.items():
        check('task3_protocol_' + key, key in protocol and protocol[key] == expected,
              {'expected': expected, 'actual': protocol.get(key)})
    split = read_json(task3 / 'inputs/split_audit.json', 'split_audit_readable')
    check('saved_split_exclusion_audit', split.get('status') == 'PASS' and
          split.get('reference_identity_split_overlap') == 0 and
          split.get('train_val_byte_overlap') == 0 and split.get('train_val_path_overlap') == 0,
          {'source': str(task3 / 'inputs/split_audit.json'),
           'scope': 'Previously verified exclusion audit; no image bytes opened by this audit.'})
    failures = [c for c in checks if not c['passed']]
    return json_safe({'status': 'BLOCKED_PREREQUISITES' if failures else 'PREREQUISITES_VERIFIED',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'Day 12 Task 4 prerequisite audit',
        'task4_complete': False, 'training_runs_started_by_this_audit': 0,
        'optimizer_steps_by_this_audit': 0, 'holdout_opened_by_this_audit': False,
        'checkpoint_loading_scope': 'SHA-256 verified against Task 3; no torch deserialization or model forward pass.',
        'checks': checks, 'failures': failures, 'hash_checks': hash_checks,
        'normalization_records': normalization_records,
        'note': 'A passing prerequisite audit alone does not implement or complete the 16-run experiment.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent / 'config.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = audit(config)
    result['config_sha256'] = sha256(args.config)
    result['auditor_sha256'] = sha256(__file__)
    out = args.config.parent
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = out / 'audit' / stamp
    archive.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    (archive / 'prerequisite_audit.json').write_text(payload)
    (out / 'prerequisite_audit.json').write_text(payload)
    lines = ['DAY 12 TASK 4: PREREQUISITE AUDIT', 'STATUS: ' + result['status'], '',
        'The PDF requires Task 3 COMPLETE for both poles before Task 4 training.',
        f"Artifact hashes checked: {len(result['hash_checks'])}; failures: {sum(not h['passed'] for h in result['hash_checks'])}.",
        f"Frozen normalization records available: {len(result['normalization_records'])}/8.",
        'Training runs started: 0. Model updates: 0. Holdout opened in this task: false.', '',
        'FAILED PREREQUISITES']
    for failure in result['failures']:
        lines.append('- ' + failure['check'] + ': ' + json.dumps(failure['detail']))
    lines += ['', 'WHAT IS NEEDED',
        'Recover the original EH reference-image list with verifiable sampling/preprocessing provenance,',
        'or obtain an explicit protocol revision defining a replacement reference pool.',
        'Then complete Task 2 EH aggregate estimation and Task 3 EH KL/W2/BHATT checks,',
        'freeze all eight training-statistics records, and prepare a new Task 4 input lock.',
        'Do not edit the old hash records to make a failed audit pass.', '',
        'The fine-tuned EH checkpoint is present; its saved legacy mu_ref/Sigma_ref',
        'do not contain the missing per-image posterior variance information.',
        'No training or selection/holdout implementation has been executed.',
        'See requirements.md for every remaining PDF requirement.', '']
    report = '\n'.join(lines)
    (archive / 'execution_report.txt').write_text(report)
    (out / 'execution_report.txt').write_text(report)
    # Export only present, valid, hash-verified frozen records. Missing values are never synthesized.
    failed_hashes = {h['path'] for h in result['hash_checks'] if not h['passed']}
    for record in result['normalization_records']:
        if record['valid'] and record['source'] not in failed_hashes:
            path = out / 'training_stats' / f"{record['pole']}_{record['distance']}.json"
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(report)
    return 2 if result['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
