"""Exercise the entry gate with temporary synthetic metadata, never research images."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_prerequisites import audit, sha256


class PrerequisiteGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'inputs').mkdir()
        self.write('preflight_status.json', {'task_complete': True,
            'ready_for_matched_distance_training': True, 'status': 'COMPLETE'})
        self.write('inputs/protocol.json', {'margin': 0.1, 'ddof': 0,
            'normalization_epsilon': 1e-8, 'variance_clamp_min': 1e-8,
            'variance_upper_cap': None, 'KL_direction': 'image posterior || aggregate reference',
            'W2': 'squared', 'EA_gap': 'normalized mild - normalized severe',
            'EH_gap': 'normalized severe - normalized mild', 'MOS_used': False})
        self.write('inputs/split_audit.json', {'status': 'PASS',
            'reference_identity_split_overlap': 0, 'train_val_byte_overlap': 0, 'train_val_path_overlap': 0})
        for pole in ('EA', 'EH'):
            (self.root / pole).mkdir()
            self.write(pole+'/status.json', {'complete': True, 'status': 'COMPLETE'})
            self.write(pole+'/frozen_training_normalization.json', {
                'ddof': 0, 'epsilon': 1e-8, 'fit_split': 'unique severity-training images',
                'n_unique_images': 2, 'distances': {key: {'mean': 1., 'std_ddof0': 0.5,
                'nonconstant': True, 'nonfinite_count': 0, 'finite_count': 2, 'n': 2}
                for key in ('D0','KL','W2_squared','Bhattacharyya')}})
            np.savez(self.root/pole/'aggregate_reference.npz', mu_agg=np.zeros(100),
                v_between=np.ones(100), v_within=np.ones(100), v_aggregate_raw=np.full(100,2.),
                v_agg=np.full(100,2.), variance_epsilon=1e-8, ddof=0, n_images=2)
            np.savez(self.root/pole/'legacy_reference.npz',mu_ref=np.zeros(100),Sigma_ref=np.ones(100))
            (self.root/pole/'init.pth').write_bytes(b'synthetic hash fixture, never deserialized')
        self.config = {'task3_directory': str(self.root), 'training': {'unique_images': 2},
            'checkpoints': {p: str(self.root/p/'init.pth') for p in ('EA','EH')},
            'aggregate_references': {p: str(self.root/p/'aggregate_reference.npz') for p in ('EA','EH')},
            'expected_hashes': {str(p): sha256(p) for p in self.root.rglob('*') if p.is_file()}}

    def write(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def mutate(self, name, function):
        obj=json.loads((self.root/name).read_text()); function(obj); self.write(name,obj)

    def failed_names(self, result):
        self.assertEqual(result['status'], 'BLOCKED_PREREQUISITES')
        return {x['check'] for x in result['failures']}

    def test_complete_consistent_artifacts_pass_prerequisites_only(self):
        result=audit(self.config)
        self.assertEqual(result['status'],'PREREQUISITES_VERIFIED')
        self.assertEqual(len(result['normalization_records']),8)
        self.assertFalse(result['task4_complete'])
        self.assertFalse(result['holdout_opened_by_this_audit'])

    def test_partial_eh_does_not_pass_when_available_checks_pass(self):
        self.mutate('EH/status.json',lambda obj: obj.update(complete=False,available_checks_passed=True))
        self.assertIn('EH_task3_complete',self.failed_names(audit(self.config)))

    def test_missing_reference_is_recorded(self):
        (self.root/'EH/aggregate_reference.npz').unlink()
        names=self.failed_names(audit(self.config))
        self.assertIn('EH_aggregate_reference',names)
        self.assertIn('frozen_artifact_hashes',names)

    def test_changed_checkpoint_fails_hash_gate(self):
        (self.root/'EA/init.pth').write_bytes(b'different checkpoint')
        self.assertIn('frozen_artifact_hashes',self.failed_names(audit(self.config)))

    def test_missing_distance_stats_is_not_filled(self):
        self.mutate('EH/frozen_training_normalization.json',lambda obj: obj['distances'].pop('KL'))
        result=audit(self.config)
        self.assertIn('EH_KL_frozen_training_stats',self.failed_names(result))
        self.assertEqual(len(result['normalization_records']),7)

    def test_reversed_eh_polarity_fails(self):
        self.mutate('inputs/protocol.json',lambda obj: obj.update(EH_gap='normalized mild - normalized severe'))
        self.assertIn('task3_protocol_EH_gap',self.failed_names(audit(self.config)))

    def test_nearly_constant_normalization_fails(self):
        self.mutate('EA/frozen_training_normalization.json',lambda obj: obj['distances']['KL'].update(std_ddof0=1e-12))
        self.assertIn('EA_KL_frozen_training_stats',self.failed_names(audit(self.config)))

    def test_nonfinite_failure_remains_serializable(self):
        self.mutate('EA/frozen_training_normalization.json',lambda obj: obj['distances']['KL'].update(mean=float('nan')))
        result=audit(self.config)
        self.assertIn('EA_KL_frozen_training_stats',self.failed_names(result))
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
