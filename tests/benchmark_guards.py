"""Regression checks for benchmark safeguards, also run with python -O."""
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'benchmarks/production_tts'
spec = importlib.util.spec_from_file_location('production_audit', SCRIPTS/'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_counts_failures_and_checks_duplicate_energies(self):
        case = dict(points=[[0, 0]], epsilon_r=5.6, lambda_tf=5.,
                    external_potential=[0.], fixed_potential=[0.], mu=-.32)
        good = dict(case='one', config=[-1], energy=0., valid=True)
        rows = [good, dict(good, energy=1.), dict(good, energy=float('nan')),
                dict(good, config=[0]), dict(good, config=[2])]
        result = audit.audit_rows(rows, {'one': case})
        self.assertEqual(result['checked_rows'], 5)
        self.assertEqual(result['rejects'], 4)
        self.assertEqual(len(result['failures']), 4)
        self.assertEqual(result['unique_states'], 2)

    def test_cli_writes_failure_and_exits_nonzero(self):
        cases = json.loads((SCRIPTS/'cases.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p/'rows.jsonl').write_text(json.dumps(dict(case=cases[0]['name'],
                config=[2], energy=0., valid=True))+'\n')
            command = [sys.executable]
            if sys.flags.optimize:
                command.append('-O')
            result = subprocess.run(command+[str(SCRIPTS/'audit.py'), directory],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(json.loads((p/'audit.json').read_text())['rejects'], 1)


class CampaignTests(unittest.TestCase):
    def run_failure(self, message, process=None, clock=None, extra=()):
        with tempfile.TemporaryDirectory() as directory:
            argv = [str(SCRIPTS/'run.py'), sys.executable,
                    str(Path(directory)/'out'), '--jobs', '1', *extra]
            with patch.object(sys, 'argv', argv), patch('subprocess.Popen', return_value=process) as spawn, patch('time.monotonic', side_effect=clock or [0, 0]), patch('os.killpg') as kill:
                with self.assertRaisesRegex(RuntimeError, message):
                    runpy.run_path(str(SCRIPTS/'run.py'), run_name='__main__')
            return spawn, kill

    def test_campaign_cap_prevents_launch(self):
        spawn, _ = self.run_failure('Campaign cap', clock=[0, 1801])
        spawn.assert_not_called()

    def test_bad_exit(self):
        process = Mock(returncode=7)
        process.communicate.return_value = ('', 'failed')
        self.run_failure('solver exited 7', process)

    def test_missing_rows(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ('', '')
        self.run_failure('expected 1 rows, got 0', process)

    def test_wrong_id(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ('{"id":"wrong"}\n', '')
        self.run_failure('ID does not match', process)

    def test_timeout_uses_remaining_budget_and_kills_group(self):
        process = Mock(pid=43210)
        process.communicate.side_effect = [subprocess.TimeoutExpired('solver', .25), ('', '')]
        _, kill = self.run_failure('time limit', process, clock=[0, 1799.75])
        self.assertEqual(process.communicate.call_args_list[0].kwargs['timeout'], .25)
        kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
