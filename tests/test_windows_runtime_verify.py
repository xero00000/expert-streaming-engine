from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / '.github/scripts/verify-windows-runtime.py'
SPEC = importlib.util.spec_from_file_location('verify_windows_runtime', SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
# These control tests mock PE import reading. They must not add a Windows-only
# parser dependency to ordinary Linux launcher regression discovery.
if importlib.util.find_spec('pefile') is None:
    with mock.patch.dict(sys.modules, {'pefile': ModuleType('pefile')}):
        SPEC.loader.exec_module(VERIFY)
else:
    SPEC.loader.exec_module(VERIFY)


class WindowsCpuRuntimeVerificationTests(unittest.TestCase):
    def fixture(self, directory: Path):
        runtime = directory / 'runtime'
        cpu = runtime / 'build/cpu/bin'
        cpu.mkdir(parents=True)
        for name in ('llama-server.exe', 'ggml.dll', 'llama.dll', 'mtmd.dll'):
            (cpu / name).touch()
        (cpu / 'ese-runtime.json').write_text('{"cuda":false}')
        build = directory / 'cpu-build'
        build.mkdir()
        (build / 'CMakeCache.txt').write_text('GGML_CUDA:BOOL=OFF\nESE_QWEN4EXP_PROBE_GATE:BOOL=OFF\n')
        launcher = runtime / 'ese.exe'
        launcher.touch()
        return runtime, cpu, build, launcher

    def test_cpu_pe_check_runs_native_smokes_and_driverless_launcher_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, cpu, build, launcher = self.fixture(Path(temp))
            report = json.dumps({'server':str(cpu / 'llama-server.exe'), 'server_cuda':False})
            calls = []
            def execute(argv, **kwargs):
                calls.append((argv, kwargs))
                return subprocess.CompletedProcess(argv, 1 if argv[-1]=='--help' else 0, stdout=report if argv[0] == str(launcher) else 'usage: server --model --port', stderr='')
            with mock.patch.object(VERIFY, 'pe_imports', return_value={'kernel32.dll'}) as imports, mock.patch.object(VERIFY.subprocess, 'run', side_effect=execute):
                VERIFY.verify_cpu_runtime(runtime, build, launcher)
            self.assertEqual(len(calls), 3)
            self.assertEqual([call[0][-1] for call in calls], ['--version','--help','--json'])
            self.assertTrue(all(call[1]['env']['CUDA_VISIBLE_DEVICES'] == '-1' for call in calls))
            self.assertTrue(all(call.kwargs['include_delay'] for call in imports.call_args_list))

    def test_cpu_help_smoke_rejects_loader_errors_and_crashes(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, cpu, build, launcher = self.fixture(Path(temp))
            for code, output in [(1, 'Cannot load llama.dll'), (2, 'usage: server --model --port'), (-1073741515, '')]:
                with self.subTest(code=code):
                    def execute(argv, **kwargs):
                        return subprocess.CompletedProcess(argv, code if argv[-1]=='--help' else 0, stdout=output, stderr='')
                    with mock.patch.object(VERIFY, 'pe_imports', return_value={'kernel32.dll'}), mock.patch.object(VERIFY.subprocess, 'run', side_effect=execute):
                        with self.assertRaisesRegex(SystemExit, 'help smoke failed'):
                            VERIFY.verify_cpu_runtime(runtime, build, launcher)

    def test_cpu_pe_check_rejects_cuda_imports_before_any_executable_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, cpu, build, launcher = self.fixture(Path(temp))
            with mock.patch.object(VERIFY, 'pe_imports', return_value={'nvcuda.dll'}), mock.patch.object(VERIFY.subprocess, 'run') as execute:
                with self.assertRaisesRegex(SystemExit, 'imports CUDA dependencies'):
                    VERIFY.verify_cpu_runtime(runtime, build, launcher)
                execute.assert_not_called()

    def test_cpu_pe_check_rejects_wrong_configuration_and_open_probe_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, cpu, build, launcher = self.fixture(Path(temp))
            for cache, error in [('GGML_CUDA:BOOL=ON\n','disable CUDA'),('GGML_CUDA:BOOL=OFF\nESE_QWEN4EXP_PROBE_GATE:BOOL=ON\n','gate closed')]:
                with self.subTest(cache=cache), mock.patch.object(VERIFY.subprocess, 'run') as execute:
                    (build / 'CMakeCache.txt').write_text(cache)
                    with self.assertRaisesRegex(SystemExit, error):
                        VERIFY.verify_cpu_runtime(runtime, build, launcher)
                    execute.assert_not_called()

    def test_cpu_pe_check_rejects_launcher_selecting_the_cuda_binary(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, cpu, build, launcher = self.fixture(Path(temp))
            report = json.dumps({'server':str(runtime / 'build/bin/llama-server.exe'), 'server_cuda':True})
            with mock.patch.object(VERIFY, 'pe_imports', return_value={'kernel32.dll'}), mock.patch.object(VERIFY.subprocess, 'run', side_effect=lambda argv, **kwargs: subprocess.CompletedProcess(argv,0,stdout='usage: server --model --port' if argv[-1]=='--help' else report,stderr='')):
                with self.assertRaisesRegex(SystemExit, 'did not select the separate CPU runtime'):
                    VERIFY.verify_cpu_runtime(runtime, build, launcher)


if __name__ == '__main__':
    unittest.main()
