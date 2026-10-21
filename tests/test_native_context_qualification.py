"""The context probe must preserve document records and reject false retrieval."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

spec = importlib.util.spec_from_file_location(
    "native_context", Path(__file__).parents[1] / "scripts/qualify-native-context.py")
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)


class ContextQualificationTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal, "setitimer"), "POSIX wall-time budget")
    def test_wall_time_budget_interrupts_a_server_that_never_responds(self):
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                release.wait(5)

            def log_message(self, *args):
                pass

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as tmp:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
            thread.start()
            try:
                output = Path(tmp) / "evidence"
                result = subprocess.run([sys.executable, str(Path(context.__file__)),
                    "--url", f"http://127.0.0.1:{server.server_port}", "--output", str(output),
                    "--context", "4096", "--max-seconds", "0.1"],
                    capture_output=True, timeout=3)
                self.assertNotEqual(result.returncode, 0)
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["status"], "time_budget_exceeded")
                self.assertEqual(report["samples"], [])
            finally:
                release.set()
                server.shutdown()
                thread.join(timeout=2)

    def test_exact_64k_fixture_keeps_records_and_generation_suffix(self):
        fixture = context.build_fixture(
            lambda text: list(text.encode()),
            lambda messages: "<user>" + messages[0]["content"] + "</user><assistant>", 65536)
        text = bytes(fixture["tokens"]).decode()
        self.assertEqual(len(fixture["tokens"]), 65536)
        self.assertTrue(text.endswith("</user><assistant>"))
        for label, code in context.CODES.items():
            self.assertEqual(text.count(f"{label} access code is {code}."), 1)
        positions = fixture["record_positions"]
        self.assertLess(positions["ALPHA"], 65536 * .03)
        self.assertLess(abs(positions["BETA"] - 32768), 65536 * .03)
        self.assertGreater(positions["GAMMA"], 65536 * .97)

    def test_retrieval_requires_correct_assignment_and_complete_codes(self):
        context.check_retrieval("ALPHA=630184; BETA=297531; GAMMA=845072", context.CODES)
        for invalid in ["630184 297531 845072", "ALPHA=630184; BETA=845072; GAMMA=297531",
                        "ALPHA=630184; BETA=297531", "ALPHA=6301840; BETA=297531; GAMMA=845072",
                        "ALPHA=630184; ALPHA=000000; BETA=297531; GAMMA=845072"]:
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                context.check_retrieval(invalid, context.CODES)

    def test_template_loss_and_insufficient_space_reject(self):
        for render in [lambda messages: "missing document", lambda messages: context.PLACEHOLDER * 2]:
            with self.assertRaises(ValueError):
                context.build_fixture(lambda text: list(text.encode()), render, 65536)
        with self.assertRaises(ValueError):
            context.build_fixture(lambda text: list(text.encode()),
                                  lambda messages: messages[0]["content"], 100)


if __name__ == "__main__":
    unittest.main()
