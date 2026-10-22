from pathlib import Path
import tempfile
import unittest

from tools import windows_msvc as msvc


class WindowsMsvcTests(unittest.TestCase):
    def fixture(self, base):
        redist = base / "14.44.35112/x64"
        crt = redist / "Microsoft.VC143.CRT"
        omp = redist / "Microsoft.VC143.OpenMP"
        crt.mkdir(parents=True)
        omp.mkdir()
        for name in ("msvcp140.dll", "vcruntime140.dll", "msvcp140_atomic_wait.dll"):
            (crt / name).write_bytes(name.encode())
        (omp / "vcomp140.dll").write_bytes(b"release OpenMP")
        native = base / "runtime/build/cpu/bin"
        native.mkdir(parents=True)
        (native / "llama-server.exe").write_bytes(b"native image")
        graph = {
            "llama-server.exe": {"MSVCP140.dll", "VCOMP140.DLL", "KERNEL32.dll"},
            "msvcp140.dll": {"vcruntime140.dll", "msvcp140_atomic_wait.dll"},
            "vcruntime140.dll": {"api-ms-win-crt-runtime-l1-1-0.dll"},
        }
        def reader(path, *, include_delay):
            self.assertTrue(include_delay)
            return graph.get(path.name.lower(), set())
        return redist, native, graph, reader

    def test_required_transitive_crt_and_openmp_are_copied_and_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            redist, native, _, reader = self.fixture(base)
            gui = base / "gui"
            report = msvc.stage([native], gui, redist, reader)
            self.assertEqual({r["name"] for r in report[str(native)]["files"]},
                             {"msvcp140.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll", "vcomp140.dll"})
            self.assertFalse((gui / "vcomp140.dll").exists())
            msvc.verify(native, reader)
            msvc.verify(gui, reader)

    def test_missing_transitive_runtime_fails_before_copying_any_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            redist, native, _, reader = self.fixture(base)
            (redist / "Microsoft.VC143.CRT/msvcp140_atomic_wait.dll").unlink()
            before = list(native.iterdir())
            with self.assertRaisesRegex(ValueError, "msvcp140_atomic_wait.dll"):
                msvc.stage([native], base / "gui", redist, reader)
            self.assertEqual(list(native.iterdir()), before)
            self.assertFalse((base / "gui").exists())

    def test_verifier_rejects_missing_and_tampered_app_local_dlls(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            redist, native, _, reader = self.fixture(base)
            msvc.stage([native], base / "gui", redist, reader)
            file = native / "vcruntime140.dll"
            original = file.read_bytes()
            file.write_bytes(b"changed runtime")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                msvc.verify(native, reader)
            file.write_bytes(original)
            file.unlink()
            with self.assertRaisesRegex(ValueError, "Missing release x64"):
                msvc.verify(native, reader)

    def test_debug_runtime_cannot_be_satisfied_by_an_installed_system_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            redist, native, graph, reader = self.fixture(base)
            graph["llama-server.exe"] = {"MSVCP140D.dll"}
            (base / "msvcp140d.dll").write_bytes(b"not a redistributable")
            with self.assertRaisesRegex(ValueError, "msvcp140d.dll"):
                msvc.stage([native], base / "gui", redist, reader)
            (redist / "Microsoft.VC143.CRT/msvcp140d.dll").write_bytes(b"misplaced debug runtime")
            with self.assertRaisesRegex(ValueError, "Debug MSVC runtime"):
                msvc.stage([native], base / "gui", redist, reader)

    def test_stale_runtime_is_removed_and_an_unlisted_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            redist, native, _, reader = self.fixture(base)
            stale = native / "concrt140.dll"
            stale.write_bytes(b"old unneeded runtime")
            msvc.stage([native], base / "gui", redist, reader)
            self.assertFalse(stale.exists())
            stale.write_bytes(b"unlisted runtime")
            with self.assertRaisesRegex(ValueError, "manifest"):
                msvc.verify(native, reader)

    def test_onecore_debug_and_wrong_architecture_are_not_redist_sources(self):
        for suffix in ("onecore/x64", "debug_nonredist/x64", "x86"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / suffix
                root.mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, "release desktop x64"):
                    msvc.redist_files(root)


if __name__ == "__main__":
    unittest.main()
