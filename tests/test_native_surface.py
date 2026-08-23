from __future__ import annotations

import unittest
from pathlib import Path


class NativeSurfaceTests(unittest.TestCase):
    def test_policy_flags_exist_in_native_parser(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "common" / "common.cpp").read_text(encoding="utf-8")
        for flag in (
            "--fit",
            "--cpu-moe",
            "--n-cpu-moe",
            "--defer-experts",
            "--expert-ram-cache-mib",
            "--expert-ram-staging-mib",
            "--expert-storage-backend",
            "--expert-sidecar-only",
            "--expert-vram-cache-mib",
            "--expert-vram-reserve-mib",
            "--expert-prefill-staging-mib",
            "--expert-cache-min-observations",
            "--expert-hybrid-gpu-experts",
            "--expert-hybrid-cpu-ns-per-expert",
            "--expert-hybrid-upload-ns-per-expert",
            "--expert-hybrid-maximum-drift-ppm",
            "--expert-hybrid-minimum-cpu-calls",
            "--memory-policy",
            "--max-ram",
            "--reserve-vram",
            "--min-kv-quality",
            "--max-context",
            "--resource-preference",
            "--resource-plan-json",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, source)

    def test_launcher_does_not_reference_unported_cache_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "tools" / "ese.py").read_text(encoding="utf-8")
        self.assertNotIn('"--moe-cache"', launcher)
        self.assertNotIn('"--moe-cache-expert-parallel"', launcher)

    def test_hybrid_guard_parameters_cross_the_common_context_boundary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        common = (root / "common" / "common.cpp").read_text(encoding="utf-8")
        for field in (
            "expert_hybrid_cpu_ns_per_expert",
            "expert_hybrid_upload_ns_per_expert",
            "expert_hybrid_maximum_drift_ppm",
            "expert_hybrid_minimum_cpu_calls",
        ):
            with self.subTest(field=field):
                self.assertIn(f"cparams.{field} = params.{field};", common)

    def test_prefill_staging_crosses_the_native_context_boundary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        common = (root / "common" / "common.cpp").read_text(encoding="utf-8")
        llama = (root / "src" / "llama.cpp").read_text(encoding="utf-8")
        public_header = (root / "include" / "llama.h").read_text(encoding="utf-8")
        backend_header = (root / "ggml" / "include" / "ggml-backend.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("cparams.expert_prefill_staging_bytes =", common)
        self.assertIn(
            "cparams.expert_prefill_staging_bytes = params.expert_prefill_staging_bytes;",
            llama,
        )
        self.assertIn("expert_prefill_staging_bytes", public_header)
        self.assertIn("ggml_backend_sched_set_expert_prefill_staging", backend_header)
        self.assertIn("llama_model_largest_expert_layer", public_header)
        backend = (root / "ggml" / "src" / "ggml-backend.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("route_global_sync_fallbacks", backend)
        self.assertIn("ggml_backend_event_synchronize(route_event)", backend)

    def test_server_props_exposes_live_hybrid_guard_state(self) -> None:
        root = Path(__file__).resolve().parents[1]
        server = (root / "examples" / "server" / "server.cpp").read_text(encoding="utf-8")
        self.assertIn("llama_get_expert_hybrid_guard_status", server)
        self.assertIn('"expert_hybrid_guard"', server)
        self.assertIn('"cpu-drift"', server)
        self.assertIn('"upload-drift"', server)

    def test_graph_reuse_observes_hybrid_guard_transitions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        llama = (root / "src" / "llama.cpp").read_text(encoding="utf-8")
        self.assertIn("expert_hybrid_guard_status;", llama)
        self.assertIn(
            "the_prev->expert_hybrid_guard_status &&\n           update_cache_copies()",
            llama,
        )


if __name__ == "__main__":
    unittest.main()
