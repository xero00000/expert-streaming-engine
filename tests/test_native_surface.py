from __future__ import annotations

import unittest
from pathlib import Path


TRANSIENT_FAILURE_BOUNDARIES = (
    "after-transient-prepare",
    "after-transient-publish",
)


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
            "--transient-mtp-mib",
            "--transient-mmproj-mib",
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

    def test_resource_snapshot_is_collected_on_the_inference_owner_thread(self) -> None:
        root = Path(__file__).resolve().parents[1]
        public_header = (root / "include" / "llama.h").read_text(encoding="utf-8")
        backend_header = (root / "ggml" / "include" / "ggml-backend.h").read_text(
            encoding="utf-8"
        )
        server = (root / "examples" / "server" / "server.cpp").read_text(
            encoding="utf-8"
        )
        context = (root / "examples" / "server" / "server-context.cpp").read_text(
            encoding="utf-8"
        )
        validator = (root / "scripts" / "validate-phase-d-rebalance.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("llama_resource_get_snapshot", public_header)
        self.assertIn("llama_expert_cache_resize", public_header)
        self.assertIn("llama_kv_cache_transaction_t", public_header)
        self.assertIn("llama_kv_cache_prepare_resize", public_header)
        self.assertIn("llama_kv_cache_prepare_retier", public_header)
        self.assertIn("llama_kv_cache_transaction_publish", public_header)
        self.assertIn("llama_kv_cache_transaction_rollback", public_header)
        self.assertIn("llama_kv_cache_transaction_finalize", public_header)
        self.assertIn("llama_kv_cache_transaction_free", public_header)
        self.assertIn("llama_expert_cache_transaction_t", public_header)
        self.assertIn("llama_expert_cache_prepare_resize", public_header)
        self.assertIn("llama_expert_cache_transaction_publish", public_header)
        self.assertIn("llama_expert_cache_transaction_rollback", public_header)
        self.assertIn("llama_expert_cache_transaction_finalize", public_header)
        self.assertIn("llama_expert_cache_transaction_free", public_header)
        llama_source = (root / "src" / "llama.cpp").read_text(encoding="utf-8")
        self.assertIn("target_bytes_per_device > 0", llama_source)
        self.assertIn("cparams.fused_mmad = false", llama_source)
        self.assertIn("cparams.fused_mmad = transaction->previous_fused_mmad", llama_source)
        self.assertIn("active_expert_cache_transaction", llama_source)
        self.assertIn("ggml_backend_sched_get_resource_device_stats", backend_header)
        self.assertIn("ggml_backend_sched_replace_expert_cache", backend_header)
        self.assertIn("ggml_backend_sched_expert_cache_txn_t", backend_header)
        self.assertIn("ggml_backend_sched_expert_cache_prepare", backend_header)
        self.assertIn("ggml_backend_sched_expert_cache_publish", backend_header)
        self.assertIn("ggml_backend_sched_expert_cache_rollback", backend_header)
        self.assertIn("ggml_backend_sched_expert_cache_finalize", backend_header)
        self.assertIn("ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES", backend_header)
        self.assertIn('svr->Get("/v1/ese/resources"', server)
        self.assertIn('svr->Post("/v1/ese/resources/rebalance"', server)
        self.assertIn("task.type = SERVER_TASK_TYPE_METRICS", server)
        self.assertIn("task.type = SERVER_TASK_TYPE_RESOURCE_REBALANCE", server)
        self.assertIn("common_resource_rebalance_target", server)
        self.assertIn('"kv-and-expert"', server)
        self.assertIn('res.data["resources"]', context)
        self.assertIn('"mutation_enabled", true', context)
        self.assertIn('"idle-atomic-multi-pool"', context)
        self.assertIn('"combined_mutation", true', context)
        self.assertIn("llama_kv_cache_prepare_resize", context)
        self.assertIn("llama_expert_cache_prepare_resize", context)
        self.assertIn("llama_kv_cache_transaction_publish", context)
        self.assertIn("llama_expert_cache_transaction_publish", context)
        self.assertIn("llama_expert_cache_transaction_rollback", context)
        self.assertIn("llama_kv_cache_transaction_rollback", context)
        self.assertIn("llama_expert_cache_transaction_finalize", context)
        self.assertIn("llama_kv_cache_transaction_finalize", context)
        self.assertIn("ESE_RESOURCE_REBALANCE_FAIL_STAGE", context)
        self.assertIn("published_resource_state", context)
        self.assertIn("logical_publish_lock(resource_plan_mutex)", context)
        self.assertIn("compare_exchange_strong", context)
        for boundary in (
            "after-kv-prepare",
            "after-expert-prepare",
            "after-kv-publish",
            "after-expert-publish",
            "before-logical-publish",
        ):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, context)
                self.assertIn(boundary, validator)
        self.assertIn("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", validator)
        self.assertIn("ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_COPIES", validator)
        self.assertIn("ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_DEVICES", validator)
        self.assertIn("ESE_RESOURCE_REBALANCE_FAIL_STAGE", validator)
        self.assertIn(
            "ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES",
            validator,
        )
        self.assertIn('"scope": "kv-and-expert"', validator)
        self.assertIn("exact_pre_completion_state_restored", validator)
        self.assertIn("same_process_retry_committed", validator)
        self.assertIn("same_process_restore_committed", validator)
        self.assertIn("rebalance_combined_observing_props", validator)
        self.assertIn("busy_rejection_http", validator)

        # Physical publication stays reversible until both pools are live;
        # rollback runs in reverse order and the logical plan is published last.
        self.assertLess(
            context.index("llama_kv_cache_prepare_resize"),
            context.index("llama_expert_cache_prepare_resize"),
        )
        self.assertLess(
            context.index("llama_kv_cache_transaction_publish"),
            context.index("llama_expert_cache_transaction_publish"),
        )
        self.assertLess(
            context.index("llama_expert_cache_transaction_rollback"),
            context.index("llama_kv_cache_transaction_rollback"),
        )
        self.assertLess(
            context.index("llama_kv_cache_transaction_finalize"),
            context.index(
                "params_base.resolved_resource_plan_json = std::move(target_plan_text)"
            ),
        )

        backend = (root / "ggml" / "src" / "ggml-backend.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("active_expert_cache_layout_catalog", backend)
        self.assertIn("active_expert_cache_route_capacity", backend)
        self.assertIn("nullptr, cache.ctx", backend)

    def test_ordinary_server_launch_skips_the_speculative_decode_probe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        context = (root / "examples" / "server" / "server-context.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "if (requested_spec && !params_base.dry_run)",
            context,
        )

    def test_transient_policy_crosses_the_native_runtime_surface(self) -> None:
        root = Path(__file__).resolve().parents[1]
        planner_header = (root / "common" / "resource-planner.h").read_text(
            encoding="utf-8"
        )
        planner = (root / "common" / "resource-planner.cpp").read_text(
            encoding="utf-8"
        )
        manager_header = (root / "common" / "transient-module-manager.h").read_text(
            encoding="utf-8"
        )
        manager = (root / "common" / "transient-module-manager.cpp").read_text(
            encoding="utf-8"
        )
        server = (root / "examples" / "server" / "server.cpp").read_text(
            encoding="utf-8"
        )
        context = (root / "examples" / "server" / "server-context.cpp").read_text(
            encoding="utf-8"
        )
        context_header = (root / "examples" / "server" / "server-context.h").read_text(
            encoding="utf-8"
        )
        task_header = (root / "examples" / "server" / "server-task.h").read_text(
            encoding="utf-8"
        )
        server_common_header = (
            root / "examples" / "server" / "server-common.h"
        ).read_text(encoding="utf-8")
        server_common = (
            root / "examples" / "server" / "server-common.cpp"
        ).read_text(encoding="utf-8")
        validator = (root / "scripts" / "validate-phase-d-rebalance.py").read_text(
            encoding="utf-8"
        )

        for policy in ("shared", "mtp-only", "multimodal-only", "off"):
            with self.subTest(policy=policy):
                self.assertIn(policy, planner)
                self.assertIn(policy, server)
                self.assertIn(policy, validator)
        for field in (
            "transient_device",
            "transient_policy",
            "transient_mtp_bytes",
            "transient_multimodal_bytes",
            "transient_capacity_bytes",
            "prepared_transient_bytes",
            "prepares_transient",
        ):
            with self.subTest(field=field):
                self.assertIn(field, planner_header)
                self.assertIn(field, planner)

        for operation in (
            "prepare_policy",
            "publish_policy",
            "rollback_policy",
            "finalize_policy",
            "free_policy",
        ):
            with self.subTest(operation=operation):
                self.assertIn(operation, manager_header)
                self.assertIn(
                    f"common_transient_module_manager::{operation}", manager
                )
        for callback in (
            "prepare_residency",
            "publish_residency",
            "rollback_residency",
            "finalize_residency",
            "free_residency",
        ):
            with self.subTest(callback=callback):
                self.assertIn(callback, manager_header)
                self.assertIn(f"mtp_desc.{callback}", context)
                self.assertIn(f"mmproj_desc.{callback}", context)

        self.assertIn('"transient_policy"', server)
        self.assertIn('"kv-expert-and-transient"', server)
        self.assertIn('svr->Get("/v1/props"', server)
        self.assertIn('mutable_pools.push_back("transient")', context)
        self.assertIn('res.data["resources"]["transient"]', context)
        self.assertIn("struct server_transient_lease_group", task_header)
        self.assertIn("server_transient_prompt_guard", server)
        self.assertIn("transient_prompt.attach_task", server)
        self.assertIn("transient_prompt.handoff()", server)
        self.assertIn("server_complete_transient_lease_group", context)
        self.assertIn("group->commit = group->commit || commit", context)
        self.assertIn(
            "manager->release(group->leases[i - 1], group->commit", context
        )
        self.assertIn("server_send_transient_rpc_error", context)
        self.assertIn("queue_tasks.take_pending_tasks", context)
        self.assertIn("task.transient_lease_group == nullptr", context)
        system_prompt_block_start = context.index(
            'if (task.data.contains("system_prompt"))'
        )
        system_prompt_block_end = context.index(
            "if (transient_manager != nullptr)", system_prompt_block_start
        )
        system_prompt_block = context[
            system_prompt_block_start:system_prompt_block_end
        ]
        self.assertIn("system_prompt_set(sys_prompt)", system_prompt_block)
        self.assertIn(
            "task.transient_lease_group->commit = true", system_prompt_block
        )
        self.assertLess(
            system_prompt_block.index("system_prompt_set(sys_prompt)"),
            system_prompt_block.index(
                "task.transient_lease_group->commit = true"
            ),
        )
        for media_prefix_surface in (
            "media_safe_prefix_size",
            "shrink_to_media_safe_prefix",
        ):
            with self.subTest(media_prefix_surface=media_prefix_surface):
                self.assertIn(media_prefix_surface, server_common_header)
                self.assertIn(f"server_tokens::{media_prefix_surface}", server_common)
                self.assertIn(media_prefix_surface, context)
        self.assertIn("target_cache_prefixes", context)
        self.assertLess(
            context.index("std::vector<size_t> target_cache_prefixes"),
            context.index("llama_kv_cache_prepare_resize"),
        )
        self.assertIn('{"expected_previous_plan", current_plan_text}', server)
        self.assertIn("std::mutex ese_rebalance_mutex", server)
        self.assertIn("rebalance_lock(ese_rebalance_mutex)", server)
        rebalance_case = context.index(
            "case SERVER_TASK_TYPE_RESOURCE_REBALANCE"
        )
        expected_plan_read = context.index(
            'task.data.at("expected_previous_plan")', rebalance_case
        )
        expected_plan_lock = context.index(
            "plan_lock(resource_plan_mutex)", expected_plan_read
        )
        stale_plan_retry = context.index(
            "retry from a fresh snapshot", expected_plan_lock
        )
        self.assertIn(
            "params_base.resolved_resource_plan_json != expected_previous_plan",
            context[expected_plan_lock:stale_plan_retry],
        )
        physical_prepare = context.index(
            "llama_kv_cache_prepare_resize", stale_plan_retry
        )
        self.assertLess(expected_plan_read, expected_plan_lock)
        self.assertLess(expected_plan_lock, stale_plan_retry)
        self.assertLess(stale_plan_retry, physical_prepare)
        for fail_stop_release in (
            "failed to retire hidden transient residency lease",
            "slot could not retire hidden transient residency lease",
            "owner RPC could not retire transient residency lease",
        ):
            with self.subTest(fail_stop_release=fail_stop_release):
                marker = context.index(fail_stop_release)
                self.assertIn("GGML_ABORT", context[marker - 120:marker])
        run_release_marker = manager.index(
            "transient run could not retire its hidden lease"
        )
        self.assertIn("GGML_ABORT", manager[run_release_marker - 120:run_release_marker])
        for task_type in (
            "SERVER_TASK_TYPE_TRANSIENT_ACQUIRE",
            "SERVER_TASK_TYPE_TRANSIENT_RELEASE",
        ):
            with self.subTest(task_type=task_type):
                self.assertIn(task_type, task_header)
                self.assertIn(task_type, context)
        for owner_rpc in (
            "acquire_transient_on_owner",
            "release_transient_on_owner",
        ):
            with self.subTest(owner_rpc=owner_rpc):
                self.assertIn(owner_rpc, context_header)
                self.assertIn(f"server_context::{owner_rpc}", context)
                self.assertIn(f"ctx.{owner_rpc}", server)
        for capability in (
            "active_leases",
            "pending_restores",
            "reconfiguration_open",
            "resident_configured_bound_bytes",
            "configured-peak-bounds-not-backend-measurement",
            "configured-main-gpu-only",
        ):
            with self.subTest(capability=capability):
                self.assertIn(capability, context)

        for boundary in TRANSIENT_FAILURE_BOUNDARIES:
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, context)
                self.assertIn(boundary, validator)
        for evidence in (
            '"scope": "kv-expert-and-transient"',
            "exact_transient_owner_restored",
            "exact_transient_round_trip",
            "rebalance_all_observing_views",
            "require_runtime_capabilities",
            "require_transient_preparation_peak",
            "multimodal_completion",
            "multimodal_completion_observing_resources",
            "malformed_multimodal_completion",
            "legacy_multimodal_batch_payload",
            "validate_multitask_multimodal_handoff",
            "exact_prior_mtp_restored",
            "multimodal_committed",
            "validate_deferred_multimodal_cancel",
            "require_zero_transient_ownership",
            "cancelled_before_launch",
            "text_rebuilt_fresh_mtp_owner",
            "deterministic_prefix_parity",
            "system_prompt_multimodal_launch_failure",
            "validate_system_prompt_launch_failure",
            "stale_prior_mtp_not_restored",
            "next_text_rebuilt_mtp",
            "validate_serialized_disjoint_rebalances",
            "serialized_by_http_mutex",
            "coherent_combined_plan",
            "allowed_simple_states",
            "multimodal_rejection_http",
            "projector_modalities",
            "policy_modalities",
            "restore_shared_owner",
            '"transient_policy": policy',
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, validator)

        # Preparation and publication follow KV -> expert -> transient, while
        # rollback is the exact reverse order before any logical-plan update.
        self.assertLess(
            context.index("llama_expert_cache_prepare_resize"),
            context.index("transient_manager->prepare_policy"),
        )
        self.assertLess(
            context.index("llama_expert_cache_transaction_publish"),
            context.index("transient_manager->publish_policy"),
        )
        self.assertLess(
            context.index("transient_owner->rollback_policy"),
            context.index("llama_expert_cache_transaction_rollback"),
        )

    def test_fused_moe_staging_allocates_bytes_not_elements(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cuda = (root / "ggml" / "src" / "ggml-cuda.cu").read_text(encoding="utf-8")
        self.assertIn(
            "final_dst_contiguous.alloc(ggml_nbytes(next))",
            cuda,
        )
        self.assertNotIn(
            "final_dst_contiguous.alloc(ggml_nelements(next))",
            cuda,
        )

    def test_graph_reuse_observes_hybrid_guard_transitions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        llama = (root / "src" / "llama.cpp").read_text(encoding="utf-8")
        self.assertIn("expert_hybrid_guard_status;", llama)
        self.assertIn(
            "the_prev->expert_hybrid_guard_status &&\n           update_cache_copies()",
            llama,
        )

    def test_compact_expert_cache_preserves_safe_graph_fallbacks(self) -> None:
        root = Path(__file__).resolve().parents[1]
        backend = (root / "ggml" / "src" / "ggml-backend.cpp").read_text(
            encoding="utf-8"
        )
        llama = (root / "src" / "llama.cpp").read_text(encoding="utf-8")
        self.assertIn(
            "const bool full_tensor_fallback_available = input_cpy->ne[2] == input->ne[2]",
            backend,
        )
        self.assertIn(
            "sched->active_expert_cache_capacity_bytes == 0 && full_tensor_fallback_available",
            backend,
        )
        self.assertIn(
            "cparams.expert_vram_cache_bytes > 0 || legacy_expert_cache_requested()",
            llama,
        )
        self.assertIn("ctx->cparams.fused_mmad = false", llama)


if __name__ == "__main__":
    unittest.main()
