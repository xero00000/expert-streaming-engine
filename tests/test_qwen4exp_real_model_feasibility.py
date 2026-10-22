import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/qwen4exp/native-recurrence-20261007"
FEASIBILITY = EVIDENCE / "real-model-feasibility-20261007.json"
INVENTORY = EVIDENCE / "coder-iq1m-inventory.json"


def byte_classes(inventory: dict) -> dict:
    total = ple = experts = 0
    for tensor in inventory["tensors"]:
        size = tensor["size_bytes"]
        total += size
        if tensor["name"] == "per_layer_token_embd.weight":
            ple += size
        elif tensor["name"].endswith("_exps.weight"):
            experts += size
    return {"total_bytes": total, "per_layer_token_embd_bytes": ple, "routed_experts_bytes": experts,
            "other_resident_bytes": total - ple - experts}


def test_feasibility_evidence_is_content_addressed_and_matches_the_inventory():
    document = json.loads(FEASIBILITY.read_text())
    digest = document.pop("evidence_sha256")
    assert hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == digest
    assert document["inventory_file"] == INVENTORY.name
    assert document["inventory_sha256"] == hashlib.sha256(INVENTORY.read_bytes()).hexdigest()

    inventory = json.loads(INVENTORY.read_text())
    assert document["tensor_inventory_sha256"] == inventory["tensor_inventory_sha256"]
    recomputed = byte_classes(inventory)
    for key, value in recomputed.items():
        assert document["model"][key] == value, key
    assert document["model"]["shards"] == len({t["shard"] for t in inventory["tensors"]}) == 2


def test_measured_classes_force_streaming_rather_than_a_resident_run():
    document = json.loads(FEASIBILITY.read_text())
    model, host = document["model"], document["host"]
    # The whole model cannot be resident on this host even with an otherwise idle
    # machine, so a resident full-model run is impossible, not merely tight.
    assert model["total_bytes"] > host["ram_total_bytes"]
    # The two streamable classes alone exceed the memory actually available next to
    # the launcher-owned Strata service, so a streaming path is mandatory and the
    # remaining resident set is what has to fit.
    streamable = model["per_layer_token_embd_bytes"] + model["routed_experts_bytes"]
    assert streamable > host["ram_available_bytes"]
    assert model["other_resident_bytes"] < host["ram_available_bytes"]
    assert streamable + model["other_resident_bytes"] == model["total_bytes"]
    # Device memory cannot substitute for the streamed classes either.
    free_device = sum(gpu["free_mib"] for gpu in host["gpus"]) * 1024 * 1024
    assert streamable > free_device
    for gpu in host["gpus"]:
        # nvidia-smi reports CUDA-usable free memory, which excludes reserved
        # driver overhead, so it is bounded by rather than equal to the
        # arithmetic difference.
        assert 0 <= gpu["free_mib"] <= gpu["total_mib"] - gpu["used_mib"]
        assert gpu["used_mib"] <= gpu["total_mib"]
