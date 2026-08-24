use crate::sweep::{SweepStatus, TrialResult};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use uuid::Uuid;

const SCHEMA_VERSION: u8 = 1;
const DEFAULT_COLLECTOR_URL: &str = "https://ese-benchmark-collector.x3ro.workers.dev";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommunityTrial {
    phase: String,
    context: u64,
    kv_type: String,
    batch_size: u32,
    slots: u32,
    stable: bool,
    tokens_per_second: Option<f64>,
    elapsed_seconds: f64,
    error_class: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CommunityBenchmark {
    schema_version: u8,
    event_id: String,
    installation_id: String,
    recorded_at_epoch: u64,
    ese_version: String,
    platform: String,
    cpu_model: Option<String>,
    logical_cpus: usize,
    ram_gib: Option<u64>,
    gpus: Vec<String>,
    model_signature: String,
    model_size_bytes: Option<u64>,
    architecture: Option<String>,
    quantization: Option<String>,
    preset: String,
    objective: String,
    safe_margin: f32,
    verified_max_context: u64,
    promoted_context: u64,
    best_kv_type: String,
    best_batch_size: u32,
    best_tokens_per_second: f64,
    trials: Vec<CommunityTrial>,
}

fn roots() -> Result<(PathBuf, PathBuf), String> {
    let dirs = ProjectDirs::from("io", "xero00000", "ese")
        .ok_or_else(|| "could not locate the user data directory".to_string())?;
    Ok((
        dirs.config_dir().join("telemetry-installation-id"),
        dirs.data_local_dir().join("telemetry/outbox"),
    ))
}

fn installation_id(path: &Path) -> Result<String, String> {
    if let Ok(value) = fs::read_to_string(path) {
        let value = value.trim();
        if Uuid::parse_str(value).is_ok() {
            return Ok(value.to_owned());
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let value = Uuid::new_v4().to_string();
    fs::write(path, &value).map_err(|error| error.to_string())?;
    Ok(value)
}

fn first_value(path: &Path, prefix: &str) -> Option<String> {
    fs::read_to_string(path).ok()?.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == prefix).then(|| value.trim().to_owned())
    })
}

fn ram_gib() -> Option<u64> {
    let kib = first_value(Path::new("/proc/meminfo"), "MemTotal")?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    Some(kib.div_ceil(1024 * 1024))
}

fn gpus() -> Vec<String> {
    let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect()
}

fn error_class(result: &TrialResult) -> Option<String> {
    let error = result.error.as_deref()?.to_ascii_lowercase();
    Some(
        if error.contains("memory") || error.contains("alloc") {
            "memory"
        } else if error.contains("timeout") || error.contains("ready") {
            "startup-timeout"
        } else if error.contains("cancel") {
            "cancelled"
        } else {
            "runtime"
        }
        .into(),
    )
}

fn model_metadata(path: &Path) -> (Option<u64>, Option<String>, Option<String>) {
    let size = fs::metadata(path).ok().map(|metadata| metadata.len());
    let metadata = crate::gguf::inspect(path).ok();
    let architecture = metadata
        .as_ref()
        .and_then(|value| value.architecture.clone());
    let upper = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_uppercase();
    let quantization = [
        "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K",
        "Q8_0", "F16",
    ]
    .into_iter()
    .find(|quant| upper.contains(quant))
    .map(str::to_owned);
    (size, architecture, quantization)
}

fn build_payload(
    status: &SweepStatus,
    installation_id: String,
) -> Result<CommunityBenchmark, String> {
    if status.state != "complete" {
        return Err("only completed verified sweeps can be shared".into());
    }
    let (model_size_bytes, architecture, quantization) = model_metadata(&status.request.model_path);
    let mut signature = Sha256::new();
    signature.update(architecture.as_deref().unwrap_or("unknown"));
    signature.update([0]);
    signature.update(quantization.as_deref().unwrap_or("unknown"));
    signature.update([0]);
    signature.update(model_size_bytes.unwrap_or_default().to_le_bytes());
    let model_signature = format!("{:x}", signature.finalize());
    let trials = status
        .results
        .iter()
        .map(|result| CommunityTrial {
            phase: result.phase.clone(),
            context: result.context,
            kv_type: result.kv_type.clone(),
            batch_size: result.batch_size,
            slots: result.slots,
            stable: result.stable,
            tokens_per_second: result.tokens_per_second,
            elapsed_seconds: result.elapsed_seconds,
            error_class: error_class(result),
        })
        .collect();
    Ok(CommunityBenchmark {
        schema_version: SCHEMA_VERSION,
        event_id: status.id.clone(),
        installation_id,
        recorded_at_epoch: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        ese_version: env!("CARGO_PKG_VERSION").into(),
        platform: std::env::consts::OS.into(),
        cpu_model: first_value(Path::new("/proc/cpuinfo"), "model name"),
        logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
        ram_gib: ram_gib(),
        gpus: gpus(),
        model_signature,
        model_size_bytes,
        architecture,
        quantization,
        preset: status.request.preset.clone(),
        objective: status.request.objective.clone(),
        safe_margin: status.request.safe_margin,
        verified_max_context: status
            .verified_max_context
            .ok_or_else(|| "verified sweep is missing maximum context".to_string())?,
        promoted_context: status
            .promoted_context
            .ok_or_else(|| "verified sweep is missing promoted context".to_string())?,
        best_kv_type: status
            .best_kv_type
            .clone()
            .ok_or_else(|| "verified sweep is missing KV result".to_string())?,
        best_batch_size: status
            .best_batch_size
            .ok_or_else(|| "verified sweep is missing batch result".to_string())?,
        best_tokens_per_second: status
            .best_tokens_per_second
            .ok_or_else(|| "verified sweep is missing speed result".to_string())?,
        trials,
    })
}

fn collector_url() -> Option<String> {
    std::env::var("ESE_BENCHMARK_COLLECTOR_URL")
        .ok()
        .filter(|value| value.starts_with("https://"))
        .or_else(|| Some(DEFAULT_COLLECTOR_URL.to_owned()))
}

fn upload_outbox(outbox: &Path) -> Result<(), String> {
    let Some(url) = collector_url() else {
        return Ok(());
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|error| error.to_string())?;
    let entries = fs::read_dir(outbox).map_err(|error| error.to_string())?;
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path).map_err(|error| error.to_string())?;
        let response = client
            .post(format!("{}/v1/benchmarks", url.trim_end_matches('/')))
            .header("content-type", "application/json")
            .body(bytes)
            .send()
            .map_err(|error| error.to_string())?;
        if response.status().is_success() || response.status().as_u16() == 409 {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        } else {
            return Err(format!(
                "collector rejected benchmark with {}",
                response.status()
            ));
        }
    }
    Ok(())
}

pub fn queue_and_upload(status: &SweepStatus) -> Result<(), String> {
    let (installation_path, outbox) = roots()?;
    fs::create_dir_all(&outbox).map_err(|error| error.to_string())?;
    let payload = build_payload(status, installation_id(&installation_path)?)?;
    let path = outbox.join(format!("{}.json", payload.event_id));
    if !path.exists() {
        let serialized = serde_json::to_vec_pretty(&payload).map_err(|error| error.to_string())?;
        fs::write(&path, serialized).map_err(|error| error.to_string())?;
    }
    upload_outbox(&outbox)
}

pub fn clear_outbox() -> Result<(), String> {
    let (_, outbox) = roots()?;
    if outbox.is_dir() {
        for entry in fs::read_dir(outbox).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path.is_file() {
                fs::remove_file(path).map_err(|error| error.to_string())?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_text_is_reduced_to_a_safe_class() {
        let result = TrialResult {
            phase: "capacity".into(),
            context: 4096,
            kv_type: "q8_0".into(),
            batch_size: 256,
            slots: 1,
            stable: false,
            tokens_per_second: None,
            elapsed_seconds: 1.0,
            error: Some("failed /home/alice/private.gguf CUDA allocation".into()),
            log_path: PathBuf::from("/home/alice/private.log"),
        };
        assert_eq!(error_class(&result).as_deref(), Some("memory"));
    }

    #[test]
    fn serialized_payload_omits_local_paths_and_raw_errors() {
        let private_path = PathBuf::from("/home/alice/secret/models/private.Q4_K_M.gguf");
        let status = SweepStatus {
            id: "verified-sweep".into(),
            state: "complete".into(),
            phase: "complete".into(),
            request: crate::sweep::SweepRequest {
                model_path: private_path.clone(),
                preset: "quick".into(),
                objective: "max-safe-context".into(),
                max_context: 32768,
                safe_margin: 0.9,
            },
            completed_trials: 1,
            total_trials: 1,
            current_label: "complete".into(),
            verified_max_context: Some(32768),
            promoted_context: Some(29440),
            best_kv_type: Some("q8_0".into()),
            best_batch_size: Some(256),
            best_slots: Some(1),
            best_tokens_per_second: Some(20.5),
            results: vec![TrialResult {
                phase: "tuning".into(),
                context: 29440,
                kv_type: "q8_0".into(),
                batch_size: 256,
                slots: 1,
                stable: true,
                tokens_per_second: Some(20.5),
                elapsed_seconds: 2.0,
                error: Some(format!("failed to read {}", private_path.display())),
                log_path: PathBuf::from("/home/alice/private.log"),
            }],
            error: None,
            checkpoint_path: PathBuf::from("/home/alice/checkpoint.json"),
        };
        let payload = build_payload(&status, "anonymous-installation".into())
            .expect("complete sweep should serialize");
        let json = serde_json::to_string(&payload).expect("payload should be JSON");
        assert!(!json.contains("alice"));
        assert!(!json.contains("private.log"));
        assert!(!json.contains("failed to read"));
        assert!(json.contains("runtime"));
    }
}
