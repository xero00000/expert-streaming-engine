use fs2::available_space;
use futures_util::StreamExt;
use reqwest::{header, Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HubModel {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,
    pub tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HubApiModel {
    id: String,
    #[serde(default)]
    sha: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    siblings: Vec<HubFile>,
}

#[derive(Debug, Clone, Deserialize)]
struct HubFile {
    rfilename: String,
    #[serde(default)]
    size: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareSummary {
    pub ram_available_bytes: u64,
    pub ram_total_bytes: u64,
    pub vram_free_bytes: u64,
    pub vram_total_bytes: u64,
    pub gpu_names: Vec<String>,
    pub resident_budget_bytes: u64,
    pub hybrid_budget_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuantVariant {
    pub id: String,
    pub quantization: String,
    pub files: Vec<String>,
    pub file_sizes: BTreeMap<String, u64>,
    pub total_size_bytes: u64,
    pub fit: String,
    pub fit_reason: String,
    pub recommended: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HubModelDetails {
    pub model: HubModel,
    pub revision: String,
    pub variants: Vec<QuantVariant>,
    pub hardware: HardwareSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadRequest {
    pub repo_id: String,
    pub revision: String,
    pub variant_id: String,
    pub quantization: String,
    pub files: Vec<String>,
    pub file_sizes: BTreeMap<String, u64>,
    pub total_bytes: u64,
    pub destination_root: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadStatus {
    pub id: String,
    pub state: String,
    pub repo_id: String,
    pub variant_id: String,
    pub current_file: Option<String>,
    pub completed_bytes: u64,
    pub total_bytes: u64,
    pub bytes_per_second: Option<u64>,
    pub eta_seconds: Option<u64>,
    pub destination: PathBuf,
    pub error: Option<String>,
}

#[derive(Clone, Default)]
pub struct DownloadManager {
    current: Arc<Mutex<Option<DownloadStatus>>>,
    cancel: Arc<AtomicBool>,
}

fn client() -> Result<Client, String> {
    Client::builder()
        .user_agent("ESE-Studio/0.1.0")
        .build()
        .map_err(|error| format!("could not create Hugging Face client: {error}"))
}

fn auth_token() -> Option<String> {
    env::var("HF_TOKEN")
        .ok()
        .or_else(|| env::var("HUGGING_FACE_HUB_TOKEN").ok())
        .filter(|token| !token.trim().is_empty())
}

fn authenticated(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some(token) = auth_token() {
        request.bearer_auth(token)
    } else {
        request
    }
}

pub async fn search_models(query: &str) -> Result<Vec<HubModel>, String> {
    let query = query.trim();
    if query.len() < 2 {
        return Err("enter at least two characters to search Hugging Face".into());
    }
    let response = authenticated(client()?.get("https://huggingface.co/api/models").query(&[
        ("search", query),
        ("filter", "gguf"),
        ("sort", "downloads"),
        ("direction", "-1"),
        ("limit", "30"),
        ("full", "true"),
    ]))
    .send()
    .await
    .map_err(|error| format!("Hugging Face search failed: {error}"))?;
    if response.status() == StatusCode::TOO_MANY_REQUESTS {
        return Err("Hugging Face rate limit reached; wait a moment and retry".into());
    }
    let models = response
        .error_for_status()
        .map_err(|error| format!("Hugging Face search failed: {error}"))?
        .json::<Vec<HubApiModel>>()
        .await
        .map_err(|error| format!("invalid Hugging Face search response: {error}"))?;
    Ok(models
        .into_iter()
        .map(|model| HubModel {
            id: model.id,
            downloads: model.downloads,
            likes: model.likes,
            last_modified: model.last_modified,
            tags: model.tags,
        })
        .collect())
}

fn quantization(filename: &str) -> Option<String> {
    let upper = filename.to_ascii_uppercase();
    let known = [
        "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0",
        "IQ4_XS", "IQ4_NL", "Q3_K_L", "Q3_K_M", "Q3_K_S", "Q2_K_L", "Q2_K", "IQ3_M", "IQ3_S",
        "IQ3_XS", "IQ3_XXS", "IQ2_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S",
    ];
    let base = known.into_iter().find(|name| upper.contains(name))?;
    let prefix = if upper.contains(&format!("UD-{base}")) {
        "UD-"
    } else {
        ""
    };
    let suffix = if upper.contains(&format!("{base}_XL")) {
        "_XL"
    } else {
        ""
    };
    Some(format!("{prefix}{base}{suffix}"))
}

fn quality_rank(quant: &str) -> u16 {
    let quant = quant.trim_start_matches("UD-").trim_end_matches("_XL");
    match quant {
        "BF16" | "F16" => 160,
        "Q8_0" => 140,
        "Q6_K" => 120,
        "Q5_K_M" => 110,
        "Q5_K_S" => 105,
        "Q4_K_M" => 100,
        "Q4_K_S" | "IQ4_XS" | "IQ4_NL" => 95,
        "Q4_1" | "Q4_0" => 90,
        "Q3_K_L" => 82,
        "Q3_K_M" | "IQ3_M" => 78,
        "Q3_K_S" | "IQ3_S" => 74,
        "Q2_K_L" => 68,
        "Q2_K" | "IQ2_M" => 64,
        "IQ2_S" | "IQ2_XS" => 58,
        "IQ2_XXS" => 54,
        "IQ1_M" | "IQ1_S" => 40,
        _ => 0,
    }
}

fn hardware(ese_binary: &Path) -> Result<HardwareSummary, String> {
    let output = Command::new(ese_binary)
        .args(["doctor", "--json"])
        .output()
        .map_err(|error| format!("could not inspect ESE hardware: {error}"))?;
    let doctor: Value = serde_json::from_slice(&output.stdout).map_err(|error| {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        if detail.is_empty() {
            format!(
                "invalid ESE hardware report ({status}): {error}",
                status = output.status
            )
        } else {
            format!("invalid ESE hardware report: {error}; {detail}")
        }
    })?;
    let ram_available = doctor
        .get("ram_available")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let ram_total = doctor
        .get("ram_total")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let gpus = if doctor
        .get("server_cuda")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        doctor
            .get("gpus")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let vram_total = gpus
        .iter()
        .filter_map(|gpu| gpu.get("total_bytes").and_then(Value::as_u64))
        .sum::<u64>();
    let vram_free = gpus
        .iter()
        .filter_map(|gpu| gpu.get("free_bytes").and_then(Value::as_u64))
        .sum::<u64>();
    let gpu_names = gpus
        .iter()
        .filter_map(|gpu| gpu.get("name").and_then(Value::as_str).map(str::to_owned))
        .collect::<Vec<_>>();
    let reserved = (gpu_names.len() as u64).saturating_mul(GIB);
    let usable_vram = vram_total.saturating_sub(reserved);
    let resident_budget = (usable_vram as f64 * 0.82) as u64;
    let hybrid_budget = resident_budget.saturating_add((ram_available as f64 * 0.65) as u64);
    Ok(HardwareSummary {
        ram_available_bytes: ram_available,
        ram_total_bytes: ram_total,
        vram_free_bytes: vram_free,
        vram_total_bytes: vram_total,
        gpu_names,
        resident_budget_bytes: resident_budget,
        hybrid_budget_bytes: hybrid_budget,
    })
}

pub async fn model_details(repo_id: &str, ese_binary: &Path) -> Result<HubModelDetails, String> {
    if repo_id.split('/').count() != 2 || repo_id.contains("..") {
        return Err("invalid Hugging Face repository id".into());
    }
    let url = format!("https://huggingface.co/api/models/{repo_id}");
    let response = authenticated(client()?.get(url).query(&[("blobs", "true")]))
        .send()
        .await
        .map_err(|error| format!("could not load repository details: {error}"))?;
    let model = response
        .error_for_status()
        .map_err(|error| format!("could not load repository details: {error}"))?
        .json::<HubApiModel>()
        .await
        .map_err(|error| format!("invalid repository response: {error}"))?;
    let hardware = hardware(ese_binary)?;
    let mut grouped: BTreeMap<String, Vec<HubFile>> = BTreeMap::new();
    for file in model.siblings.iter().filter(|file| {
        let lower = file.rfilename.to_ascii_lowercase();
        lower.ends_with(".gguf") && !lower.contains("mmproj") && !lower.contains("imatrix")
    }) {
        if let Some(quant) = quantization(&file.rfilename) {
            grouped.entry(quant).or_default().push(file.clone());
        }
    }
    let mut variants = grouped
        .into_iter()
        .map(|(quantization, mut files)| {
            files.sort_by(|left, right| left.rfilename.cmp(&right.rfilename));
            let total_size = files.iter().map(|file| file.size).sum::<u64>();
            let (fit, reason) = if total_size <= hardware.resident_budget_bytes {
                (
                    "Resident",
                    "Fits the safe multi-GPU weight budget with context headroom",
                )
            } else if total_size <= hardware.hybrid_budget_bytes {
                (
                    "Hybrid/cache",
                    "Fits combined safe VRAM and currently available RAM",
                )
            } else {
                (
                    "Streamed",
                    "Requires ESE expert streaming or additional free memory",
                )
            };
            let file_sizes = files
                .iter()
                .map(|file| (file.rfilename.clone(), file.size))
                .collect();
            QuantVariant {
                id: format!("{}-{}", quantization.to_ascii_lowercase(), total_size),
                quantization,
                files: files.into_iter().map(|file| file.rfilename).collect(),
                file_sizes,
                total_size_bytes: total_size,
                fit: fit.into(),
                fit_reason: reason.into(),
                recommended: false,
            }
        })
        .collect::<Vec<_>>();
    variants.sort_by(|left, right| {
        quality_rank(&right.quantization)
            .cmp(&quality_rank(&left.quantization))
            .then(left.total_size_bytes.cmp(&right.total_size_bytes))
    });
    let recommendation = variants
        .iter()
        .position(|variant| variant.total_size_bytes <= hardware.resident_budget_bytes)
        .or_else(|| {
            variants
                .iter()
                .position(|variant| variant.total_size_bytes <= hardware.hybrid_budget_bytes)
        })
        .or_else(|| variants.len().checked_sub(1));
    if let Some(index) = recommendation {
        variants[index].recommended = true;
    }
    if variants.is_empty() {
        return Err("this repository has no recognized downloadable GGUF quantizations".into());
    }
    Ok(HubModelDetails {
        model: HubModel {
            id: model.id,
            downloads: model.downloads,
            likes: model.likes,
            last_modified: model.last_modified,
            tags: model.tags,
        },
        revision: model.sha.unwrap_or_else(|| "main".into()),
        variants,
        hardware,
    })
}

fn safe_relative(path: &str) -> bool {
    let path = Path::new(path);
    !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn download_url(repo_id: &str, revision: &str, filename: &str) -> Result<Url, String> {
    let mut url = Url::parse("https://huggingface.co").map_err(|error| error.to_string())?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "could not construct Hugging Face URL".to_string())?;
        for segment in repo_id.split('/') {
            segments.push(segment);
        }
        segments.push("resolve").push(revision);
        for segment in filename.split('/') {
            segments.push(segment);
        }
    }
    Ok(url)
}

impl DownloadManager {
    pub fn current(&self) -> Result<Option<DownloadStatus>, String> {
        self.current
            .lock()
            .map(|status| status.clone())
            .map_err(|_| "download state is poisoned".into())
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn start(
        &self,
        request: DownloadRequest,
        app: AppHandle,
    ) -> Result<DownloadStatus, String> {
        if self
            .current()?
            .is_some_and(|status| status.state == "downloading")
        {
            return Err("another model download is already running".into());
        }
        if request.files.is_empty()
            || request.files.iter().any(|file| !safe_relative(file))
            || request
                .files
                .iter()
                .any(|file| !request.file_sizes.contains_key(file))
        {
            return Err("download contains an unsafe or empty file list".into());
        }
        let repo_folder = request.repo_id.replace('/', "--");
        let destination = request.destination_root.join(repo_folder);
        fs::create_dir_all(&destination)
            .map_err(|error| format!("could not create {}: {error}", destination.display()))?;
        let destination = destination
            .canonicalize()
            .map_err(|error| format!("could not resolve download folder: {error}"))?;
        let free = available_space(&destination)
            .map_err(|error| format!("could not check free disk space: {error}"))?;
        let required = request.total_bytes.saturating_add(request.total_bytes / 20);
        if free < required {
            return Err(format!(
                "not enough free disk space: need {:.1} GiB including safety margin, have {:.1} GiB",
                required as f64 / GIB as f64,
                free as f64 / GIB as f64
            ));
        }
        self.cancel.store(false, Ordering::SeqCst);
        let status = DownloadStatus {
            id: Uuid::new_v4().to_string(),
            state: "downloading".into(),
            repo_id: request.repo_id.clone(),
            variant_id: request.variant_id.clone(),
            current_file: None,
            completed_bytes: 0,
            total_bytes: request.total_bytes,
            bytes_per_second: None,
            eta_seconds: None,
            destination: destination.clone(),
            error: None,
        };
        *self
            .current
            .lock()
            .map_err(|_| "download state is poisoned")? = Some(status.clone());
        let manager = self.clone();
        tauri::async_runtime::spawn(async move {
            let result = manager.run_download(&request, &destination, &app).await;
            let mut current = match manager.current.lock() {
                Ok(current) => current,
                Err(_) => return,
            };
            if let Some(status) = current.as_mut() {
                match result {
                    Ok(()) => status.state = "complete".into(),
                    Err(error) if manager.cancel.load(Ordering::SeqCst) => {
                        status.state = "cancelled".into();
                        status.error = Some(error);
                    }
                    Err(error) => {
                        status.state = "failed".into();
                        status.error = Some(error);
                    }
                }
                let _ = app.emit("download-progress", status.clone());
            }
        });
        Ok(status)
    }

    async fn run_download(
        &self,
        request: &DownloadRequest,
        destination: &Path,
        app: &AppHandle,
    ) -> Result<(), String> {
        let client = client()?;
        let mut completed_before_file = 0_u64;
        let transfer_started = Instant::now();
        let mut transferred_this_run = 0_u64;
        let mut last_emit = Instant::now() - Duration::from_secs(1);
        for filename in &request.files {
            if self.cancel.load(Ordering::SeqCst) {
                return Err("download cancelled; partial files were kept for resume".into());
            }
            let final_path = destination.join(filename);
            let expected_size = request
                .file_sizes
                .get(filename)
                .copied()
                .unwrap_or_default();
            let parent = final_path
                .parent()
                .ok_or_else(|| "download file has no parent folder".to_string())?;
            fs::create_dir_all(parent)
                .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
            let parent = parent
                .canonicalize()
                .map_err(|error| format!("could not resolve download subfolder: {error}"))?;
            if !parent.starts_with(destination) {
                return Err("download path escaped the selected model folder".into());
            }
            if fs::symlink_metadata(&final_path).is_ok_and(|meta| meta.file_type().is_symlink()) {
                return Err("refusing to overwrite a symbolic link".into());
            }
            let partial = final_path.with_extension(format!(
                "{}.part",
                final_path
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .unwrap_or("download")
            ));
            if let Ok(metadata) = fs::metadata(&final_path) {
                if metadata.len() == expected_size {
                    if partial.is_file() {
                        fs::remove_file(&partial).map_err(|error| {
                            format!("could not remove stale {}: {error}", partial.display())
                        })?;
                    }
                    completed_before_file = completed_before_file.saturating_add(expected_size);
                    let mut status = self
                        .current
                        .lock()
                        .map_err(|_| "download state is poisoned")?;
                    if let Some(status) = status.as_mut() {
                        status.current_file = Some(filename.clone());
                        status.completed_bytes = completed_before_file;
                        let _ = app.emit("download-progress", status.clone());
                    }
                    continue;
                }
                return Err(format!(
                    "{} already exists but its size does not match the repository; move it aside before retrying",
                    final_path.display()
                ));
            }
            let mut offset = fs::metadata(&partial)
                .map(|meta| meta.len())
                .unwrap_or_default();
            if offset > expected_size {
                offset = 0;
            }
            let url = download_url(&request.repo_id, &request.revision, filename)?;
            let mut download = authenticated(client.get(url));
            if offset > 0 {
                download = download.header(header::RANGE, format!("bytes={offset}-"));
            }
            let response = download
                .send()
                .await
                .map_err(|error| format!("download failed for {filename}: {error}"))?;
            if response.status() == StatusCode::UNAUTHORIZED
                || response.status() == StatusCode::FORBIDDEN
            {
                return Err("repository is gated or private; set HF_TOKEN and retry".into());
            }
            if offset > 0 && response.status() != StatusCode::PARTIAL_CONTENT {
                offset = 0;
            }
            let response = response
                .error_for_status()
                .map_err(|error| format!("download failed for {filename}: {error}"))?;
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(offset > 0)
                .truncate(offset == 0)
                .open(&partial)
                .map_err(|error| format!("could not write {}: {error}", partial.display()))?;
            {
                let mut status = self
                    .current
                    .lock()
                    .map_err(|_| "download state is poisoned")?;
                if let Some(status) = status.as_mut() {
                    status.current_file = Some(filename.clone());
                    status.completed_bytes = completed_before_file.saturating_add(offset);
                    let _ = app.emit("download-progress", status.clone());
                }
            }
            let mut downloaded = offset;
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                if self.cancel.load(Ordering::SeqCst) {
                    return Err("download cancelled; partial files were kept for resume".into());
                }
                let chunk = chunk.map_err(|error| format!("download interrupted: {error}"))?;
                file.write_all(&chunk)
                    .map_err(|error| format!("could not write download: {error}"))?;
                downloaded = downloaded.saturating_add(chunk.len() as u64);
                transferred_this_run = transferred_this_run.saturating_add(chunk.len() as u64);
                let mut status = self
                    .current
                    .lock()
                    .map_err(|_| "download state is poisoned")?;
                if let Some(status) = status.as_mut() {
                    status.completed_bytes = completed_before_file.saturating_add(downloaded);
                    let elapsed = transfer_started.elapsed().as_secs_f64();
                    if elapsed > 0.0 {
                        let speed = (transferred_this_run as f64 / elapsed) as u64;
                        status.bytes_per_second = Some(speed);
                        status.eta_seconds = (speed > 0).then(|| {
                            status.total_bytes.saturating_sub(status.completed_bytes) / speed
                        });
                    }
                    if last_emit.elapsed() >= Duration::from_millis(120) {
                        let _ = app.emit("download-progress", status.clone());
                        last_emit = Instant::now();
                    }
                }
            }
            file.flush()
                .map_err(|error| format!("could not flush download: {error}"))?;
            if downloaded != expected_size {
                return Err(format!(
                    "downloaded size for {filename} was {downloaded} bytes; expected {expected_size}"
                ));
            }
            fs::rename(&partial, &final_path)
                .map_err(|error| format!("could not finalize {}: {error}", final_path.display()))?;
            completed_before_file = completed_before_file.saturating_add(downloaded);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_common_and_unsloth_dynamic_quants() {
        assert_eq!(quantization("model-Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(
            quantization("model-UD-Q6_K_XL.gguf").as_deref(),
            Some("UD-Q6_K_XL")
        );
        assert_eq!(
            quantization("mmproj-model-F16.gguf").as_deref(),
            Some("F16")
        );
    }

    #[test]
    fn rejects_unsafe_download_paths() {
        assert!(safe_relative("Q4/model-00001-of-00002.gguf"));
        assert!(!safe_relative("../model.gguf"));
        assert!(!safe_relative("/tmp/model.gguf"));
    }

    #[test]
    fn quality_order_prefers_lossless_then_balanced_quants() {
        assert!(quality_rank("F16") > quality_rank("Q8_0"));
        assert!(quality_rank("Q5_K_M") > quality_rank("Q4_K_M"));
        assert!(quality_rank("Q4_K_M") > quality_rank("Q2_K"));
    }
}
