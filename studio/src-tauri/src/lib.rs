use directories::{BaseDirs, ProjectDirs};
use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Duration,
};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use walkdir::WalkDir;

mod gguf;
mod hub;
mod sweep;
mod telemetry;
use hub::{DownloadManager, DownloadRequest, DownloadStatus, HubModel, HubModelDetails};
use sweep::{SweepManager, SweepPlan, SweepRequest, SweepStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelProfile {
    id: String,
    name: String,
    path: PathBuf,
    #[serde(alias = "size_bytes")]
    size_bytes: Option<u64>,
    available: bool,
    architecture: Option<String>,
    quantization: Option<String>,
    context: Option<u64>,
    #[serde(default, alias = "kv_type")]
    kv_type: Option<String>,
    #[serde(default, alias = "batch_size")]
    batch_size: Option<u32>,
    #[serde(default, alias = "ubatch_size")]
    ubatch_size: Option<u32>,
    source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AppProfile {
    id: String,
    name: String,
    command: String,
    #[serde(default)]
    args: Vec<String>,
    #[serde(default, alias = "working_directory")]
    working_directory: Option<PathBuf>,
    #[serde(default, alias = "endpoint_aware")]
    endpoint_aware: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StudioConfig {
    version: u8,
    #[serde(default, alias = "onboarding_complete")]
    onboarding_complete: bool,
    #[serde(default, alias = "help_improve_ese")]
    help_improve_ese: bool,
    #[serde(default, alias = "model_roots")]
    model_roots: Vec<PathBuf>,
    #[serde(default)]
    models: Vec<ModelProfile>,
    #[serde(default)]
    apps: Vec<AppProfile>,
    #[serde(default = "default_safe_margin", alias = "safe_context_margin")]
    safe_context_margin: f32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StudioSnapshot {
    platform: String,
    config_path: PathBuf,
    onboarding_complete: bool,
    help_improve_ese: bool,
    model_roots: Vec<PathBuf>,
    models: Vec<ModelProfile>,
    apps: Vec<AppProfile>,
    ese_binary: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatStatus {
    active: bool,
    model_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatChunk {
    request_id: String,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatFinished {
    request_id: String,
    stopped: bool,
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TerminalOutput {
    session_id: String,
    data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelEndpoint {
    base_url: String,
    api_key: String,
    model_id: String,
    model_path: PathBuf,
    context: u64,
    architecture: Option<String>,
    quantization: Option<String>,
    kv_type: Option<String>,
    batch_size: Option<u32>,
    ubatch_size: Option<u32>,
    #[serde(default)]
    resource_plan: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LaunchRequest {
    command: String,
    #[serde(default)]
    args: Vec<String>,
    working_directory: Option<PathBuf>,
    columns: Option<u16>,
    rows: Option<u16>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    app_id: Option<String>,
    #[serde(default)]
    endpoint_aware: bool,
    #[serde(default)]
    model_endpoint: Option<ModelEndpoint>,
}

struct TerminalSession {
    master: Box<dyn MasterPty + Send>,
    child: Box<dyn Child + Send + Sync>,
    writer: Box<dyn Write + Send>,
}

#[derive(Clone)]
struct ManagedModel {
    session_id: String,
    request: LaunchRequest,
    endpoint: ModelEndpoint,
}

#[derive(Default)]
struct StudioState {
    terminals: Mutex<HashMap<String, TerminalSession>>,
    active_model: Mutex<Option<ManagedModel>>,
    sweep: SweepManager,
    downloads: DownloadManager,
    chat_cancellations: Mutex<HashMap<String, Arc<AtomicBool>>>,
}

fn default_safe_margin() -> f32 {
    0.9
}

fn project_dirs() -> Result<ProjectDirs, String> {
    ProjectDirs::from("io", "xero00000", "ese")
        .ok_or_else(|| "could not locate the user config directory".into())
}

fn config_path() -> Result<PathBuf, String> {
    Ok(project_dirs()?.config_dir().join("studio.toml"))
}

fn home_dir() -> Option<PathBuf> {
    BaseDirs::new().map(|directories| directories.home_dir().to_path_buf())
}

fn executable_names(command: &str) -> Vec<String> {
    #[cfg(target_os = "windows")]
    {
        if Path::new(command).extension().is_none() {
            return ["exe", "cmd", "bat", "com"]
                .into_iter()
                .map(|extension| format!("{command}.{extension}"))
                .chain(std::iter::once(command.to_owned()))
                .collect();
        }
    }
    vec![command.to_owned()]
}

fn executable_candidates(directory: &Path, command: &str) -> Vec<PathBuf> {
    executable_names(command)
        .into_iter()
        .map(|name| directory.join(name))
        .collect()
}

fn resolve_executable(command: &str) -> Option<PathBuf> {
    let command_path = Path::new(command);
    if command_path.is_absolute() || command_path.components().count() > 1 {
        return Path::new(command).is_file().then(|| PathBuf::from(command));
    }

    let mut candidates = Vec::new();
    if let Some(paths) = std::env::var_os("PATH") {
        for path in std::env::split_paths(&paths) {
            candidates.extend(executable_candidates(&path, command));
        }
    }
    if let Some(home) = home_dir() {
        for directory in [
            home.join(".local/bin"),
            home.join(".cargo/bin"),
            home.join(".bun/bin"),
            home.join(".npm-global/bin"),
            home.join("scoop/shims"),
        ] {
            candidates.extend(executable_candidates(&directory, command));
        }

        #[cfg(target_os = "windows")]
        if let Some(app_data) = std::env::var_os("APPDATA") {
            candidates.extend(executable_candidates(
                &PathBuf::from(app_data).join("npm"),
                command,
            ));
        }

        let node_versions = home.join(".nvm/versions/node");
        if let Ok(entries) = fs::read_dir(node_versions) {
            let mut versions = entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>();
            versions.sort_by_key(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or_default()
                    .trim_start_matches('v')
                    .split('.')
                    .take(3)
                    .map(|part| part.parse::<u64>().unwrap_or_default())
                    .collect::<Vec<_>>()
            });
            candidates.extend(
                versions
                    .into_iter()
                    .rev()
                    .map(|version| version.join("bin").join(command)),
            );
        }
    }

    candidates
        .into_iter()
        .find(|candidate| candidate.is_file())
        .and_then(|candidate| candidate.canonicalize().ok().or(Some(candidate)))
}

fn detected_apps() -> Vec<AppProfile> {
    [
        ("codex", "Codex", "codex", &[][..]),
        ("claude", "Claude Code", "claude", &[][..]),
        ("opencode", "OpenCode", "opencode", &[][..]),
        ("hermes", "Hermes", "hermes", &["chat"][..]),
        ("gemini", "Gemini CLI", "gemini", &[][..]),
        ("aider", "Aider", "aider", &[][..]),
        ("goose", "Goose", "goose", &[][..]),
        ("amp", "Amp", "amp", &[][..]),
        ("crush", "Crush", "crush", &[][..]),
        ("qwen-code", "Qwen Code", "qwen", &[][..]),
    ]
    .into_iter()
    .filter_map(|(id, name, command, args)| {
        resolve_executable(command).map(|executable| AppProfile {
            id: id.into(),
            name: name.into(),
            command: executable.to_string_lossy().into_owned(),
            args: args.iter().map(|argument| (*argument).into()).collect(),
            working_directory: None,
            endpoint_aware: true,
        })
    })
    .collect()
}

fn merge_detected_apps(configured: &mut Vec<AppProfile>, detected: Vec<AppProfile>) -> bool {
    let mut changed = false;
    for profile in detected {
        if configured.iter().all(|app| app.id != profile.id) {
            configured.push(profile);
            changed = true;
        }
    }
    changed
}

fn default_model_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(home) = home_dir() {
        let mut candidates = vec![home.join("models")];
        #[cfg(target_os = "windows")]
        candidates.push(home.join("Documents/ESE/models"));
        #[cfg(not(target_os = "windows"))]
        candidates.push(home.join(".local/share/ese/models"));
        for path in candidates {
            if path.is_dir() {
                roots.push(path);
            }
        }
    }
    roots
}

fn load_config() -> Result<StudioConfig, String> {
    let path = config_path()?;
    if path.is_file() {
        let raw = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let mut config: StudioConfig = toml::from_str(&raw)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        if merge_detected_apps(&mut config.apps, detected_apps()) {
            write_config(&config)?;
        }
        return Ok(config);
    }
    Ok(StudioConfig {
        version: 1,
        onboarding_complete: false,
        help_improve_ese: false,
        model_roots: default_model_roots(),
        models: Vec::new(),
        apps: detected_apps(),
        safe_context_margin: default_safe_margin(),
    })
}

fn write_config(config: &StudioConfig) -> Result<(), String> {
    let path = config_path()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let serialized = toml::to_string_pretty(config).map_err(|error| error.to_string())?;
    let temporary = path.with_extension(format!("toml.{}.tmp", std::process::id()));
    fs::write(&temporary, serialized)
        .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
    replace_file(&temporary, &path)
}

fn replace_file(temporary: &Path, destination: &Path) -> Result<(), String> {
    if !destination.exists() {
        return fs::rename(temporary, destination)
            .map_err(|error| format!("failed to create {}: {error}", destination.display()));
    }
    let backup = destination.with_extension(format!("toml.{}.bak", std::process::id()));
    fs::rename(destination, &backup)
        .map_err(|error| format!("failed to stage {}: {error}", destination.display()))?;
    if let Err(error) = fs::rename(temporary, destination) {
        let _ = fs::rename(&backup, destination);
        return Err(format!(
            "failed to replace {}: {error}",
            destination.display()
        ));
    }
    fs::remove_file(&backup)
        .map_err(|error| format!("failed to remove {}: {error}", backup.display()))
}

fn model_name(path: &Path) -> String {
    path.file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("Unnamed model")
        .replace(['_', '.'], " ")
}

fn model_id(path: &Path) -> String {
    path.to_string_lossy()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .into()
}

fn infer_quantization(path: &Path) -> Option<String> {
    let upper = path.file_name()?.to_string_lossy().to_uppercase();
    [
        "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K",
        "Q8_0", "F16",
    ]
    .into_iter()
    .find(|quant| upper.contains(quant))
    .map(str::to_owned)
}

fn is_launchable_gguf(path: &Path) -> bool {
    if !path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
    {
        return false;
    }
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    !name.starts_with("mmproj-") && !name.contains(".imatrix.")
}

fn discover_models(config: &StudioConfig) -> Vec<ModelProfile> {
    let mut by_path: HashMap<PathBuf, ModelProfile> = config
        .models
        .iter()
        .cloned()
        .map(|mut model| {
            model.available = model.path.is_file();
            model.size_bytes = fs::metadata(&model.path)
                .ok()
                .map(|metadata| metadata.len());
            if model.available && (model.architecture.is_none() || model.context.is_none()) {
                if let Ok(metadata) = gguf::inspect(&model.path) {
                    model.architecture = model.architecture.or(metadata.architecture);
                    model.context = model.context.or(metadata.context_length);
                }
            }
            (
                model
                    .path
                    .canonicalize()
                    .unwrap_or_else(|_| model.path.clone()),
                model,
            )
        })
        .collect();

    for root in &config.model_roots {
        for entry in WalkDir::new(root)
            .max_depth(5)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if entry.file_type().is_file() && is_launchable_gguf(path) {
                let path = path.to_path_buf();
                let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                let metadata = gguf::inspect(&path).unwrap_or_default();
                by_path.entry(canonical).or_insert_with(|| ModelProfile {
                    id: model_id(&path),
                    name: model_name(&path),
                    path: path.clone(),
                    size_bytes: entry.metadata().ok().map(|metadata| metadata.len()),
                    available: true,
                    architecture: metadata.architecture,
                    quantization: infer_quantization(&path),
                    context: metadata.context_length,
                    kv_type: None,
                    batch_size: None,
                    ubatch_size: None,
                    source: "discovered".into(),
                });
            }
        }
    }
    let mut models: Vec<_> = by_path.into_values().collect();
    models.sort_by(|left, right| {
        right
            .available
            .cmp(&left.available)
            .then_with(|| left.name.cmp(&right.name))
    });
    models
}

fn locate_ese(app: Option<&AppHandle>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(explicit) = std::env::var_os("ESE_BIN") {
        candidates.push(PathBuf::from(explicit));
    }
    if let Some(resource_dir) = app.and_then(|handle| handle.path().resource_dir().ok()) {
        for name in executable_names("ese") {
            candidates.push(resource_dir.join("ese").join(name));
        }
    }
    if let Ok(executable) = std::env::current_exe() {
        for name in executable_names("ese") {
            candidates.push(executable.with_file_name(name));
        }
        for ancestor in executable.ancestors() {
            candidates.extend(executable_candidates(ancestor, "ese"));
        }
    }
    if let Some(home) = home_dir() {
        candidates.extend(executable_candidates(&home.join(".local/bin"), "ese"));
        candidates.extend(executable_candidates(&home.join("bin"), "ese"));
    }
    if let Some(resolved) = resolve_executable("ese") {
        candidates.push(resolved);
    }
    candidates.into_iter().find(|candidate| candidate.is_file())
}

#[tauri::command]
fn get_studio_snapshot(app: AppHandle) -> Result<StudioSnapshot, String> {
    let config = load_config()?;
    Ok(StudioSnapshot {
        platform: std::env::consts::OS.into(),
        config_path: config_path()?,
        onboarding_complete: config.onboarding_complete,
        help_improve_ese: config.help_improve_ese,
        model_roots: config.model_roots.clone(),
        models: discover_models(&config),
        apps: config.apps,
        ese_binary: locate_ese(Some(&app)),
    })
}

#[tauri::command]
async fn search_hf_models(query: String) -> Result<Vec<HubModel>, String> {
    hub::search_models(&query).await
}

#[tauri::command]
async fn get_hf_model_details(repo_id: String, app: AppHandle) -> Result<HubModelDetails, String> {
    let ese_binary =
        locate_ese(Some(&app)).ok_or_else(|| "ESE launcher was not found".to_string())?;
    hub::model_details(&repo_id, &ese_binary).await
}

#[tauri::command]
fn start_hf_download(
    request: DownloadRequest,
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<DownloadStatus, String> {
    let config = load_config()?;
    let destination = request
        .destination_root
        .canonicalize()
        .map_err(|error| format!("cannot use {}: {error}", request.destination_root.display()))?;
    let allowed = config.model_roots.iter().any(|root| {
        root.canonicalize()
            .is_ok_and(|configured| configured == destination)
    });
    if !allowed {
        return Err("downloads must use a configured model folder".into());
    }
    let mut request = request;
    request.destination_root = destination;
    state.downloads.start(request, app)
}

#[tauri::command]
fn get_hf_download_status(state: State<'_, StudioState>) -> Result<Option<DownloadStatus>, String> {
    state.downloads.current()
}

#[tauri::command]
fn cancel_hf_download(state: State<'_, StudioState>) {
    state.downloads.cancel();
}

#[tauri::command]
fn save_config(config: StudioConfig) -> Result<(), String> {
    write_config(&config)
}

#[tauri::command]
fn set_help_improve_ese(enabled: bool, app: AppHandle) -> Result<StudioSnapshot, String> {
    let mut config = load_config()?;
    config.onboarding_complete = true;
    config.help_improve_ese = enabled;
    write_config(&config)?;
    if !enabled {
        telemetry::clear_outbox()?;
    }
    get_studio_snapshot(app)
}

#[tauri::command]
fn get_chat_status(state: State<'_, StudioState>) -> Result<ChatStatus, String> {
    let active = state
        .active_model
        .lock()
        .map_err(|_| "model state is poisoned")?;
    Ok(ChatStatus {
        active: active.is_some(),
        model_id: active.as_ref().map(|model| model.endpoint.model_id.clone()),
    })
}

#[tauri::command]
fn start_chat(
    request_id: String,
    messages: Vec<ChatMessage>,
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<(), String> {
    if request_id.trim().is_empty() {
        return Err("chat request id is required".into());
    }
    if messages.is_empty() || messages.len() > 200 {
        return Err("chat history must contain between 1 and 200 messages".into());
    }
    if messages.iter().any(|message| {
        !matches!(message.role.as_str(), "system" | "user" | "assistant")
            || message.content.len() > 1_000_000
    }) {
        return Err("chat history contains an unsupported role or oversized message".into());
    }
    let endpoint = {
        let mut active = state
            .active_model
            .lock()
            .map_err(|_| "model state is poisoned")?;
        let model = active
            .as_mut()
            .ok_or_else(|| "start a model before opening a chat".to_string())?;
        refresh_model_endpoint(&mut model.endpoint);
        model.endpoint.clone()
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    state
        .chat_cancellations
        .lock()
        .map_err(|_| "chat state is poisoned")?
        .insert(request_id.clone(), cancelled.clone());

    thread::spawn(move || {
        let result = stream_chat(&request_id, &messages, &endpoint, &cancelled, &app);
        let managed = app.state::<StudioState>();
        if let Ok(mut cancellations) = managed.chat_cancellations.lock() {
            cancellations.remove(&request_id);
        }
        let stopped = cancelled.load(Ordering::Relaxed);
        let _ = app.emit(
            "chat-finished",
            ChatFinished {
                request_id,
                stopped,
                error: result.err(),
            },
        );
    });
    Ok(())
}

fn stream_chat(
    request_id: &str,
    messages: &[ChatMessage],
    endpoint: &ModelEndpoint,
    cancelled: &AtomicBool,
    app: &AppHandle,
) -> Result<(), String> {
    let url = format!(
        "{}/chat/completions",
        endpoint.base_url.trim_end_matches('/')
    );
    let response = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(3600))
        .build()
        .map_err(|error| error.to_string())?
        .post(url)
        .bearer_auth(&endpoint.api_key)
        .json(&serde_json::json!({
            "model": endpoint.model_id,
            "messages": messages,
            "stream": true,
            "temperature": 0.7
        }))
        .send()
        .map_err(|error| format!("local model connection failed: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().unwrap_or_default();
        return Err(format!(
            "local model returned {status}: {}",
            detail.chars().take(300).collect::<String>()
        ));
    }
    let reader = std::io::BufReader::new(response);
    for line in std::io::BufRead::lines(reader) {
        if cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }
        let line = line.map_err(|error| format!("chat stream interrupted: {error}"))?;
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            return Ok(());
        }
        if let Some(content) = chat_delta_content(data) {
            let _ = app.emit(
                "chat-token",
                ChatChunk {
                    request_id: request_id.to_owned(),
                    content,
                },
            );
        }
    }
    Ok(())
}

fn chat_delta_content(data: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(data)
        .ok()?
        .pointer("/choices/0/delta/content")?
        .as_str()
        .filter(|content| !content.is_empty())
        .map(str::to_owned)
}

#[tauri::command]
fn cancel_chat(request_id: String, state: State<'_, StudioState>) -> Result<(), String> {
    if let Some(cancelled) = state
        .chat_cancellations
        .lock()
        .map_err(|_| "chat state is poisoned")?
        .get(&request_id)
    {
        cancelled.store(true, Ordering::Relaxed);
    }
    Ok(())
}

#[tauri::command]
fn add_model_root(path: PathBuf) -> Result<(), String> {
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("cannot use {}: {error}", path.display()))?;
    if !canonical.is_dir() {
        return Err(format!(
            "model root is not a directory: {}",
            canonical.display()
        ));
    }
    let mut config = load_config()?;
    if !config.model_roots.contains(&canonical) {
        config.model_roots.push(canonical);
    }
    write_config(&config)
}

#[tauri::command]
fn add_app_profile(profile: AppProfile) -> Result<(), String> {
    if profile.name.trim().is_empty() || profile.command.trim().is_empty() {
        return Err("app name and command are required".into());
    }
    let mut config = load_config()?;
    if let Some(existing) = config.apps.iter_mut().find(|app| app.id == profile.id) {
        *existing = profile;
    } else {
        config.apps.push(profile);
    }
    write_config(&config)
}

#[tauri::command]
fn launch_terminal(
    request: LaunchRequest,
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<String, String> {
    if request.role.as_deref() == Some("model")
        && state.sweep.current()?.is_some_and(|status| {
            matches!(
                status.state.as_str(),
                "preparing" | "running" | "cancelling"
            )
        })
    {
        return Err("a model cannot be started while a sweep owns the GPUs".into());
    }
    spawn_terminal(request, app, state.inner())
}

fn configure_hermes(command: &str, endpoint: &ModelEndpoint) -> Result<(), String> {
    let context = endpoint.context.to_string();
    let updates = [
        ("model.default", endpoint.model_id.as_str()),
        ("model.provider", "llamacpp"),
        ("model.base_url", endpoint.base_url.as_str()),
        ("model.api_key", endpoint.api_key.as_str()),
        ("model.context_length", context.as_str()),
    ];
    for (key, value) in updates {
        let status = process_command(command, &["config", "set", key, value])
            .status()
            .map_err(|error| format!("failed to configure Hermes field {key}: {error}"))?;
        if !status.success() {
            return Err(format!("Hermes rejected model configuration field {key}"));
        }
    }
    Ok(())
}

fn process_command(command: &str, arguments: &[&str]) -> ProcessCommand {
    #[cfg(target_os = "windows")]
    if Path::new(command).extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
    }) {
        let mut process = ProcessCommand::new("cmd.exe");
        process.args(["/d", "/s", "/c"]);
        process.arg(windows_command_line(command, arguments.iter().copied()));
        return process;
    }
    let mut process = ProcessCommand::new(command);
    process.args(arguments);
    process
}

#[cfg(target_os = "windows")]
fn windows_command_line<'a>(command: &'a str, arguments: impl Iterator<Item = &'a str>) -> String {
    std::iter::once(command)
        .chain(arguments)
        .map(|value| format!("\"{}\"", value.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

fn server_props(base_url: &str) -> Option<serde_json::Value> {
    let authority = base_url
        .strip_prefix("http://")?
        .split('/')
        .next()
        .filter(|value| !value.is_empty())?;
    let address = authority.to_socket_addrs().ok()?.next()?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(2)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(2))).ok()?;
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .ok()?;
    stream
        .write_all(
            format!("GET /props HTTP/1.1\r\nHost: {authority}\r\nConnection: close\r\n\r\n")
                .as_bytes(),
        )
        .ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    let (_, body) = response.split_once("\r\n\r\n")?;
    serde_json::from_str(body).ok()
}

#[cfg(target_os = "linux")]
fn argument_value(arguments: &[String], flags: &[&str]) -> Option<String> {
    arguments
        .windows(2)
        .find_map(|pair| flags.contains(&pair[0].as_str()).then(|| pair[1].clone()))
}

fn refresh_model_endpoint(endpoint: &mut ModelEndpoint) {
    if let Some(props) = server_props(&endpoint.base_url) {
        endpoint.context = props
            .get("n_ctx")
            .or_else(|| props.pointer("/default_generation_settings/n_ctx"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(endpoint.context);
        if let Some(model_id) = props
            .get("model_alias")
            .or_else(|| props.get("model_path"))
            .and_then(serde_json::Value::as_str)
        {
            endpoint.model_id = model_id.into();
        }
        endpoint.resource_plan = props.get("resource_plan").cloned();
    }

    #[cfg(target_os = "linux")]
    {
        let model_path = endpoint.model_path.to_string_lossy();
        let matching_arguments = llama_server_pids().into_iter().find_map(|pid| {
            let bytes = fs::read(format!("/proc/{pid}/cmdline")).ok()?;
            let arguments = bytes
                .split(|byte| *byte == 0)
                .filter(|value| !value.is_empty())
                .map(|value| String::from_utf8_lossy(value).into_owned())
                .collect::<Vec<_>>();
            arguments
                .iter()
                .any(|argument| argument == model_path.as_ref())
                .then_some(arguments)
        });
        let Some(arguments) = matching_arguments else {
            return;
        };
        if let Some(context) =
            argument_value(&arguments, &["-c", "--ctx-size"]).and_then(|value| value.parse().ok())
        {
            endpoint.context = context;
        }
        let key_type = argument_value(&arguments, &["-ctk", "--cache-type-k"]);
        let value_type = argument_value(&arguments, &["-ctv", "--cache-type-v"]);
        endpoint.kv_type = match (key_type, value_type) {
            (Some(key), Some(value)) if key != value => Some(format!("{key}/{value}")),
            (Some(key), _) => Some(key),
            (_, Some(value)) => Some(value),
            _ => endpoint.kv_type.take(),
        };
        endpoint.batch_size = argument_value(&arguments, &["-b", "--batch-size"])
            .and_then(|value| value.parse().ok())
            .or(endpoint.batch_size);
        endpoint.ubatch_size = argument_value(&arguments, &["-ub", "--ubatch-size"])
            .and_then(|value| value.parse().ok())
            .or(endpoint.ubatch_size);
    }
}

fn apply_model_environment(command: &mut CommandBuilder, endpoint: &ModelEndpoint) {
    let context = endpoint.context.to_string();
    let model_path = endpoint.model_path.to_string_lossy();
    let variables = [
        ("OPENAI_BASE_URL", endpoint.base_url.as_str()),
        ("OPENAI_API_KEY", endpoint.api_key.as_str()),
        ("OPENAI_MODEL", endpoint.model_id.as_str()),
        ("LLM_MODEL", endpoint.model_id.as_str()),
        ("LLM_CONTEXT_LENGTH", context.as_str()),
        ("HERMES_INFERENCE_MODEL", endpoint.model_id.as_str()),
        ("ESE_ENDPOINT", endpoint.base_url.as_str()),
        ("ESE_API_KEY", endpoint.api_key.as_str()),
        ("ESE_MODEL_ID", endpoint.model_id.as_str()),
        ("ESE_MODEL_PATH", model_path.as_ref()),
        ("ESE_MODEL_CONTEXT", context.as_str()),
    ];
    for (key, value) in variables {
        command.env(key, value);
    }
    for (key, value) in [
        ("ESE_MODEL_ARCHITECTURE", endpoint.architecture.as_deref()),
        ("ESE_MODEL_QUANTIZATION", endpoint.quantization.as_deref()),
        ("ESE_KV_TYPE", endpoint.kv_type.as_deref()),
    ] {
        if let Some(value) = value {
            command.env(key, value);
        }
    }
    if let Some(batch_size) = endpoint.batch_size {
        command.env("ESE_BATCH_SIZE", batch_size.to_string());
    }
    if let Some(ubatch_size) = endpoint.ubatch_size {
        command.env("ESE_UBATCH_SIZE", ubatch_size.to_string());
    }
    if let Some(resource_plan) = &endpoint.resource_plan {
        command.env("ESE_RESOURCE_PLAN_JSON", resource_plan.to_string());
    }
    if let Ok(model_info) = serde_json::to_string(endpoint) {
        command.env("ESE_MODEL_INFO_JSON", model_info);
    }
}

fn spawn_terminal(
    request: LaunchRequest,
    app: AppHandle,
    state: &StudioState,
) -> Result<String, String> {
    if request.role.as_deref() == Some("model") {
        if let Some(existing) = state
            .active_model
            .lock()
            .map_err(|_| "model state is poisoned")?
            .take()
        {
            stop_session(&existing.session_id, state)?;
        }
    }
    let endpoint = if request.role.as_deref() == Some("model") {
        Some(
            request
                .model_endpoint
                .clone()
                .ok_or_else(|| "model launch is missing endpoint metadata".to_string())?,
        )
    } else if request.endpoint_aware {
        let mut active = state
            .active_model
            .lock()
            .map_err(|_| "model state is poisoned")?;
        let model = active
            .as_mut()
            .ok_or_else(|| "start a model before launching an endpoint-aware app".to_string())?;
        refresh_model_endpoint(&mut model.endpoint);
        Some(model.endpoint.clone())
    } else {
        None
    };
    if request.app_id.as_deref() == Some("hermes") {
        configure_hermes(
            &request.command,
            endpoint
                .as_ref()
                .ok_or_else(|| "Hermes requires an active model endpoint".to_string())?,
        )?;
    }
    let session_id = Uuid::new_v4().to_string();
    let pair = native_pty_system()
        .openpty(PtySize {
            rows: request.rows.unwrap_or(30),
            cols: request.columns.unwrap_or(100),
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|error| error.to_string())?;
    let mut command = terminal_command(&request.command, &request.args);
    if let Some(endpoint) = &endpoint {
        apply_model_environment(&mut command, endpoint);
    }
    if let Some(directory) = &request.working_directory {
        command.cwd(directory);
    }
    let child = pair
        .slave
        .spawn_command(command)
        .map_err(|error| error.to_string())?;
    drop(pair.slave);
    let mut reader = pair
        .master
        .try_clone_reader()
        .map_err(|error| error.to_string())?;
    let writer = pair
        .master
        .take_writer()
        .map_err(|error| error.to_string())?;
    state
        .terminals
        .lock()
        .map_err(|_| "terminal state is poisoned")?
        .insert(
            session_id.clone(),
            TerminalSession {
                master: pair.master,
                child,
                writer,
            },
        );
    if request.role.as_deref() == Some("model") {
        *state
            .active_model
            .lock()
            .map_err(|_| "model state is poisoned")? = Some(ManagedModel {
            session_id: session_id.clone(),
            request,
            endpoint: endpoint.expect("model endpoint was validated"),
        });
    }
    let output_id = session_id.clone();
    thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        while let Ok(count) = reader.read(&mut buffer) {
            if count == 0 {
                break;
            }
            let _ = app.emit(
                "terminal-output",
                TerminalOutput {
                    session_id: output_id.clone(),
                    data: String::from_utf8_lossy(&buffer[..count]).into_owned(),
                },
            );
        }
        let managed_state = app.state::<StudioState>();
        if let Ok(mut terminals) = managed_state.terminals.lock() {
            terminals.remove(&output_id);
        }
        if let Ok(mut active) = managed_state.active_model.lock() {
            if active
                .as_ref()
                .is_some_and(|model| model.session_id == output_id)
            {
                *active = None;
            }
        }
        let _ = app.emit("terminal-exit", output_id);
    });
    Ok(session_id)
}

fn terminal_command(command: &str, arguments: &[String]) -> CommandBuilder {
    #[cfg(target_os = "windows")]
    if Path::new(command).extension().is_some_and(|extension| {
        extension.eq_ignore_ascii_case("cmd") || extension.eq_ignore_ascii_case("bat")
    }) {
        let mut builder = CommandBuilder::new("cmd.exe");
        builder.args(["/d", "/s", "/c"]);
        builder.arg(windows_command_line(
            command,
            arguments.iter().map(String::as_str),
        ));
        return builder;
    }
    let mut builder = CommandBuilder::new(command);
    builder.args(arguments);
    builder
}

#[tauri::command]
fn write_terminal(
    session_id: String,
    data: String,
    state: State<'_, StudioState>,
) -> Result<(), String> {
    let mut terminals = state
        .terminals
        .lock()
        .map_err(|_| "terminal state is poisoned")?;
    let session = terminals
        .get_mut(&session_id)
        .ok_or_else(|| "terminal session not found".to_string())?;
    session
        .writer
        .write_all(data.as_bytes())
        .map_err(|error| error.to_string())?;
    session.writer.flush().map_err(|error| error.to_string())
}

#[tauri::command]
fn resize_terminal(
    session_id: String,
    columns: u16,
    rows: u16,
    state: State<'_, StudioState>,
) -> Result<(), String> {
    let terminals = state
        .terminals
        .lock()
        .map_err(|_| "terminal state is poisoned")?;
    let session = terminals
        .get(&session_id)
        .ok_or_else(|| "terminal session not found".to_string())?;
    session
        .master
        .resize(PtySize {
            rows,
            cols: columns,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn stop_terminal(session_id: String, state: State<'_, StudioState>) -> Result<(), String> {
    stop_session(&session_id, state.inner())
}

fn stop_session(session_id: &str, state: &StudioState) -> Result<(), String> {
    let mut session = state
        .terminals
        .lock()
        .map_err(|_| "terminal state is poisoned")?
        .remove(session_id)
        .ok_or_else(|| "terminal session not found".to_string())?;
    session.child.kill().map_err(|error| error.to_string())?;
    let _ = session.child.wait();
    if let Ok(mut active) = state.active_model.lock() {
        if active
            .as_ref()
            .is_some_and(|model| model.session_id == session_id)
        {
            *active = None;
        }
    }
    Ok(())
}

#[tauri::command]
fn plan_sweep(request: SweepRequest) -> Result<SweepPlan, String> {
    sweep::plan(&request)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RestoredTerminal {
    session_id: String,
    name: String,
    command: String,
}

#[tauri::command]
fn get_sweep_status(state: State<'_, StudioState>) -> Result<Option<SweepStatus>, String> {
    state.sweep.current()
}

#[tauri::command]
fn cancel_sweep(state: State<'_, StudioState>) {
    state.sweep.cancel();
}

#[tauri::command]
fn promote_sweep(state: State<'_, StudioState>) -> Result<ModelProfile, String> {
    let status = state
        .sweep
        .current()?
        .ok_or_else(|| "no sweep result is available".to_string())?;
    if status.state != "complete" {
        return Err("only a completed sweep can be applied".into());
    }

    let context = status
        .promoted_context
        .ok_or_else(|| "the completed sweep has no promoted context".to_string())?;
    let kv_type = status
        .best_kv_type
        .clone()
        .ok_or_else(|| "the completed sweep has no stable KV result".to_string())?;
    let batch_size = status
        .best_batch_size
        .ok_or_else(|| "the completed sweep has no stable batch result".to_string())?;
    let path = status.request.model_path.clone();

    let mut config = load_config()?;
    let discovered = discover_models(&config)
        .into_iter()
        .find(|model| model.path == path);
    let mut profile = discovered.unwrap_or_else(|| ModelProfile {
        id: model_id(&path),
        name: model_name(&path),
        path: path.clone(),
        size_bytes: fs::metadata(&path).ok().map(|metadata| metadata.len()),
        available: path.is_file(),
        architecture: None,
        quantization: infer_quantization(&path),
        context: None,
        kv_type: None,
        batch_size: None,
        ubatch_size: None,
        source: "discovered".into(),
    });
    profile.context = Some(context);
    profile.kv_type = Some(kv_type);
    profile.batch_size = Some(batch_size);
    profile.ubatch_size = Some(batch_size.min(512));
    profile.source = "sweep".into();

    config.models.retain(|model| model.path != path);
    config.models.push(profile.clone());
    write_config(&config)?;
    Ok(profile)
}

#[tauri::command]
fn start_sweep(
    request: SweepRequest,
    confirmed: bool,
    app: AppHandle,
    state: State<'_, StudioState>,
) -> Result<SweepStatus, String> {
    if !confirmed {
        return Err("exclusive GPU access must be confirmed before starting a sweep".into());
    }

    let restore = state
        .active_model
        .lock()
        .map_err(|_| "model state is poisoned")?
        .clone();
    if let Some(model) = &restore {
        stop_session(&model.session_id, state.inner())?;
        thread::sleep(Duration::from_millis(500));
    }
    if let Some(pid) = llama_server_pids().first() {
        if let Some(model) = restore.clone() {
            let _ = spawn_terminal(model.request, app.clone(), state.inner());
        }
        return Err(format!(
            "llama-server process {pid} is not managed by Studio; stop it before running an exclusive sweep"
        ));
    }
    if let Some(blocker_message) = gpu_blocker_message() {
        if let Some(model) = restore.clone() {
            let _ = spawn_terminal(model.request, app.clone(), state.inner());
        }
        return Err(blocker_message);
    }

    let ese_binary = locate_ese(Some(&app))
        .ok_or_else(|| "ESE launcher was not found in PATH or the Studio bundle".to_string())?;
    let checkpoint_root = project_dirs()?.config_dir().join("sweeps");
    let status = match state
        .sweep
        .start(request, ese_binary, checkpoint_root, app.clone())
    {
        Ok(status) => status,
        Err(error) => {
            if let Some(model) = restore {
                let _ = spawn_terminal(model.request, app, state.inner());
            }
            return Err(error);
        }
    };

    let telemetry_manager = state.sweep.clone();
    thread::spawn(move || loop {
        thread::sleep(Duration::from_millis(500));
        let Ok(Some(completed)) = telemetry_manager.current() else {
            break;
        };
        match completed.state.as_str() {
            "complete" => {
                if load_config().is_ok_and(|config| config.help_improve_ese) {
                    let _ = telemetry::queue_and_upload(&completed);
                }
                break;
            }
            "failed" | "cancelled" => break,
            _ => {}
        }
    });

    if let Some(model) = restore {
        let watcher_app = app.clone();
        let manager = state.sweep.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(500));
            let Ok(Some(status)) = manager.current() else {
                break;
            };
            if matches!(status.state.as_str(), "complete" | "failed" | "cancelled") {
                let managed_state = watcher_app.state::<StudioState>();
                if let Ok(session_id) = spawn_terminal(
                    model.request.clone(),
                    watcher_app.clone(),
                    managed_state.inner(),
                ) {
                    let model_path = model
                        .request
                        .args
                        .get(1)
                        .map(PathBuf::from)
                        .unwrap_or_default();
                    let _ = watcher_app.emit(
                        "terminal-restored",
                        RestoredTerminal {
                            session_id,
                            name: format!("Serving · {}", model_name(&model_path)),
                            command: model.request.command.clone(),
                        },
                    );
                }
                break;
            }
        });
    }
    Ok(status)
}

const GPU_BLOCKER_THRESHOLD_MIB: u64 = 512;

fn parse_gpu_consumers(output: &str) -> Vec<(u32, String, u64)> {
    let mut consumers: HashMap<u32, (String, u64)> = HashMap::new();
    for line in output.lines() {
        let mut fields = line.splitn(3, ',').map(str::trim);
        let (Some(pid), Some(name), Some(memory)) = (fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (Ok(pid), Ok(memory_mib)) = (pid.parse::<u32>(), memory.parse::<u64>()) else {
            continue;
        };
        let entry = consumers.entry(pid).or_insert_with(|| (name.to_owned(), 0));
        entry.1 = entry.1.saturating_add(memory_mib);
    }
    let current_pid = std::process::id();
    let mut blockers = consumers
        .into_iter()
        .filter(|(pid, (_, memory_mib))| {
            *pid != current_pid && *memory_mib >= GPU_BLOCKER_THRESHOLD_MIB
        })
        .map(|(pid, (name, memory_mib))| (pid, name, memory_mib))
        .collect::<Vec<_>>();
    blockers.sort_by_key(|blocker| std::cmp::Reverse(blocker.2));
    blockers
}

fn external_gpu_consumers() -> Vec<(u32, String, u64)> {
    ProcessCommand::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| parse_gpu_consumers(&String::from_utf8_lossy(&output.stdout)))
        .unwrap_or_default()
}

fn gpu_blocker_message() -> Option<String> {
    let blockers = external_gpu_consumers();
    if blockers.is_empty() {
        return None;
    }
    let details = blockers
        .iter()
        .map(|(pid, name, memory_mib)| {
            format!("{name} (PID {pid}, {:.1} GiB)", *memory_mib as f64 / 1024.0)
        })
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "GPU benchmark preflight blocked by active CUDA workloads: {details}. Pause or stop them, then retry the sweep"
    ))
}

#[cfg(target_os = "linux")]
fn llama_server_pids() -> Vec<u32> {
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_string_lossy().parse::<u32>().ok())
        .filter(|pid| {
            let is_server = fs::read_to_string(format!("/proc/{pid}/comm"))
                .is_ok_and(|name| name.trim() == "llama-server");
            let is_zombie = fs::read_to_string(format!("/proc/{pid}/stat"))
                .ok()
                .is_some_and(|stat| {
                    stat.rsplit_once(')')
                        .is_some_and(|(_, remainder)| remainder.trim_start().starts_with('Z'))
                });
            is_server && !is_zombie
        })
        .collect()
}

#[cfg(target_os = "windows")]
fn llama_server_pids() -> Vec<u32> {
    let output = ProcessCommand::new("tasklist.exe")
        .args(["/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let fields = line
                .trim()
                .trim_matches('"')
                .split("\",\"")
                .collect::<Vec<_>>();
            (fields
                .first()
                .is_some_and(|name| name.eq_ignore_ascii_case("llama-server.exe")))
            .then(|| fields.get(1)?.replace(',', "").parse().ok())
            .flatten()
        })
        .collect()
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn llama_server_pids() -> Vec<u32> {
    Vec::new()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(StudioState::default())
        .invoke_handler(tauri::generate_handler![
            get_studio_snapshot,
            search_hf_models,
            get_hf_model_details,
            start_hf_download,
            get_hf_download_status,
            cancel_hf_download,
            save_config,
            set_help_improve_ese,
            get_chat_status,
            start_chat,
            cancel_chat,
            add_model_root,
            add_app_profile,
            launch_terminal,
            write_terminal,
            resize_terminal,
            stop_terminal,
            plan_sweep,
            get_sweep_status,
            start_sweep,
            cancel_sweep,
            promote_sweep
        ])
        .run(tauri::generate_context!())
        .expect("error while running ESE Studio");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacing_an_existing_config_preserves_the_new_contents() {
        let root = std::env::temp_dir().join(format!("ese-replace-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("studio.toml");
        let temporary = root.join("studio.tmp");
        fs::write(&destination, "old").unwrap();
        fs::write(&temporary, "new").unwrap();

        replace_file(&temporary, &destination).unwrap();

        assert_eq!(fs::read_to_string(&destination).unwrap(), "new");
        assert!(!temporary.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn chat_stream_parser_extracts_only_text_deltas() {
        assert_eq!(
            chat_delta_content(r#"{"choices":[{"delta":{"content":"hello"}}]}"#).as_deref(),
            Some("hello")
        );
        assert_eq!(
            chat_delta_content(r#"{"choices":[{"delta":{"role":"assistant"}}]}"#),
            None
        );
        assert_eq!(chat_delta_content("not json"), None);
    }

    #[test]
    fn quantization_is_inferred_from_common_names() {
        assert_eq!(
            infer_quantization(Path::new("model.Q4_K_M.gguf")).as_deref(),
            Some("Q4_K_M")
        );
        assert_eq!(infer_quantization(Path::new("model.gguf")), None);
    }

    #[test]
    fn gpu_preflight_aggregates_large_multi_gpu_consumers() {
        let sample = "508393, /opt/comfy/python, 5124\n508393, /opt/comfy/python, 216\n4356, /usr/bin/kwin_wayland, 115\n";
        let blockers = parse_gpu_consumers(sample);
        assert_eq!(blockers, vec![(508393, "/opt/comfy/python".into(), 5340)]);
    }
    #[test]
    fn ids_are_stable_and_safe() {
        assert_eq!(
            model_id(Path::new("/Models/My_Model.Q4.gguf")),
            "models-my-model-q4-gguf"
        );
    }

    #[test]
    fn auxiliary_ggufs_are_not_launchable_models() {
        assert!(!is_launchable_gguf(Path::new("mmproj-F16.gguf")));
        assert!(!is_launchable_gguf(Path::new("model.imatrix.gguf")));
        assert!(is_launchable_gguf(Path::new("model.Q4_K_M.gguf")));
    }

    #[test]
    fn detected_apps_are_added_without_overwriting_user_profiles() {
        let mut configured = vec![AppProfile {
            id: "hermes".into(),
            name: "My Hermes".into(),
            command: "/custom/hermes".into(),
            args: vec!["--tui".into()],
            working_directory: Some(PathBuf::from("/workspace")),
            endpoint_aware: false,
        }];
        let detected = vec![
            AppProfile {
                id: "hermes".into(),
                name: "Hermes".into(),
                command: "/home/me/.local/bin/hermes".into(),
                args: vec!["chat".into()],
                working_directory: None,
                endpoint_aware: true,
            },
            AppProfile {
                id: "codex".into(),
                name: "Codex".into(),
                command: "/usr/bin/codex".into(),
                args: Vec::new(),
                working_directory: None,
                endpoint_aware: true,
            },
        ];

        assert!(merge_detected_apps(&mut configured, detected));
        assert_eq!(configured.len(), 2);
        assert_eq!(configured[0].name, "My Hermes");
        assert_eq!(configured[0].command, "/custom/hermes");
        assert!(!merge_detected_apps(&mut configured, Vec::new()));
    }

    #[test]
    fn endpoint_aware_apps_receive_the_complete_model_handoff() {
        let endpoint = ModelEndpoint {
            base_url: "http://127.0.0.1:8080/v1".into(),
            api_key: "sk-local-placeholder".into(),
            model_id: "/models/model.Q4_K_M.gguf".into(),
            model_path: PathBuf::from("/models/model.Q4_K_M.gguf"),
            context: 65_536,
            architecture: Some("qwen3moe".into()),
            quantization: Some("Q4_K_M".into()),
            kv_type: Some("q8_0".into()),
            batch_size: Some(512),
            ubatch_size: Some(256),
            resource_plan: Some(serde_json::json!({"policy": "resident"})),
        };
        let mut command = CommandBuilder::new("agent");
        apply_model_environment(&mut command, &endpoint);

        for (key, expected) in [
            ("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
            ("OPENAI_API_KEY", "sk-local-placeholder"),
            ("OPENAI_MODEL", "/models/model.Q4_K_M.gguf"),
            ("ESE_MODEL_PATH", "/models/model.Q4_K_M.gguf"),
            ("ESE_MODEL_CONTEXT", "65536"),
            ("ESE_MODEL_ARCHITECTURE", "qwen3moe"),
            ("ESE_MODEL_QUANTIZATION", "Q4_K_M"),
            ("ESE_KV_TYPE", "q8_0"),
            ("ESE_BATCH_SIZE", "512"),
            ("ESE_UBATCH_SIZE", "256"),
            ("ESE_RESOURCE_PLAN_JSON", "{\"policy\":\"resident\"}"),
        ] {
            assert_eq!(
                command.get_env(key).and_then(|value| value.to_str()),
                Some(expected)
            );
        }
    }

    #[test]
    fn standard_sweep_promotes_the_safe_context() {
        let plan = plan_sweep(SweepRequest {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
            preset: "standard".into(),
            objective: "max-safe-context".into(),
            max_context: 131_072,
            safe_margin: 0.9,
        })
        .expect("sweep preview should be valid");

        assert_eq!(plan.promoted_context, 117_760);
        assert_eq!(plan.trial_count, 14);
        assert!(plan.requires_exclusive_gpu);
    }
}
