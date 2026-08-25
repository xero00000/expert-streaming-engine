use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    cmp::Ordering,
    fs::{self, File},
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepRequest {
    pub model_path: PathBuf,
    pub preset: String,
    pub objective: String,
    pub max_context: u64,
    pub safe_margin: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepPlan {
    pub model_path: PathBuf,
    pub preset: String,
    pub objective: String,
    pub candidate_contexts: Vec<u64>,
    pub kv_types: Vec<String>,
    pub batch_sizes: Vec<u32>,
    pub slot_counts: Vec<u32>,
    pub promoted_context: u64,
    pub trial_count: usize,
    pub requires_exclusive_gpu: bool,
    pub safe_margin: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrialResult {
    pub phase: String,
    pub context: u64,
    pub kv_type: String,
    pub batch_size: u32,
    #[serde(default)]
    pub tensor_split: Option<String>,
    #[serde(default = "default_slots")]
    pub slots: u32,
    pub stable: bool,
    pub tokens_per_second: Option<f64>,
    pub elapsed_seconds: f64,
    pub error: Option<String>,
    pub log_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SweepStatus {
    pub id: String,
    pub state: String,
    pub phase: String,
    pub request: SweepRequest,
    pub completed_trials: usize,
    pub total_trials: usize,
    pub current_label: String,
    pub verified_max_context: Option<u64>,
    pub promoted_context: Option<u64>,
    pub best_kv_type: Option<String>,
    pub best_batch_size: Option<u32>,
    pub best_slots: Option<u32>,
    pub best_tokens_per_second: Option<f64>,
    pub results: Vec<TrialResult>,
    pub error: Option<String>,
    pub checkpoint_path: PathBuf,
}

fn default_slots() -> u32 {
    1
}

#[derive(Clone, Default)]
pub struct SweepManager {
    status: Arc<Mutex<Option<SweepStatus>>>,
    cancel: Arc<AtomicBool>,
}

impl SweepManager {
    pub fn current(&self) -> Result<Option<SweepStatus>, String> {
        self.status
            .lock()
            .map(|status| status.clone())
            .map_err(|_| "sweep state is poisoned".into())
    }

    pub fn cancel(&self) {
        self.cancel.store(true, AtomicOrdering::SeqCst);
    }

    pub fn start(
        &self,
        request: SweepRequest,
        ese_binary: PathBuf,
        checkpoint_root: PathBuf,
        app: AppHandle,
    ) -> Result<SweepStatus, String> {
        if self.current()?.is_some_and(|status| {
            matches!(
                status.state.as_str(),
                "preparing" | "running" | "cancelling"
            )
        }) {
            return Err("a sweep is already running".into());
        }
        let plan = plan(&request)?;
        fs::create_dir_all(&checkpoint_root)
            .map_err(|error| format!("failed to create {}: {error}", checkpoint_root.display()))?;
        let mut status = resumable_status(&checkpoint_root, &request).unwrap_or_else(|| {
            let id = Uuid::new_v4().to_string();
            SweepStatus {
                checkpoint_path: checkpoint_root.join(format!("{id}.json")),
                id,
                state: "preparing".into(),
                phase: "capacity".into(),
                request: request.clone(),
                completed_trials: 0,
                total_trials: plan.trial_count,
                current_label: "Preparing exclusive benchmark run".into(),
                verified_max_context: None,
                promoted_context: None,
                best_kv_type: None,
                best_batch_size: None,
                best_slots: None,
                best_tokens_per_second: None,
                results: Vec::new(),
                error: None,
            }
        });
        status.state = "preparing".into();
        status.error = None;
        status.total_trials = plan.trial_count;
        status.current_label = if status.completed_trials > 0 {
            format!(
                "Resuming from {} checkpointed trials",
                status.completed_trials
            )
        } else {
            "Preparing exclusive benchmark run".into()
        };
        *self.status.lock().map_err(|_| "sweep state is poisoned")? = Some(status.clone());
        self.cancel.store(false, AtomicOrdering::SeqCst);

        let manager = self.clone();
        thread::spawn(move || manager.run(plan, ese_binary, app));
        Ok(status)
    }

    fn run(&self, plan: SweepPlan, ese_binary: PathBuf, app: AppHandle) {
        self.update(&app, |status| {
            status.state = "running".into();
            status.current_label = "Testing context allocation".into();
        });

        let mut contexts = plan.candidate_contexts.clone();
        contexts.sort_unstable_by(|left, right| right.cmp(left));
        let existing = self.current().ok().flatten();
        let mut verified_max = existing
            .as_ref()
            .and_then(|status| status.verified_max_context);

        for context in contexts {
            if verified_max.is_some() {
                break;
            }
            if existing.as_ref().is_some_and(|status| {
                status
                    .results
                    .iter()
                    .any(|result| result.phase == "capacity" && result.context == context)
            }) {
                continue;
            }
            if self.cancelled(&app) {
                return;
            }
            if let Some(message) = crate::gpu_blocker_message() {
                self.fail(&app, message);
                return;
            }
            self.update(&app, |status| {
                status.phase = "capacity".into();
                status.current_label = format!("Verifying {context} token context");
            });
            let result = run_trial(
                &ese_binary,
                &plan.model_path,
                context,
                "q8_0",
                512,
                false,
                1,
                1,
                &self.cancel,
                &checkpoint_directory(&self.current().ok().flatten()),
            );
            if let Some(message) = crate::gpu_blocker_message() {
                self.fail(&app, message);
                return;
            }
            if result.error.as_deref() == Some("cancelled") {
                self.cancelled(&app);
                return;
            }
            let stable = result.stable;
            self.update(&app, |status| {
                status.completed_trials += 1;
                status.results.push(result);
            });
            if stable {
                verified_max = Some(context);
                break;
            }
        }

        let Some(verified_max) = verified_max else {
            let detail = self
                .current()
                .ok()
                .flatten()
                .and_then(|status| {
                    status
                        .results
                        .iter()
                        .rev()
                        .find_map(|result| result.error.as_deref())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "the server did not provide diagnostics".into());
            self.fail(
                &app,
                format!("no candidate context completed stable allocation and inference: {detail}"),
            );
            return;
        };
        let promoted = floor_context((verified_max as f64 * plan.safe_margin as f64) as u64);
        self.update(&app, |status| {
            status.verified_max_context = Some(verified_max);
            status.promoted_context = Some(promoted);
            status.phase = "tuning".into();
        });

        let repetitions = match plan.preset.as_str() {
            "exhaustive" => 3,
            "standard" => 2,
            _ => 1,
        };
        for kv_type in &plan.kv_types {
            for batch_size in &plan.batch_sizes {
                for slots in &plan.slot_counts {
                    if existing.as_ref().is_some_and(|status| {
                        status.results.iter().any(|result| {
                            result.phase == "tuning"
                                && result.context == promoted
                                && result.kv_type == *kv_type
                                && result.batch_size == *batch_size
                                && result.slots == *slots
                        })
                    }) {
                        continue;
                    }
                    if self.cancelled(&app) {
                        return;
                    }
                    if let Some(message) = crate::gpu_blocker_message() {
                        self.fail(&app, message);
                        return;
                    }
                    self.update(&app, |status| {
                        status.current_label = format!(
                            "Benchmarking {promoted} context · {kv_type} KV · batch {batch_size} · {slots} session(s)"
                        );
                    });
                    let result = run_trial(
                        &ese_binary,
                        &plan.model_path,
                        promoted,
                        kv_type,
                        *batch_size,
                        true,
                        repetitions,
                        *slots,
                        &self.cancel,
                        &checkpoint_directory(&self.current().ok().flatten()),
                    );
                    if let Some(message) = crate::gpu_blocker_message() {
                        self.fail(&app, message);
                        return;
                    }
                    if result.error.as_deref() == Some("cancelled") {
                        self.cancelled(&app);
                        return;
                    }
                    self.update(&app, |status| {
                        status.completed_trials += 1;
                        if candidate_is_better(status, &result) {
                            status.best_tokens_per_second = result.tokens_per_second;
                            status.best_kv_type = Some(result.kv_type.clone());
                            status.best_batch_size = Some(result.batch_size);
                            status.best_slots = Some(result.slots);
                        }
                        status.results.push(result);
                    });
                }
            }
        }

        self.update(&app, |status| {
            status.state = "complete".into();
            status.phase = "complete".into();
            status.completed_trials = status.total_trials;
            status.current_label = "Sweep complete — verified profile ready for review".into();
        });
    }

    fn update(&self, app: &AppHandle, update: impl FnOnce(&mut SweepStatus)) {
        if let Ok(mut guard) = self.status.lock() {
            if let Some(status) = guard.as_mut() {
                update(status);
                persist(status);
                let _ = app.emit("sweep-progress", status.clone());
            }
        }
    }

    fn fail(&self, app: &AppHandle, message: impl Into<String>) {
        let message = message.into();
        self.update(app, |status| {
            status.state = "failed".into();
            status.phase = "failed".into();
            status.current_label = "Sweep failed".into();
            status.error = Some(message);
        });
    }

    fn cancelled(&self, app: &AppHandle) -> bool {
        if !self.cancel.load(AtomicOrdering::SeqCst) {
            return false;
        }
        self.update(app, |status| {
            status.state = "cancelled".into();
            status.phase = "cancelled".into();
            status.current_label = "Sweep cancelled; completed trials were checkpointed".into();
        });
        true
    }
}

fn checkpoint_directory(status: &Option<SweepStatus>) -> PathBuf {
    status
        .as_ref()
        .and_then(|status| status.checkpoint_path.parent().map(Path::to_path_buf))
        .unwrap_or_else(std::env::temp_dir)
}

fn persist(status: &SweepStatus) {
    if let Ok(serialized) = serde_json::to_vec_pretty(status) {
        let temporary = status.checkpoint_path.with_extension("json.tmp");
        if fs::write(&temporary, serialized).is_ok() {
            let _ = fs::rename(temporary, &status.checkpoint_path);
        }
    }
}

fn floor_context(context: u64) -> u64 {
    if context < 512 {
        512
    } else {
        (context / 256) * 256
    }
}

fn kv_memory_rank(kv_type: &str) -> u8 {
    match kv_type {
        "q4_0" => 0,
        "q8_0" => 1,
        "f16" => 2,
        _ => 3,
    }
}

fn candidate_is_better(status: &SweepStatus, candidate: &TrialResult) -> bool {
    if !candidate.stable {
        return false;
    }
    let current = status.results.iter().find(|result| {
        result.stable
            && status.best_kv_type.as_deref() == Some(result.kv_type.as_str())
            && status.best_batch_size == Some(result.batch_size)
            && status.best_slots == Some(result.slots)
    });
    let Some(current) = current else {
        return true;
    };
    match status.request.objective.as_str() {
        "max-concurrency" => {
            (
                candidate.slots,
                candidate.tokens_per_second.unwrap_or(f64::NEG_INFINITY),
            ) > (
                current.slots,
                current.tokens_per_second.unwrap_or(f64::NEG_INFINITY),
            )
        }
        "minimum-vram" => {
            (kv_memory_rank(&candidate.kv_type), candidate.batch_size)
                < (kv_memory_rank(&current.kv_type), current.batch_size)
        }
        "lowest-latency" => candidate.elapsed_seconds < current.elapsed_seconds,
        _ => {
            candidate.tokens_per_second.unwrap_or(f64::NEG_INFINITY)
                > current.tokens_per_second.unwrap_or(f64::NEG_INFINITY)
        }
    }
}

pub fn plan(request: &SweepRequest) -> Result<SweepPlan, String> {
    if !request.model_path.is_file() {
        return Err(format!(
            "model does not exist: {}",
            request.model_path.display()
        ));
    }
    if !(0.5..=1.0).contains(&request.safe_margin) {
        return Err("safe margin must be between 0.5 and 1.0".into());
    }
    if request.max_context < 512 {
        return Err("maximum context must be at least 512".into());
    }
    if !matches!(
        request.objective.as_str(),
        "max-safe-context"
            | "balanced"
            | "max-throughput"
            | "minimum-vram"
            | "lowest-latency"
            | "max-concurrency"
    ) {
        return Err("unsupported sweep objective".into());
    }
    let sample_count = match request.preset.as_str() {
        "quick" => 4,
        "standard" => 7,
        "exhaustive" => 10,
        _ => return Err("preset must be quick, standard, or exhaustive".into()),
    };
    let minimum = 4096_u64.min(request.max_context);
    let step = ((request.max_context.saturating_sub(minimum)) / sample_count as u64).max(1);
    let mut candidate_contexts = (0..=sample_count)
        .map(|index| {
            minimum
                .saturating_add(step * index as u64)
                .min(request.max_context)
        })
        .collect::<Vec<_>>();
    candidate_contexts.sort_unstable();
    candidate_contexts.dedup();
    let kv_types = if request.preset == "quick" {
        vec!["q8_0".into(), "f16".into()]
    } else {
        vec!["q4_0".into(), "q8_0".into(), "f16".into()]
    };
    let batch_sizes = if request.preset == "exhaustive" {
        vec![128, 256, 512, 1024]
    } else {
        vec![256, 512]
    };
    let slot_counts = if request.objective == "max-concurrency" {
        if request.preset == "quick" {
            vec![1, 2]
        } else {
            vec![1, 2, 4]
        }
    } else {
        vec![1]
    };
    let promoted_context =
        floor_context((request.max_context as f64 * request.safe_margin as f64) as u64);
    let trial_count =
        candidate_contexts.len() + kv_types.len() * batch_sizes.len() * slot_counts.len();
    Ok(SweepPlan {
        model_path: request.model_path.clone(),
        preset: request.preset.clone(),
        objective: request.objective.clone(),
        candidate_contexts,
        kv_types,
        batch_sizes,
        slot_counts,
        promoted_context,
        trial_count,
        requires_exclusive_gpu: true,
        safe_margin: request.safe_margin,
    })
}

fn resumable_status(root: &Path, request: &SweepRequest) -> Option<SweepStatus> {
    let mut checkpoints = fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .is_some_and(|extension| extension == "json")
        })
        .filter_map(|entry| {
            let modified = entry.metadata().ok()?.modified().ok()?;
            let status: SweepStatus = serde_json::from_slice(&fs::read(entry.path()).ok()?).ok()?;
            Some((modified, status))
        })
        .filter(|(_, status)| {
            matches!(status.state.as_str(), "cancelled" | "failed")
                && status.request.model_path == request.model_path
                && status.request.preset == request.preset
                && status.request.objective == request.objective
                && status.request.max_context == request.max_context
                && (status.request.safe_margin - request.safe_margin).abs() < f32::EPSILON
        })
        .collect::<Vec<_>>();
    checkpoints.sort_by_key(|checkpoint| std::cmp::Reverse(checkpoint.0));
    checkpoints.into_iter().next().map(|(_, status)| status)
}

#[allow(clippy::too_many_arguments)]
fn run_trial(
    ese_binary: &Path,
    model: &Path,
    context: u64,
    kv_type: &str,
    batch_size: u32,
    benchmark: bool,
    repetitions: usize,
    slots: u32,
    cancel: &AtomicBool,
    log_root: &Path,
) -> TrialResult {
    let started = Instant::now();
    let phase = if benchmark { "tuning" } else { "capacity" };
    let log_path = log_root.join(format!(
        "trial-{phase}-{context}-{kv_type}-{batch_size}-slots{slots}.log"
    ));
    let mut result = TrialResult {
        phase: phase.into(),
        context,
        kv_type: kv_type.into(),
        batch_size,
        tensor_split: None,
        slots,
        stable: false,
        tokens_per_second: None,
        elapsed_seconds: 0.0,
        error: None,
        log_path: log_path.clone(),
    };

    let trial = (|| -> Result<Option<f64>, String> {
        let port = free_port()?;
        let plan_output = Command::new(ese_binary)
            .arg("plan")
            .arg(model)
            .args(["-c", &context.to_string(), "--kv", kv_type])
            .args(["--batch-size", &batch_size.to_string()])
            .args(["--ubatch-size", &batch_size.min(512).to_string()])
            .args(["--slots", &slots.to_string()])
            .args(["--host", "127.0.0.1", "--port", &port.to_string(), "--json"])
            .output()
            .map_err(|error| format!("failed to run ESE planner: {error}"))?;
        if !plan_output.status.success() {
            return Err(String::from_utf8_lossy(&plan_output.stderr)
                .trim()
                .to_owned());
        }
        let plan_json: Value = serde_json::from_slice(&plan_output.stdout)
            .map_err(|error| format!("invalid ESE plan JSON: {error}"))?;
        let command = plan_json["command"]
            .as_array()
            .ok_or_else(|| "ESE plan omitted command array".to_string())?
            .iter()
            .map(|part| {
                part.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "invalid command part".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (program, arguments) = command
            .split_first()
            .ok_or_else(|| "empty ESE command".to_string())?;
        result.tensor_split = command_argument(arguments, &["--tensor-split", "-ts"]);
        let log = File::create(&log_path).map_err(|error| error.to_string())?;
        let error_log = log.try_clone().map_err(|error| error.to_string())?;
        let mut child = Command::new(program)
            .args(arguments)
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(error_log))
            .spawn()
            .map_err(|error| format!("failed to launch planned server: {error}"))?;
        let server_result = wait_for_server(&mut child, port, cancel, &log_path);
        if let Err(error) = server_result {
            terminate(&mut child);
            return Err(error);
        }
        let workload = (|| -> Result<Option<f64>, String> {
            if benchmark {
                let mut samples = Vec::with_capacity(repetitions);
                for _ in 0..repetitions {
                    if cancel.load(AtomicOrdering::SeqCst) {
                        return Err("cancelled".into());
                    }
                    samples.push(completion_throughput(port, slots)?);
                }
                samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
                Ok(Some(samples[samples.len() / 2]))
            } else {
                completion_tokens(port)?;
                Ok(None)
            }
        })();
        terminate(&mut child);
        workload
    })();

    if let Ok(log) = fs::read_to_string(&log_path) {
        if let Some(actual_kv) = actual_kv_quality(&log) {
            result.kv_type = actual_kv;
        }
    }
    match trial {
        Ok(speed) => {
            result.stable = true;
            result.tokens_per_second = speed;
        }
        Err(error) => {
            result.error = Some(if error.is_empty() {
                "trial failed without diagnostics".into()
            } else {
                error
            })
        }
    }
    result.elapsed_seconds = started.elapsed().as_secs_f64();
    result
}

fn command_argument(arguments: &[String], names: &[&str]) -> Option<String> {
    arguments
        .windows(2)
        .find_map(|pair| names.contains(&pair[0].as_str()).then(|| pair[1].clone()))
}

fn actual_kv_quality(log: &str) -> Option<String> {
    log.lines().rev().find_map(|line| {
        let json = line.strip_prefix("resource_plan: ")?;
        serde_json::from_str::<Value>(json)
            .ok()?
            .get("kv_quality")?
            .as_str()
            .map(str::to_owned)
    })
}

fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(|error| error.to_string())?;
    Ok(listener
        .local_addr()
        .map_err(|error| error.to_string())?
        .port())
}

fn wait_for_server(
    child: &mut Child,
    port: u16,
    cancel: &AtomicBool,
    log_path: &Path,
) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(180);
    while Instant::now() < deadline {
        if cancel.load(AtomicOrdering::SeqCst) {
            return Err("cancelled".into());
        }
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            let diagnostic = log_tail(log_path, 12);
            return Err(if diagnostic.is_empty() {
                format!(
                    "server exited before becoming healthy ({status}); see {}",
                    log_path.display()
                )
            } else {
                format!("server exited before becoming healthy ({status}): {diagnostic}")
            });
        }
        if http_request(port, "GET", "/health", None).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err("server health check timed out after 180 seconds".into())
}

fn log_tail(path: &Path, line_count: usize) -> String {
    let Ok(contents) = fs::read_to_string(path) else {
        return String::new();
    };
    let lines = contents.lines().collect::<Vec<_>>();
    let diagnostic = lines
        .iter()
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            [
                "error",
                "failed",
                "invalid",
                "unsupported",
                "exception",
                "cuda",
            ]
            .iter()
            .any(|needle| lower.contains(needle))
        })
        .take(line_count)
        .copied()
        .collect::<Vec<_>>();
    let selected = if diagnostic.is_empty() {
        &lines[lines.len().saturating_sub(line_count)..]
    } else {
        diagnostic.as_slice()
    };
    selected.join(" | ").chars().take(2_000).collect()
}

fn completion_speed(port: u16) -> Result<f64, String> {
    let parsed = completion_response(port)?;
    parsed["timings"]["predicted_per_second"]
        .as_f64()
        .or_else(|| {
            parsed["timings"]["predicted_n"]
                .as_f64()
                .zip(parsed["timings"]["predicted_ms"].as_f64())
                .map(|(tokens, ms)| tokens * 1000.0 / ms)
        })
        .ok_or_else(|| "completion response omitted throughput timing".into())
}

fn completion_tokens(port: u16) -> Result<f64, String> {
    completion_response(port)?["timings"]["predicted_n"]
        .as_f64()
        .ok_or_else(|| "completion response omitted generated token count".into())
}

fn completion_throughput(port: u16, slots: u32) -> Result<f64, String> {
    if slots <= 1 {
        return completion_speed(port);
    }
    let started = Instant::now();
    let total_tokens = thread::scope(|scope| {
        let handles = (0..slots)
            .map(|_| scope.spawn(|| completion_tokens(port)))
            .collect::<Vec<_>>();
        let mut peak_processing = 0_u32;
        while handles.iter().any(|handle| !handle.is_finished()) {
            if let Ok(processing) = health_slots_processing(port) {
                peak_processing = peak_processing.max(processing);
            }
            thread::sleep(Duration::from_millis(10));
        }
        if peak_processing < slots {
            return Err(format!(
                "server processed only {peak_processing} of {slots} requested concurrent sessions"
            ));
        }
        handles.into_iter().try_fold(0.0, |total, handle| {
            let tokens = handle
                .join()
                .map_err(|_| "concurrent completion worker panicked".to_string())??;
            Ok::<_, String>(total + tokens)
        })
    })?;
    let elapsed = started.elapsed().as_secs_f64();
    if elapsed <= 0.0 {
        return Err("concurrent completion elapsed time was zero".into());
    }
    Ok(total_tokens / elapsed)
}

fn health_slots_processing(port: u16) -> Result<u32, String> {
    let response = http_request(port, "GET", "/health", None)?;
    let parsed: Value =
        serde_json::from_str(&response).map_err(|error| format!("invalid health JSON: {error}"))?;
    parsed["slots_processing"]
        .as_u64()
        .and_then(|value| value.try_into().ok())
        .ok_or_else(|| "health response omitted slots_processing".into())
}

fn completion_response(port: u16) -> Result<Value, String> {
    let prompt = "A careful local inference benchmark checks stable throughput. ".repeat(64);
    let body = serde_json::json!({
        "prompt": prompt,
        "n_predict": 32,
        "temperature": 0,
        "cache_prompt": false,
    })
    .to_string();
    let response = http_request(port, "POST", "/completion", Some(&body))?;
    serde_json::from_str(&response).map_err(|error| format!("invalid completion JSON: {error}"))
}

fn http_request(port: u16, method: &str, path: &str, body: Option<&str>) -> Result<String, String> {
    let mut stream = TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}")
            .parse()
            .map_err(|error| format!("invalid endpoint: {error}"))?,
        Duration::from_secs(2),
    )
    .map_err(|error| error.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .map_err(|error| error.to_string())?;
    let payload = body.unwrap_or("");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    )
    .map_err(|error| error.to_string())?;
    stream.flush().map_err(|error| error.to_string())?;
    let mut response = String::new();
    stream
        .read_to_string(&mut response)
        .map_err(|error| error.to_string())?;
    let (headers, body) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    if !headers
        .lines()
        .next()
        .is_some_and(|line| line.contains(" 200 "))
    {
        return Err(format!(
            "HTTP request failed: {}",
            headers.lines().next().unwrap_or("unknown status")
        ));
    }
    Ok(body.to_owned())
}

fn terminate(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(preset: &str) -> SweepRequest {
        SweepRequest {
            model_path: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
            preset: preset.into(),
            objective: "max-safe-context".into(),
            max_context: 131_072,
            safe_margin: 0.9,
        }
    }

    #[test]
    fn standard_plan_separates_capacity_from_tuning_trials() {
        let plan = plan(&request("standard")).expect("plan should be valid");
        assert_eq!(plan.promoted_context, 117_760);
        assert_eq!(plan.trial_count, 14);
    }

    #[test]
    fn quick_plan_is_bounded() {
        let plan = plan(&request("quick")).expect("plan should be valid");
        assert!(plan.trial_count <= 10);
        assert_eq!(plan.kv_types, ["q8_0", "f16"]);
        assert_eq!(plan.slot_counts, [1]);
    }

    #[test]
    fn concurrency_objective_tests_bounded_session_counts() {
        let mut request = request("standard");
        request.objective = "max-concurrency".into();
        let plan = plan(&request).expect("concurrency plan should be valid");
        assert_eq!(plan.slot_counts, [1, 2, 4]);
        assert_eq!(plan.trial_count, 26);
    }

    #[test]
    fn cancelled_matching_checkpoint_is_resumable() {
        let root = std::env::temp_dir().join(format!("ese-studio-resume-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create resume fixture");
        let request = request("quick");
        let checkpoint_path = root.join("checkpoint.json");
        let status = SweepStatus {
            id: "resume-fixture".into(),
            state: "cancelled".into(),
            phase: "cancelled".into(),
            request: request.clone(),
            completed_trials: 1,
            total_trials: 9,
            current_label: "cancelled".into(),
            verified_max_context: Some(65_536),
            promoted_context: Some(58_880),
            best_kv_type: None,
            best_batch_size: None,
            best_slots: None,
            best_tokens_per_second: None,
            results: Vec::new(),
            error: None,
            checkpoint_path: checkpoint_path.clone(),
        };
        fs::write(
            &checkpoint_path,
            serde_json::to_vec_pretty(&status).expect("serialize checkpoint"),
        )
        .expect("write checkpoint");

        let resumed = resumable_status(&root, &request).expect("matching checkpoint");
        assert_eq!(resumed.id, status.id);
        assert_eq!(resumed.completed_trials, 1);

        let mut different = request;
        different.max_context += 256;
        assert!(resumable_status(&root, &different).is_none());
        fs::remove_dir_all(root).expect("remove resume fixture");
    }

    #[test]
    fn minimum_vram_prefers_quantized_kv_and_smaller_batches() {
        assert!(kv_memory_rank("q4_0") < kv_memory_rank("q8_0"));
        assert!(kv_memory_rank("q8_0") < kv_memory_rank("f16"));
    }

    #[test]
    fn runtime_kv_quality_overrides_the_requested_label() {
        let log = "noise\nresource_plan: {\"kv_quality\":\"f16\",\"context\":58368}\n";
        assert_eq!(actual_kv_quality(log).as_deref(), Some("f16"));
    }

    #[test]
    fn winning_trial_records_the_exact_planned_tensor_split() {
        let arguments = vec![
            "-m".into(),
            "model.gguf".into(),
            "--tensor-split".into(),
            "32,32,36".into(),
            "-c".into(),
            "65536".into(),
        ];
        assert_eq!(
            command_argument(&arguments, &["--tensor-split", "-ts"]).as_deref(),
            Some("32,32,36")
        );
    }

    #[test]
    fn server_failure_log_tail_is_bounded_and_actionable() {
        let root = std::env::temp_dir().join(format!("ese-studio-log-tail-{}", Uuid::new_v4()));
        fs::create_dir_all(&root).expect("create log fixture");
        let path = root.join("trial.log");
        fs::write(&path, "invalid parameter: --tensor-split\nhelp\nfooter\n")
            .expect("write log fixture");
        assert_eq!(log_tail(&path, 2), "invalid parameter: --tensor-split");
        fs::remove_dir_all(root).expect("remove log fixture");
    }

    #[test]
    fn real_server_trial_when_e2e_fixture_is_configured() {
        let Ok(model) = std::env::var("ESE_STUDIO_E2E_MODEL") else {
            return;
        };
        let Ok(ese) = std::env::var("ESE_STUDIO_E2E_BINARY") else {
            return;
        };
        let slots = std::env::var("ESE_STUDIO_E2E_SLOTS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1);
        let log_root = std::env::temp_dir().join(format!("ese-studio-e2e-{}", Uuid::new_v4()));
        fs::create_dir_all(&log_root).expect("create e2e log directory");
        let result = run_trial(
            Path::new(&ese),
            Path::new(&model),
            1024,
            "q8_0",
            256,
            true,
            1,
            slots,
            &AtomicBool::new(false),
            &log_root,
        );
        assert!(
            result.stable,
            "trial failed: {:?} ({})",
            result.error,
            result.log_path.display()
        );
        assert!(result.tokens_per_second.is_some_and(|speed| speed > 0.0));
    }
}
