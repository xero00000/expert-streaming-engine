import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getVersion } from "@tauri-apps/api/app";
import { relaunch } from "@tauri-apps/plugin-process";
import { check, type Update } from "@tauri-apps/plugin-updater";
import { Activity, AppWindow, Bot, Box, ChevronDown, ChevronRight, ChevronUp, CloudDownload, Download, Gauge, HardDrive, Heart, Library, MessageSquare, Play, Plus, RefreshCw, RotateCcw, Search, Send, Settings, ShieldCheck, SlidersHorizontal, Sparkles, SquareTerminal, StopCircle, Trash2, User, X } from "lucide-react";
import { TerminalPane } from "./components/TerminalPane";
import type { AppProfile, ChatStatus, DownloadStatus, HubModel, HubModelDetails, ModelProfile, StudioSnapshot, SweepPlan, SweepStatus, TerminalTab, View } from "./types";
import "./App.css";

const demoSnapshot: StudioSnapshot = {
  platform: navigator.platform.toLowerCase().includes("win") ? "windows" : "linux",
  configPath: "~/.config/ese/studio.toml",
  onboardingComplete: true,
  helpImproveEse: false,
  modelRoots: ["~/models"],
  eseBinary: "~/.local/bin/ese",
  models: [
    { id: "demo-1", name: "Qwen3 Coder 30B A3B", path: "~/models/Qwen3-Coder-30B-A3B.Q4_K_M.gguf", sizeBytes: 18_800_000_000, available: true, architecture: "qwen3moe", quantization: "Q4_K_M", context: 131072, source: "discovered" },
    { id: "demo-2", name: "Llama 3.1 8B Instruct", path: "~/models/Llama-3.1-8B-Instruct.Q6_K.gguf", sizeBytes: 6_600_000_000, available: true, architecture: "llama", quantization: "Q6_K", context: 131072, source: "discovered" },
  ],
  apps: [
    { id: "codex", name: "Codex", command: "codex", args: [], endpointAware: true },
    { id: "claude", name: "Claude Code", command: "claude", args: [], endpointAware: true },
    { id: "hermes", name: "Hermes", command: "hermes", args: ["chat"], endpointAware: true },
  ],
};

const nav: Array<{ id: View; label: string; icon: typeof Library }> = [
  { id: "models", label: "Models", icon: Library },
  { id: "chat", label: "Chat", icon: MessageSquare },
  { id: "hub", label: "Model hub", icon: CloudDownload },
  { id: "apps", label: "Apps", icon: AppWindow },
  { id: "sweeper", label: "Config sweeper", icon: Gauge },
];

interface ChatEntry {
  id: string;
  role: "user" | "assistant";
  content: string;
}

type UpdateState = "idle" | "checking" | "available" | "downloading" | "current" | "error";

function storedChat(): ChatEntry[] {
  try {
    const parsed = JSON.parse(globalThis.localStorage?.getItem("ese-chat-history") ?? "[]");
    return Array.isArray(parsed) ? parsed.filter((item) => item && ["user", "assistant"].includes(item.role) && typeof item.content === "string").slice(-100) : [];
  } catch {
    return [];
  }
}

function formatBytes(bytes?: number) {
  if (!bytes) return "—";
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function formatCount(value: number) {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatEta(seconds?: number) {
  if (seconds === undefined) return "Estimating…";
  if (seconds < 60) return `${seconds}s remaining`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m remaining`;
  return `${Math.ceil(seconds / 3600)}h remaining`;
}

function formatSpeed(bytesPerSecond: number) {
  return bytesPerSecond >= 1024 ** 3
    ? `${(bytesPerSecond / 1024 ** 3).toFixed(1)} GB/s`
    : `${(bytesPerSecond / 1024 ** 2).toFixed(1)} MB/s`;
}

const familyOrder = [
  "Qwen 3.8 · 27B", "Qwen 3.6 · 27B", "Qwen 3.6 · 35B", "Qwen", "Qwopus Coder",
  "Gemma 4", "Agents-A1", "AgentWorld", "Ornith", "Falcon", "Mamba", "MoE experiments", "Other",
];

function modelFamily(model: ModelProfile) {
  const name = model.name.toLowerCase();
  const compact = name.replace(/[^a-z0-9]+/g, "");
  if (compact.includes("agentworld")) return "AgentWorld";
  if (compact.includes("agentsa1")) return "Agents-A1";
  if (compact.includes("qwopus")) return "Qwopus Coder";
  if (compact.includes("qwen38")) return "Qwen 3.8 · 27B";
  if (compact.includes("qwen36") && compact.includes("27b")) return "Qwen 3.6 · 27B";
  if (compact.includes("qwen36")) return "Qwen 3.6 · 35B";
  if (compact.includes("qwen")) return "Qwen";
  if (compact.includes("gemma")) return "Gemma 4";
  if (compact.includes("ornith")) return "Ornith";
  if (compact.includes("falcon")) return "Falcon";
  if (compact.includes("mamba")) return "Mamba";
  if (compact.includes("moe")) return "MoE experiments";
  return "Other";
}

function groupModelsByFamily(models: ModelProfile[]) {
  const groups = new Map<string, ModelProfile[]>();
  for (const model of models) {
    const family = modelFamily(model);
    groups.set(family, [...(groups.get(family) ?? []), model]);
  }
  return [...groups.entries()]
    .map(([family, familyModels]) => [family, familyModels.sort((left, right) => left.name.localeCompare(right.name))] as const)
    .sort(([left], [right]) => {
      const leftIndex = familyOrder.indexOf(left);
      const rightIndex = familyOrder.indexOf(right);
      return (leftIndex < 0 ? familyOrder.length : leftIndex) - (rightIndex < 0 ? familyOrder.length : rightIndex) || left.localeCompare(right);
    });
}

function ModelRow({ model, selected, onSelect }: { model: ModelProfile; selected: boolean; onSelect: () => void }) {
  return (
    <button className={`model-row ${selected ? "selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
      <span className="model-icon"><Box size={16} /></span>
      <span className="model-primary"><strong>{model.name}</strong><small title={model.path}>{model.path}</small></span>
      <span className="model-cell"><small>Architecture</small>{model.architecture ?? "Unknown"}</span>
      <span className="model-cell"><small>Quant</small>{model.quantization ?? "—"}</span>
      <span className="model-cell"><small>Size</small>{formatBytes(model.sizeBytes)}</span>
      <span className={`availability ${model.available ? "ready" : "missing"}`}><i />{model.available ? "Ready" : "Missing"}</span>
    </button>
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }) {
  return <div className="empty-state"><Box size={22} /><strong>{title}</strong><span>{detail}</span></div>;
}

function App() {
  const [view, setView] = useState<View>("models");
  const [snapshot, setSnapshot] = useState<StudioSnapshot>(demoSnapshot);
  const [native, setNative] = useState(false);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [filterOpen, setFilterOpen] = useState(false);
  const [familyFilter, setFamilyFilter] = useState("");
  const [quantFilter, setQuantFilter] = useState("");
  const [showUnavailable, setShowUnavailable] = useState(false);
  const [expandedFamilies, setExpandedFamilies] = useState<Set<string>>(() => new Set());
  const [selectedId, setSelectedId] = useState<string>();
  const [tabs, setTabs] = useState<TerminalTab[]>([]);
  const [activeTab, setActiveTab] = useState<string>();
  const [terminalOpen, setTerminalOpen] = useState(false);
  const [terminalHeight, setTerminalHeight] = useState(() => {
    const stored = Number(globalThis.localStorage?.getItem("ese-terminal-height"));
    return Number.isFinite(stored) && stored >= 140 ? stored : 320;
  });
  const terminalHistory = useRef(new Map<string, string>());
  const [preset, setPreset] = useState("standard");
  const [advanced, setAdvanced] = useState(false);
  const [objective, setObjective] = useState("balanced");
  const [sweepPlan, setSweepPlan] = useState<SweepPlan>();
  const [sweepStatus, setSweepStatus] = useState<SweepStatus>();
  const [sweepConfirmation, setSweepConfirmation] = useState(false);
  const [profileApplied, setProfileApplied] = useState(false);
  const [maxContext, setMaxContext] = useState(131072);
  const [notice, setNotice] = useState<string>();
  const [modelRoot, setModelRoot] = useState("");
  const [showAppEditor, setShowAppEditor] = useState(false);
  const [appDraft, setAppDraft] = useState({ name: "", command: "", args: "", workingDirectory: "" });
  const [hubQuery, setHubQuery] = useState("");
  const [hubResults, setHubResults] = useState<HubModel[]>([]);
  const [hubSearching, setHubSearching] = useState(false);
  const [hubDetails, setHubDetails] = useState<HubModelDetails>();
  const [hubLoadingDetails, setHubLoadingDetails] = useState(false);
  const [hubVariantId, setHubVariantId] = useState("");
  const [downloadRoot, setDownloadRoot] = useState("");
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus>();
  const [onboardingChoice, setOnboardingChoice] = useState(false);
  const [savingConsent, setSavingConsent] = useState(false);
  const [chatStatus, setChatStatus] = useState<ChatStatus>({ active: false });
  const [chatMessages, setChatMessages] = useState<ChatEntry[]>(storedChat);
  const [chatDraft, setChatDraft] = useState("");
  const [chatRunning, setChatRunning] = useState(false);
  const [chatRequestId, setChatRequestId] = useState<string>();
  const [appVersion, setAppVersion] = useState("0.1.1");
  const [pendingUpdate, setPendingUpdate] = useState<Update>();
  const [updateState, setUpdateState] = useState<UpdateState>("idle");
  const [updateDetail, setUpdateDetail] = useState("Check GitHub Releases for a signed ESE Studio and runtime update.");
  const [updateProgress, setUpdateProgress] = useState(0);
  const chatEnd = useRef<HTMLDivElement>(null);

  const refresh = async () => {
    setLoading(true);
    try {
      const result = await invoke<StudioSnapshot>("get_studio_snapshot");
      setSnapshot(result);
      setNative(true);
      setSelectedId((value) => value ?? result.models.find((model) => model.available)?.id);
      setDownloadRoot((value) => value || result.modelRoots[0] || "");
    } catch {
      setNative(false);
      setSelectedId((value) => value ?? demoSnapshot.models[0].id);
    } finally { setLoading(false); }
  };

  useEffect(() => {
    void refresh();
    if (!("__TAURI_INTERNALS__" in window)) return;
    void getVersion().then(setAppVersion).catch(() => undefined);
    const cleanups: Array<() => void> = [];
    let disposed = false;
    const register = (listener: Promise<() => void>) => void listener.then((cleanup) => {
      if (disposed) cleanup(); else cleanups.push(cleanup);
    });
    register(listen<{ sessionId: string; data: string }>("terminal-output", ({ payload }) => {
      const previous = terminalHistory.current.get(payload.sessionId) ?? "";
      terminalHistory.current.set(payload.sessionId, (previous + payload.data).slice(-2_000_000));
    }));
    register(listen<string>("terminal-exit", ({ payload }) => {
      setTabs((items) => items.map((item) => item.sessionId === payload ? { ...item, status: "exited" } : item));
    }));
    register(listen<{ sessionId: string; name: string; command: string }>("terminal-restored", ({ payload }) => {
      const id = crypto.randomUUID();
      setTabs((items) => [...items, { id, sessionId: payload.sessionId, name: payload.name, command: payload.command, status: "running" }]);
      setActiveTab(id); setTerminalOpen(true);
    }));
    register(listen<SweepStatus>("sweep-progress", ({ payload }) => setSweepStatus(payload)));
    register(listen<DownloadStatus>("download-progress", ({ payload }) => {
      setDownloadStatus(payload);
      if (payload.state === "complete") void refresh();
    }));
    register(listen<{ requestId: string; content: string }>("chat-token", ({ payload }) => {
      setChatMessages((messages) => messages.map((message) => message.id === payload.requestId ? { ...message, content: message.content + payload.content } : message));
    }));
    register(listen<{ requestId: string; stopped: boolean; error?: string }>("chat-finished", ({ payload }) => {
      setChatMessages((messages) => messages.filter((message) => message.id !== payload.requestId || message.content.length > 0));
      setChatRunning(false);
      setChatRequestId(undefined);
      if (payload.error) setNotice(payload.error);
    }));
    void invoke<SweepStatus | null>("get_sweep_status").then((status) => { if (status) setSweepStatus(status); }).catch(() => undefined);
    void invoke<DownloadStatus | null>("get_hf_download_status").then((status) => { if (status) setDownloadStatus(status); }).catch(() => undefined);
    return () => { disposed = true; cleanups.forEach((cleanup) => cleanup()); };
  }, []);

  const checkForUpdate = async () => {
    if (!native) return;
    setUpdateState("checking");
    setUpdateDetail("Contacting the signed GitHub release channel…");
    setUpdateProgress(0);
    try {
      const availableUpdate = await check({ timeout: 30_000 });
      setPendingUpdate(availableUpdate ?? undefined);
      if (availableUpdate) {
        setUpdateState("available");
        setUpdateDetail(`Version ${availableUpdate.version} is ready to download.`);
      } else {
        setUpdateState("current");
        setUpdateDetail(`ESE Studio ${appVersion} is up to date.`);
      }
    } catch (error) {
      setUpdateState("error");
      setUpdateDetail(`Update check failed: ${String(error)} Try again when your connection is available.`);
    }
  };

  const installUpdate = async () => {
    if (!pendingUpdate) return;
    setUpdateState("downloading");
    setUpdateDetail(`Downloading ESE Studio ${pendingUpdate.version} and its matching runtime…`);
    let downloaded = 0;
    let contentLength = 0;
    try {
      await pendingUpdate.downloadAndInstall((event) => {
        if (event.event === "Started") contentLength = event.data.contentLength ?? 0;
        if (event.event === "Progress") downloaded += event.data.chunkLength;
        if (event.event === "Started" || event.event === "Progress") {
          setUpdateProgress(contentLength ? Math.min(100, downloaded / contentLength * 100) : 0);
        }
      });
      setUpdateProgress(100);
      setUpdateDetail("Update installed. Restarting ESE Studio…");
      await relaunch();
    } catch (error) {
      setUpdateState("error");
      setUpdateDetail(`Update failed: ${String(error)} Your current installation was left in place.`);
    }
  };

  useEffect(() => {
    globalThis.localStorage?.setItem("ese-terminal-height", String(terminalHeight));
  }, [terminalHeight]);

  useEffect(() => {
    globalThis.localStorage?.setItem("ese-chat-history", JSON.stringify(chatMessages.filter((message) => message.content).slice(-100)));
    chatEnd.current?.scrollIntoView({ behavior: chatRunning ? "auto" : "smooth", block: "end" });
  }, [chatMessages, chatRunning]);

  useEffect(() => {
    if (!native) return;
    const update = () => void invoke<ChatStatus>("get_chat_status").then(setChatStatus).catch(() => setChatStatus({ active: false }));
    update();
    const interval = window.setInterval(update, 2000);
    return () => window.clearInterval(interval);
  }, [native]);

  const availableModels = useMemo(() => snapshot.models.filter((model) => model.available), [snapshot.models]);
  const familyOptions = useMemo(() => familyOrder.filter((family) => availableModels.some((model) => modelFamily(model) === family)), [availableModels]);
  const quantOptions = useMemo(() => [...new Set(availableModels.map((model) => model.quantization).filter((value): value is string => Boolean(value)))].sort(), [availableModels]);
  const available = useMemo(() => availableModels.filter((model) => `${model.name} ${model.path} ${modelFamily(model)}`.toLowerCase().includes(query.toLowerCase()) && (!familyFilter || modelFamily(model) === familyFilter) && (!quantFilter || model.quantization === quantFilter)), [availableModels, query, familyFilter, quantFilter]);
  const unavailable = useMemo(() => snapshot.models.filter((model) => !model.available && `${model.name} ${model.path}`.toLowerCase().includes(query.toLowerCase())), [snapshot.models, query]);
  const groupedAvailable = useMemo(() => groupModelsByFamily(available), [available]);
  const groupedSweepModels = useMemo(() => groupModelsByFamily(availableModels), [availableModels]);
  const selected = snapshot.models.find((model) => model.id === selectedId);
  const currentTab = tabs.find((tab) => tab.id === activeTab);
  const visibleDownloadStatus = downloadStatus && (downloadStatus.state === "downloading" || downloadStatus.repoId === hubDetails?.model.id) ? downloadStatus : undefined;
  useEffect(() => {
    if (selected?.context) setMaxContext(Math.max(512, selected.context));
  }, [selectedId, selected?.context]);

  useEffect(() => {
    setSweepPlan(undefined);
    setSweepConfirmation(false);
  }, [selectedId, preset, advanced, objective, maxContext]);

  const launchApp = async (profile: AppProfile, role?: "model" | "tool") => {
    const previousActive = activeTab;
    const id = crypto.randomUUID();
    const tab: TerminalTab = { id, name: profile.name, command: profile.command, status: "starting" };
    setTabs((items) => [...items, tab]); setActiveTab(id); setTerminalOpen(true);
    if (!native) { setTabs((items) => items.filter((item) => item.id !== id)); setActiveTab(previousActive); setTerminalOpen(tabs.length > 0); setNotice("Terminal launch requires the installed ESE Studio desktop app."); return; }
    try {
      const modelEndpoint = role === "model" && selected ? {
        baseUrl: "http://127.0.0.1:8080/v1",
        apiKey: "sk-local-placeholder",
        modelId: selected.path,
        modelPath: selected.path,
        context: selected.context ?? maxContext,
        architecture: selected.architecture,
        quantization: selected.quantization,
        kvType: selected.kvType,
        batchSize: selected.batchSize,
        ubatchSize: selected.ubatchSize,
        resourcePlan: undefined,
      } : undefined;
      const sessionId = await invoke<string>("launch_terminal", { request: { command: profile.command, args: profile.args, workingDirectory: profile.workingDirectory, columns: 110, rows: 32, role, appId: profile.id, endpointAware: profile.endpointAware, modelEndpoint } });
      setTabs((items) => items.map((item) => item.id === id ? { ...item, sessionId, status: "running" } : item));
    } catch (error) {
      setTabs((items) => items.filter((item) => item.id !== id));
      setActiveTab(previousActive);
      if (tabs.length === 0) setTerminalOpen(false);
      setNotice(String(error));
    }
  };

  const closeTab = async (tab: TerminalTab) => {
    if (tab.sessionId && native) await invoke("stop_terminal", { sessionId: tab.sessionId }).catch(() => undefined);
    if (tab.sessionId) terminalHistory.current.delete(tab.sessionId);
    const remaining = tabs.filter((item) => item.id !== tab.id);
    setTabs(remaining); setActiveTab(remaining[remaining.length - 1]?.id); if (!remaining.length) setTerminalOpen(false);
  };

  const previewSweep = async () => {
    if (!selected) return;
    const request = { modelPath: selected.path, preset, objective: advanced ? objective : "max-safe-context", maxContext, safeMargin: 0.9 };
    try {
      const plan = native ? await invoke<SweepPlan>("plan_sweep", { request }) : { ...request, candidateContexts: [4096, 32768, 65536, 98304, 131072], kvTypes: ["q4_0", "q8_0", "f16"], batchSizes: [256, 512], promotedContext: 117760, trialCount: 11, requiresExclusiveGpu: true, safeMargin: 0.9 };
      setSweepPlan(plan as SweepPlan);
    } catch (error) { setNotice(String(error)); }
  };

  const runSweep = async () => {
    if (!selected || !native) return;
    const request = { modelPath: selected.path, preset, objective: advanced ? objective : "max-safe-context", maxContext, safeMargin: 0.9 };
    try {
      const status = await invoke<SweepStatus>("start_sweep", { request, confirmed: true });
      setSweepStatus(status); setSweepConfirmation(false);
    } catch (error) { setNotice(String(error)); setSweepConfirmation(false); }
  };

  const cancelSweep = async () => {
    await invoke("cancel_sweep");
    setSweepStatus((status) => status ? { ...status, state: "cancelling", currentLabel: "Cancelling after the active trial…" } : status);
  };

  const applySweepProfile = async () => {
    if (!native) return;
    try {
      const profile = await invoke<ModelProfile>("promote_sweep");
      await refresh();
      setSelectedId(profile.id);
      setProfileApplied(true);
      setNotice(`Applied verified profile for ${profile.name}. Future launches use the measured context, KV type, and batch size.`);
    } catch (error) { setNotice(String(error)); }
  };

  const addModelRoot = async () => {
    if (!modelRoot.trim() || !native) return;
    try {
      await invoke("add_model_root", { path: modelRoot.trim() });
      setModelRoot("");
      await refresh();
    } catch (error) { setNotice(String(error)); }
  };

  const addApp = async () => {
    if (!appDraft.name.trim() || !appDraft.command.trim() || !native) return;
    const id = appDraft.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || crypto.randomUUID();
    const profile: AppProfile = { id, name: appDraft.name.trim(), command: appDraft.command.trim(), args: appDraft.args.trim() ? appDraft.args.trim().split(/\s+/) : [], workingDirectory: appDraft.workingDirectory.trim() || undefined, endpointAware: true };
    try {
      await invoke("add_app_profile", { profile });
      setAppDraft({ name: "", command: "", args: "", workingDirectory: "" }); setShowAppEditor(false);
      await refresh();
    } catch (error) { setNotice(String(error)); }
  };

  const searchHub = async () => {
    if (!native || hubQuery.trim().length < 2) return;
    setHubSearching(true); setNotice(undefined);
    try {
      const results = await invoke<HubModel[]>("search_hf_models", { query: hubQuery.trim() });
      setHubResults(results); setHubDetails(undefined); setHubVariantId("");
    } catch (error) { setNotice(String(error)); }
    finally { setHubSearching(false); }
  };

  const inspectHubModel = async (model: HubModel) => {
    if (!native) return;
    setHubLoadingDetails(true); setNotice(undefined); setHubDetails(undefined); setHubVariantId("");
    try {
      const details = await invoke<HubModelDetails>("get_hf_model_details", { repoId: model.id });
      setHubDetails(details);
      setHubVariantId(details.variants.find((variant) => variant.recommended)?.id ?? details.variants[0]?.id ?? "");
    } catch (error) { setNotice(String(error)); }
    finally { setHubLoadingDetails(false); }
  };

  const startHubDownload = async () => {
    const variant = hubDetails?.variants.find((item) => item.id === hubVariantId);
    if (!hubDetails || !variant || !downloadRoot || !native) return;
    try {
      const status = await invoke<DownloadStatus>("start_hf_download", { request: { repoId: hubDetails.model.id, revision: hubDetails.revision, variantId: variant.id, quantization: variant.quantization, files: variant.files, fileSizes: variant.fileSizes, totalBytes: variant.totalSizeBytes, destinationRoot: downloadRoot } });
      setDownloadStatus(status);
    } catch (error) { setNotice(String(error)); }
  };

  const saveHelpImproveEse = async (enabled: boolean) => {
    if (!native) return;
    setSavingConsent(true);
    try {
      const result = await invoke<StudioSnapshot>("set_help_improve_ese", { enabled });
      setSnapshot(result);
      setOnboardingChoice(enabled);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setSavingConsent(false);
    }
  };

  const launchEse = (mode: "plan" | "serve") => {
    if (!selected) return;
    const args = [mode, selected.path];
    if (mode === "serve" && selected.source === "sweep") {
      if (selected.context) args.push("-c", String(selected.context));
      if (selected.kvType) args.push("--kv", selected.kvType);
      if (selected.batchSize) args.push("--batch-size", String(selected.batchSize));
      if (selected.ubatchSize) args.push("--ubatch-size", String(selected.ubatchSize));
    }
    void launchApp({ id: `ese-${mode}`, name: mode === "serve" ? `Serving · ${selected.name}` : `Plan · ${selected.name}`, command: snapshot.eseBinary ?? "ese", args, endpointAware: false }, mode === "serve" ? "model" : "tool");
  };

  const sendChat = async (text: string, history = chatMessages) => {
    const content = text.trim();
    if (!content || !native || !chatStatus.active || chatRunning) return;
    const requestId = crypto.randomUUID();
    const userMessage: ChatEntry = { id: crypto.randomUUID(), role: "user", content };
    const assistantMessage: ChatEntry = { id: requestId, role: "assistant", content: "" };
    const next = [...history, userMessage];
    setChatMessages([...next, assistantMessage]);
    setChatDraft("");
    setChatRunning(true);
    setChatRequestId(requestId);
    try {
      await invoke("start_chat", { requestId, messages: next.map(({ role, content: messageContent }) => ({ role, content: messageContent })) });
    } catch (error) {
      setChatMessages(next);
      setChatRunning(false);
      setChatRequestId(undefined);
      setNotice(String(error));
    }
  };

  const stopChat = async () => {
    if (!chatRequestId) return;
    await invoke("cancel_chat", { requestId: chatRequestId }).catch(() => undefined);
  };

  const regenerateChat = () => {
    if (chatRunning) return;
    let lastUser = -1;
    for (let index = chatMessages.length - 1; index >= 0; index -= 1) {
      if (chatMessages[index].role === "user") {
        lastUser = index;
        break;
      }
    }
    if (lastUser < 0) return;
    const prompt = chatMessages[lastUser].content;
    const history = chatMessages.slice(0, lastUser);
    setChatMessages(history);
    void sendChat(prompt, history);
  };

  const toggleFamily = (family: string) => {
    setExpandedFamilies((current) => {
      const next = new Set(current);
      if (next.has(family)) next.delete(family); else next.add(family);
      return next;
    });
  };

  const clampTerminalHeight = (height: number) => Math.round(Math.min(Math.max(140, height), Math.max(140, window.innerHeight - 170)));

  const beginTerminalResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.currentTarget.focus();
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = terminalHeight;
    const previousCursor = document.body.style.cursor;
    const previousSelection = document.body.style.userSelect;
    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";
    const move = (pointerEvent: PointerEvent) => setTerminalHeight(clampTerminalHeight(startHeight + startY - pointerEvent.clientY));
    const finish = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", finish);
      document.body.style.cursor = previousCursor;
      document.body.style.userSelect = previousSelection;
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", finish, { once: true });
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">E</span><span><strong>ESE Studio</strong><small>Expert Streaming Engine</small></span></div>
        <nav aria-label="Primary">
          {nav.map(({ id, label, icon: Icon }) => <button key={id} className={view === id ? "active" : ""} onClick={() => setView(id)}><Icon size={16} />{label}</button>)}
        </nav>
        <div className="sidebar-spacer" />
        <div className="runtime-card"><span><i /> Runtime ready</span><small>{snapshot.eseBinary ?? "ESE not found"}</small></div>
        <nav aria-label="Secondary"><button className={view === "settings" ? "active" : ""} onClick={() => setView("settings")}><Settings size={16} />Settings</button></nav>
      </aside>

      <main className={`workspace ${terminalOpen ? "with-terminal" : ""}`} style={terminalOpen ? { gridTemplateRows: `auto minmax(100px, 1fr) ${clampTerminalHeight(terminalHeight)}px` } : undefined}>
        <header className="topbar">
          <div><h1>{view === "models" ? "Models" : view === "chat" ? "Chat" : view === "hub" ? "Hugging Face models" : view === "apps" ? "Apps" : view === "sweeper" ? "Config sweeper" : "Settings"}</h1><p>{view === "models" ? "Discover, configure, and start local GGUF models." : view === "chat" ? "Talk directly to the active local model." : view === "hub" ? "Find GGUF models and choose a hardware-matched quantization." : view === "apps" ? "Launch terminal apps without leaving your workspace." : view === "sweeper" ? "Find the fastest stable configuration at your target context." : "Portable configuration and discovery."}</p></div>
          <div className="top-actions"><button className="icon-button" onClick={() => void refresh()} aria-label="Refresh" title="Refresh"><RefreshCw size={16} className={loading ? "spin" : ""} /></button>{view === "models" && <button className="primary" onClick={() => setView("settings")}><Plus size={15} />Add model folder</button>}</div>
        </header>

        <section className="content">
          {notice && <div className="notice" role="alert"><span>{notice}</span><button onClick={() => setNotice(undefined)} aria-label="Dismiss"><X size={14} /></button></div>}

          {view === "chat" && <div className="chat-layout">
            <div className="chat-statusbar"><span className={`chat-model-state ${chatStatus.active ? "ready" : "offline"}`}><i />{chatStatus.active ? "Model connected" : "No active model"}</span><code title={chatStatus.modelId}>{chatStatus.modelId ?? "Start a model from Models to begin"}</code><div><button onClick={regenerateChat} disabled={chatRunning || !chatMessages.some((message) => message.role === "user")} aria-label="Regenerate last response" title="Regenerate last response"><RotateCcw size={14} />Regenerate</button><button onClick={() => setChatMessages([])} disabled={chatRunning || chatMessages.length === 0} aria-label="Clear conversation" title="Clear conversation"><Trash2 size={14} />Clear</button></div></div>
            <div className="chat-transcript" role="log" aria-live="polite" aria-label="Conversation">
              {chatMessages.length ? chatMessages.map((message) => <article className={`chat-message ${message.role}`} key={message.id}><span className="chat-avatar" aria-hidden="true">{message.role === "user" ? <User size={16} /> : <Bot size={16} />}</span><div><strong>{message.role === "user" ? "You" : "ESE"}</strong><p>{message.content || <span className="typing-indicator"><i /><i /><i /><span className="sr-only">Generating response</span></span>}</p></div></article>) : <div className="chat-empty"><span><MessageSquare size={22} /></span><h2>Chat with your local model</h2><p>{chatStatus.active ? "Messages stay on this computer and are sent only to the active ESE server." : "Start a model from the Models screen. Chat will connect automatically when it is ready."}</p>{!chatStatus.active && <button className="primary" onClick={() => setView("models")}><Play size={14} />Choose a model</button>}</div>}
              <div ref={chatEnd} />
            </div>
            <form className="chat-composer" onSubmit={(event) => { event.preventDefault(); void sendChat(chatDraft); }}><textarea value={chatDraft} onChange={(event) => setChatDraft(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void sendChat(chatDraft); } }} placeholder={chatStatus.active ? "Message your local model" : "Start a model to enable chat"} aria-label="Chat message" rows={2} disabled={!chatStatus.active || chatRunning} /><div><small>Enter to send · Shift+Enter for a new line</small>{chatRunning ? <button type="button" className="chat-stop" onClick={() => void stopChat()}><StopCircle size={15} />Stop</button> : <button type="submit" className="primary" disabled={!chatStatus.active || !chatDraft.trim()}><Send size={15} />Send</button>}</div></form>
          </div>}

          {view === "hub" && <div className="hub-layout">
            <section className="panel hub-search-panel">
              <div className="panel-heading"><div><h2>Browse GGUF models</h2><p>Search Hugging Face repositories tagged GGUF.</p></div></div>
              <form className="hub-search" onSubmit={(event) => { event.preventDefault(); void searchHub(); }}><label className="search"><Search size={15} /><input value={hubQuery} onChange={(event) => setHubQuery(event.target.value)} placeholder="Model, author, or repository" aria-label="Search Hugging Face models" /></label><button className="primary" type="submit" disabled={!native || hubSearching || hubQuery.trim().length < 2}>{hubSearching ? <RefreshCw className="spin" size={14} /> : <Search size={14} />}{hubSearching ? "Searching" : "Search"}</button></form>
              <div className="hub-results" aria-live="polite">{hubSearching ? <EmptyState title="Searching Hugging Face" detail="Loading popular matching GGUF repositories." /> : hubResults.length ? hubResults.map((model) => <button key={model.id} className={`hub-result ${hubDetails?.model.id === model.id ? "selected" : ""}`} onClick={() => void inspectHubModel(model)}><span className="hub-result-icon"><CloudDownload size={16} /></span><span><strong>{model.id}</strong><small><Download size={11} />{formatCount(model.downloads)} downloads <Heart size={11} />{formatCount(model.likes)}</small></span><ChevronRight size={15} /></button>) : <EmptyState title="Find a model" detail="Try a family such as Qwen, Llama, Gemma, or Mixtral." />}</div>
            </section>
            <section className="panel hub-details-panel">
              {hubLoadingDetails ? <EmptyState title="Inspecting repository" detail="Reading GGUF files and matching memory use to this computer." /> : hubDetails ? <><div className="panel-heading hub-title"><div><h2>{hubDetails.model.id}</h2><p>Select a quantization. ESE recommends the highest-quality option within a conservative memory budget.</p></div><span className="revision" title={hubDetails.revision}>Pinned revision</span></div>
                <div className="hardware-card"><HardDrive size={18} /><div><strong>{hubDetails.hardware.gpuNames.length ? hubDetails.hardware.gpuNames.join(" + ") : "CPU / system memory"}</strong><span>{formatBytes(hubDetails.hardware.vramTotalBytes)} VRAM · {formatBytes(hubDetails.hardware.ramAvailableBytes)} RAM available</span></div></div>
                <fieldset className="quant-list"><legend>Quantizations</legend>{hubDetails.variants.map((variant) => <label key={variant.id} className={`${hubVariantId === variant.id ? "selected" : ""} ${variant.recommended ? "recommended" : ""}`}><input type="radio" name="hub-variant" checked={hubVariantId === variant.id} onChange={() => setHubVariantId(variant.id)} /><span className="quant-main"><strong>{variant.quantization}{variant.recommended && <em>Recommended</em>}</strong><small>{variant.files.length} {variant.files.length === 1 ? "file" : "shards"} · {formatBytes(variant.totalSizeBytes)}</small></span><span className={`fit fit-${variant.fit.toLowerCase().replace(/[^a-z]+/g, "-")}`}><strong>{variant.fit}</strong><small>{variant.fitReason}</small></span></label>)}</fieldset>
                <label className="field"><span>Download to model folder</span><select value={downloadRoot} onChange={(event) => setDownloadRoot(event.target.value)}>{snapshot.modelRoots.map((root) => <option key={root} value={root}>{root}</option>)}</select></label>
                {visibleDownloadStatus && <div className={`download-card ${visibleDownloadStatus.state}`}><div><strong>{visibleDownloadStatus.state === "downloading" ? "Downloading model" : visibleDownloadStatus.state === "complete" ? "Download complete" : visibleDownloadStatus.state === "cancelled" ? "Download paused" : "Download failed"}</strong><span>{visibleDownloadStatus.currentFile ?? visibleDownloadStatus.repoId}</span></div><div className="progress-track" role="progressbar" aria-label="Model download" aria-valuemin={0} aria-valuemax={visibleDownloadStatus.totalBytes} aria-valuenow={Math.min(visibleDownloadStatus.completedBytes, visibleDownloadStatus.totalBytes)}><span style={{ width: `${visibleDownloadStatus.totalBytes ? Math.min(100, visibleDownloadStatus.completedBytes / visibleDownloadStatus.totalBytes * 100) : 0}%` }} /></div><div className="progress-meta"><span>{formatBytes(visibleDownloadStatus.completedBytes)} of {formatBytes(visibleDownloadStatus.totalBytes)}</span><span>{visibleDownloadStatus.state === "downloading" && visibleDownloadStatus.bytesPerSecond ? `${formatSpeed(visibleDownloadStatus.bytesPerSecond)} · ${formatEta(visibleDownloadStatus.etaSeconds)}` : visibleDownloadStatus.state}</span></div>{visibleDownloadStatus.error && <small className="download-error">{visibleDownloadStatus.error}</small>}</div>}
                <div className="hub-actions"><small>Downloads resume from partial files after cancellation.</small>{visibleDownloadStatus?.state === "downloading" ? <button onClick={() => void invoke("cancel_hf_download")}>Cancel safely</button> : <button className="primary" onClick={() => void startHubDownload()} disabled={!downloadRoot || !hubVariantId}><CloudDownload size={14} />Download selected quant</button>}</div></> : <EmptyState title="Choose a repository" detail="Hardware recommendations appear here after Studio inspects its GGUF files." />}
            </section>
          </div>}

          {view === "models" && <div className="models-view">
            <div className="model-controls"><div className="toolbar"><label className="search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search models" /></label><button className={familyFilter || quantFilter ? "active-filter" : ""} onClick={() => setFilterOpen((open) => !open)} aria-expanded={filterOpen} aria-controls="model-filters"><SlidersHorizontal size={15} />Filter{(familyFilter || quantFilter) && <span className="filter-active-count">{Number(Boolean(familyFilter)) + Number(Boolean(quantFilter))}</span>}</button><span className="toolbar-count">{available.length === availableModels.length ? `${available.length} available` : `${available.length} of ${availableModels.length}`}</span></div>{filterOpen && <div className="filter-panel" id="model-filters"><label><span>Family</span><select value={familyFilter} onChange={(event) => setFamilyFilter(event.target.value)}><option value="">All families</option>{familyOptions.map((family) => <option key={family} value={family}>{family}</option>)}</select></label><label><span>Quantization</span><select value={quantFilter} onChange={(event) => setQuantFilter(event.target.value)}><option value="">All quantizations</option>{quantOptions.map((quantization) => <option key={quantization} value={quantization}>{quantization}</option>)}</select></label><button onClick={() => { setFamilyFilter(""); setQuantFilter(""); }} disabled={!familyFilter && !quantFilter}>Clear filters</button></div>}</div>
            <div className="model-browser" tabIndex={0} aria-label="Available models by family">
              {groupedAvailable.length ? <div className="model-groups">{groupedAvailable.map(([family, models]) => {
                const expanded = query.trim().length > 0 || expandedFamilies.has(family);
                return <section className="model-group" key={family}><button className="family-heading" onClick={() => toggleFamily(family)} aria-expanded={expanded}>{expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}<strong>{family}</strong><span>{models.length}</span></button>{expanded && <div className="model-list">{models.map((model) => <ModelRow key={model.id} model={model} selected={model.id === selectedId} onSelect={() => setSelectedId(model.id)} />)}</div>}</section>;
              })}</div> : <EmptyState title="No models found" detail="Add a model folder in Settings to begin discovery." />}
              {!!unavailable.length && <div className="unavailable"><button onClick={() => setShowUnavailable(!showUnavailable)}>{showUnavailable ? <ChevronDown size={15} /> : <ChevronRight size={15} />}Unavailable profiles <span>{unavailable.length}</span></button>{showUnavailable && <div className="model-list">{unavailable.map((model) => <ModelRow key={model.id} model={model} selected={model.id === selectedId} onSelect={() => setSelectedId(model.id)} />)}</div>}</div>}
            </div>
            {selected && <div className="selection-bar"><div><span className="availability ready"><i />Selected</span><strong>{selected.name}</strong><small>{selected.context ? `${selected.context.toLocaleString()} max context` : "Context will be planned by ESE"}</small></div><button onClick={() => launchEse("plan")}>Inspect plan</button><button className="primary" onClick={() => launchEse("serve")}><Play size={14} fill="currentColor" />Start model</button></div>}
          </div>}

          {view === "apps" && <><div className="apps-grid">{snapshot.apps.map((app) => <article className="app-card" key={app.id}><span className="app-icon"><SquareTerminal size={20} /></span><div><h2>{app.name}</h2><code>{app.command} {app.args.join(" ")}</code><p>{app.endpointAware ? "Receives the active model endpoint, identity, context, and tuning details in its terminal session." : "Runs in a persistent terminal tab. The session stays alive when models change."}</p></div><button className="primary" onClick={() => void launchApp(app)}><Play size={14} fill="currentColor" />Launch</button></article>)}<button className="add-card" onClick={() => setShowAppEditor(true)}><Plus size={18} />Add a custom app<span>Command, arguments, and optional working directory</span></button></div>{showAppEditor && <section className="panel inline-editor"><div className="panel-heading"><div><h2>Custom terminal app</h2><p>The command runs directly in a native PTY. Arguments are stored in portable TOML.</p></div><button className="icon-button" onClick={() => setShowAppEditor(false)} aria-label="Close"><X size={14} /></button></div><div className="editor-grid"><label className="field"><span>Name</span><input value={appDraft.name} onChange={(event) => setAppDraft({ ...appDraft, name: event.target.value })} placeholder="My coding agent" /></label><label className="field"><span>Command</span><input value={appDraft.command} onChange={(event) => setAppDraft({ ...appDraft, command: event.target.value })} placeholder="agent-command" /></label><label className="field"><span>Arguments</span><input value={appDraft.args} onChange={(event) => setAppDraft({ ...appDraft, args: event.target.value })} placeholder="--optional flags" /></label><label className="field"><span>Working directory</span><input value={appDraft.workingDirectory} onChange={(event) => setAppDraft({ ...appDraft, workingDirectory: event.target.value })} placeholder="/home/me/project" /></label></div><button className="primary" onClick={() => void addApp()} disabled={!native || !appDraft.name.trim() || !appDraft.command.trim()}>Save app</button></section>}</>}

          {view === "sweeper" && <div className="sweep-layout">
            <section className="panel">
              <div className="panel-heading"><div><h2>Sweep setup</h2><p>Verify maximum stable context, then tune speed at the promoted safe context.</p></div><label className="toggle"><input type="checkbox" checked={advanced} onChange={(event) => setAdvanced(event.target.checked)} /><span />Advanced</label></div>
              <label className="field"><span>Model</span><select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>{groupedSweepModels.map(([family, models]) => <optgroup key={family} label={family}>{models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</optgroup>)}</select></label>
              <fieldset className="preset-grid"><legend>Depth</legend>{[["quick", "Quick", "Capacity + 4 tuning trials"], ["standard", "Standard", "Repeated tuning"], ["exhaustive", "Exhaustive", "Broad overnight search"]].map(([id, label, time]) => <label key={id} className={preset === id ? "selected" : ""}><input type="radio" name="preset" value={id} checked={preset === id} onChange={() => setPreset(id)} /><strong>{label}</strong><span>{time}</span></label>)}</fieldset>
              <div className="advanced-fields">
                <label className="field"><span>Maximum context to verify</span><input type="number" min="512" step="256" value={maxContext} onChange={(event) => setMaxContext(Math.max(512, Number(event.target.value) || 512))} /></label>
                <label className="field"><span>Promotion margin</span><input value="90% of verified maximum" readOnly /></label>
              </div>
              {advanced && <div className="advanced-fields"><label className="field"><span>Objective</span><select value={objective} onChange={(event) => setObjective(event.target.value)}><option value="balanced">Balanced at maximum context</option><option value="max-throughput">Maximum throughput</option><option value="minimum-vram">Minimum VRAM</option><option value="lowest-latency">Lowest latency</option></select></label><label className="field"><span>Checkpointing</span><input value="Every completed trial" readOnly /></label></div>}
              <div className="exclusive-note"><Activity size={16} /><span><strong>Exclusive GPU access required</strong>Studio stops and later restores its active model. Unmanaged llama-server instances or CUDA apps using significant VRAM cause a safe refusal.</span></div>
              <button className="primary wide" onClick={() => void previewSweep()} disabled={!selected || sweepStatus?.state === "running"}><Gauge size={15} />Preview sweep</button>
            </section>
            <section className="panel results">
              <div className="panel-heading"><div><h2>{sweepStatus && ["running", "preparing", "cancelling"].includes(sweepStatus.state) ? "Sweep running" : "Sweep plan"}</h2><p>{sweepStatus?.currentLabel ?? "Preview does not reserve or use a GPU."}</p></div></div>
              {sweepStatus && ["running", "preparing", "cancelling"].includes(sweepStatus.state) ? <>
                <div className="progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={sweepStatus.totalTrials} aria-valuenow={sweepStatus.completedTrials}><span style={{ width: `${Math.max(3, sweepStatus.completedTrials / sweepStatus.totalTrials * 100)}%` }} /></div>
                <div className="progress-meta"><span>{sweepStatus.completedTrials} / {sweepStatus.totalTrials} trials</span><span>{sweepStatus.phase}</span></div>
                <button className="wide" onClick={() => void cancelSweep()} disabled={sweepStatus.state === "cancelling"}>Cancel safely</button>
              </> : sweepStatus && ["complete", "failed", "cancelled"].includes(sweepStatus.state) ? <>
                <div className={`sweep-state ${sweepStatus.state}`}><strong>{sweepStatus.state === "complete" ? "Verified sweep complete" : sweepStatus.state === "failed" ? "Sweep failed" : "Sweep cancelled"}</strong><span>{sweepStatus.error ?? sweepStatus.currentLabel}</span></div>
                {sweepStatus.verifiedMaxContext && <div className="result-hero"><small>Promoted safe context</small><strong>{sweepStatus.promotedContext?.toLocaleString()}</strong><span>verified maximum {sweepStatus.verifiedMaxContext.toLocaleString()} tokens</span></div>}
                <dl><div><dt>Best stable speed</dt><dd>{sweepStatus.bestTokensPerSecond ? `${sweepStatus.bestTokensPerSecond.toFixed(2)} tok/s` : "—"}</dd></div><div><dt>KV / batch</dt><dd>{sweepStatus.bestKvType ? `${sweepStatus.bestKvType} / ${sweepStatus.bestBatchSize}` : "—"}</dd></div><div><dt>Checkpoint</dt><dd title={sweepStatus.checkpointPath}>Saved</dd></div></dl>
                {sweepStatus.state === "complete" && <button className="primary wide" onClick={() => void applySweepProfile()} disabled={profileApplied}>{profileApplied ? "Verified profile applied" : "Apply verified profile"}</button>}
                <button className="wide result-secondary" onClick={() => { setSweepStatus(undefined); setSweepPlan(undefined); setProfileApplied(false); }}>New sweep</button>
              </> : sweepPlan ? <>
                <div className="result-hero"><small>Candidate promoted context</small><strong>{sweepPlan.promotedContext.toLocaleString()}</strong><span>final value follows measured capacity</span></div>
                <dl><div><dt>Maximum trials</dt><dd>{sweepPlan.trialCount}</dd></div><div><dt>KV types</dt><dd>{sweepPlan.kvTypes.join(", ")}</dd></div><div><dt>Batch sizes</dt><dd>{sweepPlan.batchSizes.join(", ")}</dd></div><div><dt>Checkpointing</dt><dd>Every trial</dd></div></dl>
                {!sweepConfirmation ? <button className="primary wide" onClick={() => setSweepConfirmation(true)} disabled={!native}>Run verified sweep</button> : <div className="sweep-confirm"><strong>Give the sweep exclusive GPU access?</strong><span>Studio-managed serving stops now and is restored after completion, failure, or cancellation.</span><div><button onClick={() => setSweepConfirmation(false)}>Not now</button><button className="primary" onClick={() => void runSweep()}>Stop model & run</button></div></div>}
              </> : <EmptyState title="No plan yet" detail="Choose a model and preview the measured search space." />}
            </section>
          </div>}

          {view === "settings" && <div className="settings-layout">
            <section className="panel settings-models">
              <div className="panel-heading"><div><h2>Model discovery</h2><p>Folders are scanned recursively for GGUF files.</p></div></div>
              <div className="path-entry"><input value={modelRoot} onChange={(event) => setModelRoot(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void addModelRoot(); }} placeholder={snapshot.platform === "windows" ? "C:\\Models" : "/path/to/models"} aria-label="Model folder path" /><button className="primary" onClick={() => void addModelRoot()} disabled={!native || !modelRoot.trim()}><Plus size={14} />Add folder</button></div>
              {snapshot.modelRoots.length ? <div className="settings-paths">{snapshot.modelRoots.map((root) => <div className="setting-row" key={root}><Library size={16} /><code title={root}>{root}</code><span className="availability ready"><i />Scanning</span></div>)}</div> : <EmptyState title="No model folders" detail="Add a folder to make local models discoverable." />}
            </section>
            <div className="settings-side">
              <section className="panel community-setting">
                <div className="panel-heading"><div><h2>Help improve ESE</h2><p>Share sanitized results automatically after verified sweeps.</p></div><label className="toggle telemetry-toggle"><input type="checkbox" checked={snapshot.helpImproveEse} disabled={!native || savingConsent} onChange={(event) => void saveHelpImproveEse(event.target.checked)} aria-describedby="telemetry-settings-detail" /><span /></label></div>
                <div className="privacy-note" id="telemetry-settings-detail"><ShieldCheck size={17} /><span><strong>{snapshot.helpImproveEse ? "Community sharing is on" : "Community sharing is off"}</strong>Raw results stay in the private collector. Public GitHub data is grouped and never includes prompts, responses, usernames, hostnames, local paths, or logs.</span></div>
              </section>
              <section className="panel settings-updates">
                <div className="panel-heading"><div><h2>Updates</h2><p>Signed updates replace Studio and the bundled ESE runtime together.</p></div><span className="version-badge">v{appVersion}</span></div>
                <div className={`update-state ${updateState}`} aria-live="polite"><Sparkles size={17} /><span><strong>{updateState === "available" ? `ESE Studio ${pendingUpdate?.version}` : updateState === "downloading" ? "Installing update" : updateState === "current" ? "You’re up to date" : updateState === "error" ? "Couldn’t update" : updateState === "checking" ? "Checking for updates" : "Stable release channel"}</strong>{updateDetail}</span></div>
                {updateState === "downloading" && <><div className="progress-track" role="progressbar" aria-label="Update download" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(updateProgress)}><span style={{ width: `${Math.max(3, updateProgress)}%` }} /></div><div className="progress-meta"><span>Downloading and verifying</span><span>{updateProgress ? `${Math.round(updateProgress)}%` : "Preparing…"}</span></div></>}
                <button className={updateState === "available" ? "primary wide update-action" : "wide update-action"} onClick={() => void (updateState === "available" ? installUpdate() : checkForUpdate())} disabled={!native || updateState === "checking" || updateState === "downloading"}>{updateState === "available" ? <><Download size={14} />Download, install & restart</> : <><RefreshCw size={14} className={updateState === "checking" ? "spin" : ""} />{updateState === "checking" ? "Checking…" : "Check for updates"}</>}</button>
              </section>
              <section className="panel settings-about">
                <div className="panel-heading"><div><h2>About ESE Studio</h2><p>{snapshot.platform === "windows" ? "Windows" : snapshot.platform === "linux" ? "Linux" : snapshot.platform} control center for Expert Streaming Engine · v{appVersion}</p></div></div>
                <div className="config-path"><small>Configuration file</small><code title={snapshot.configPath}>{snapshot.configPath}</code></div>
                <p>Model and app settings use portable TOML. Keep secrets in environment variables or your operating system's credential manager. The visible path makes the configuration easy to inspect, back up, or edit with your preferred tool.</p>
              </section>
            </div>
          </div>}
        </section>

        {terminalOpen && <section className="terminal-drawer"><div className="terminal-resize-handle" role="separator" aria-label="Resize terminal" aria-orientation="horizontal" aria-valuemin={140} aria-valuemax={Math.max(140, window.innerHeight - 170)} aria-valuenow={clampTerminalHeight(terminalHeight)} tabIndex={0} onPointerDown={beginTerminalResize} onDoubleClick={() => setTerminalHeight(320)} onKeyDown={(event) => { if (event.key === "ArrowUp") { event.preventDefault(); setTerminalHeight((height) => clampTerminalHeight(height + 24)); } else if (event.key === "ArrowDown") { event.preventDefault(); setTerminalHeight((height) => clampTerminalHeight(height - 24)); } }}><span /></div><div className="terminal-tabs">{tabs.map((tab) => <div className={`terminal-tab ${activeTab === tab.id ? "active" : ""}`} key={tab.id}><button className="terminal-tab-select" onClick={() => setActiveTab(tab.id)} aria-label={`${tab.name}, ${tab.status}`}><SquareTerminal size={13} /><span>{tab.name}</span><i className={tab.status} /></button><button className="terminal-tab-close" onClick={() => void closeTab(tab)} aria-label={`Close ${tab.name}`} title={`Close ${tab.name}`}><X size={12} /></button></div>)}<button className="terminal-collapse" onClick={() => setTerminalOpen(false)} aria-label="Collapse terminal" title="Collapse terminal"><ChevronDown size={15} /></button></div>{currentTab && <TerminalPane tab={currentTab} history={currentTab.sessionId ? terminalHistory.current.get(currentTab.sessionId) : ""} />}</section>}
        {!terminalOpen && tabs.length > 0 && <button className="terminal-restore" onClick={() => setTerminalOpen(true)} aria-label="Show terminal"><SquareTerminal size={14} /><span>{currentTab?.name ?? "Terminal"}</span><i>{tabs.length}</i><ChevronUp size={14} /></button>}
      </main>
      {native && !loading && !snapshot.onboardingComplete && <div className="consent-backdrop" role="presentation"><section className="consent-dialog" role="dialog" aria-modal="true" aria-labelledby="consent-title" aria-describedby="consent-detail"><span className="consent-icon"><Heart size={20} /></span><div><small>One optional choice</small><h2 id="consent-title">Help improve ESE</h2><p id="consent-detail">Allow ESE Studio to automatically share sanitized benchmark summaries after verified sweeps. Raw submissions remain private; only organized aggregate results are published on GitHub.</p></div><label className="consent-choice"><span><strong>Share verified sweep data</strong><small>No prompts, responses, usernames, hostnames, paths, or raw logs.</small></span><span className="toggle telemetry-toggle"><input type="checkbox" checked={onboardingChoice} onChange={(event) => setOnboardingChoice(event.target.checked)} /><span /></span></label><button className="primary wide consent-continue" onClick={() => void saveHelpImproveEse(onboardingChoice)} disabled={savingConsent}>{savingConsent ? "Saving…" : "Continue"}</button><small className="consent-footnote">Off by default. You can change this anytime in Settings.</small></section></div>}
    </div>
  );
}

export default App;
