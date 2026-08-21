import { useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { FitAddon } from "@xterm/addon-fit";
import { Terminal } from "@xterm/xterm";
import "@xterm/xterm/css/xterm.css";
import type { TerminalTab } from "../types";

interface TerminalOutput { sessionId: string; data: string }

export function TerminalPane({ tab, history = "" }: { tab: TerminalTab; history?: string }) {
  const host = useRef<HTMLDivElement>(null);
  const terminal = useRef<Terminal | null>(null);

  useEffect(() => {
    if (!host.current || !tab.sessionId) return;
    const term = new Terminal({
      cursorBlink: true,
      cursorStyle: "bar",
      fontFamily: "'JetBrains Mono', 'SFMono-Regular', Consolas, monospace",
      fontSize: 13,
      lineHeight: 1.3,
      scrollback: 10_000,
      theme: { background: "#151618", foreground: "#d9d9da", cursor: "#d9d9da", selectionBackground: "#3b3c40" },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(host.current);
    if (history) term.write(history);
    fit.fit();
    terminal.current = term;
    term.focus();

    const dataDisposable = term.onData((data) => {
      void invoke("write_terminal", { sessionId: tab.sessionId, data });
    });
    const resize = () => {
      fit.fit();
      void invoke("resize_terminal", { sessionId: tab.sessionId, columns: term.cols, rows: term.rows });
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host.current);
    let disposed = false;
    let unlisten = () => {};
    void listen<TerminalOutput>("terminal-output", ({ payload }) => {
      if (payload.sessionId === tab.sessionId) term.write(payload.data);
    }).then((cleanup) => { if (disposed) cleanup(); else unlisten = cleanup; });

    return () => {
      disposed = true;
      unlisten();
      observer.disconnect();
      dataDisposable.dispose();
      term.dispose();
      terminal.current = null;
    };
  }, [tab.sessionId]);

  if (!tab.sessionId) return <div className="terminal-placeholder">Starting {tab.command}…</div>;
  return <div className="terminal-host" ref={host} aria-label={`${tab.name} terminal`} />;
}
