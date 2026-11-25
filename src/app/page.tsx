"use client";

import { useState, useEffect } from "react";
import {
  Bot,
  FolderOpen,
  LayoutDashboard,
  Settings,
  Plus,
  Send,
  Paperclip,
  Code2,
  Sparkles,
  User,
  RefreshCw,
} from "lucide-react";

type ShopifyToolkitStatus = {
  agentProfile?: string;
  skillVersion?: string;
  topics?: string[];
  references?: string[];
  mcp?: {
    enabled: boolean;
    connected: boolean;
    toolsDiscovered: number;
    toolsExposed: number;
    callsMade: number;
    lastError?: string;
    reason?: string;
  };
};

type ThemeEditMessage = {
  applied: boolean;
  summary?: string;
  files: Array<{ path: string; operations?: number; reasons?: string[] }>;
  previewNote?: string;
};

type LocalFigDesignInfo = {
  fileName: string;
  filePath: string;
  formatVersion: number | null;
  nodeCount: number;
  pageCount: number;
  pageNames: string[];
  frameCount: number;
  imageCount: number;
  frames: Array<{ id: string; name: string; page: string; width: number | null; height: number | null }>;
  images: Array<{ name: string; mimeType: string; bytes: number; sha256: string }>;
};

type LocalFigDesignStatus = {
  available: boolean;
  design?: LocalFigDesignInfo;
  error?: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  shopifyToolkit?: ShopifyToolkitStatus;
  themeEdit?: ThemeEditMessage;
  localFigDesign?: { fileName: string; pageNames: string[]; nodeCount: number; frameCount: number; imageCount: number } | null;
};

const STORAGE_KEY = "horizon-ai-chat-history";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [localFigStatus, setLocalFigStatus] = useState<LocalFigDesignStatus | null>(null);
  const [localFigLoading, setLocalFigLoading] = useState(false);

  async function refreshLocalFigStatus() {
    setLocalFigLoading(true);
    try {
      const response = await fetch("/api/figma/local", { cache: "no-store" });
      const data = await response.json();
      if (!response.ok || !data.success) {
        setLocalFigStatus({ available: false, error: data.error || "The local .fig file could not be read." });
        return;
      }
      setLocalFigStatus({ available: true, design: data.design });
    } catch (error) {
      setLocalFigStatus({
        available: false,
        error: error instanceof Error ? error.message : "Could not check the local .fig file.",
      });
    } finally {
      setLocalFigLoading(false);
    }
  }

  useEffect(() => {
    void refreshLocalFigStatus();
  }, []);

  // Load chat history
  useEffect(() => {
    try {
      const savedMessages = localStorage.getItem("test");

      if (savedMessages) {
       setMessages(JSON.parse(savedMessages));
      }
    } catch (error) {
      console.error("Failed to load chat history:", error);
    } finally {
      setHydrated(true);
    }
  }, []);

  // Save chat history
  useEffect(() => {
    if (!hydrated) return;

    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(messages)
      );
    } catch (error) {
      console.error("Failed to save chat history:", error);
    }
  }, [messages, hydrated]);

  // Send message to Agent API
  async function sendMessage() {
    if (!prompt.trim() || loading || !hydrated) return;

    const userMessage: Message = {
      role: "user",
      content: prompt.trim(),
    };

    const updatedMessages = [
      ...messages,
      userMessage,
    ];

    setMessages(updatedMessages);
    setPrompt("");
    setLoading(true);

    try {
      const response = await fetch("/api/agent", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: userMessage.content,
          history: messages,
        }),
      });

      const data = await response.json();

      // Debug API response
      console.log("API Response:", data);

      if (!response.ok) {
        throw new Error(
          data.message || "AI request failed"
        );
      }

      if (data.localFigDesign) {
        void refreshLocalFigStatus();
      }

      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: data.message || "No response received.",
          shopifyToolkit: data.shopifyToolkit,
          themeEdit: data.themeEdit,
          localFigDesign: data.localFigDesign ?? null,
        },
      ]);

    } catch (error) {
      console.error("API Error:", error);

      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content:
            error instanceof Error
              ? error.message
              : "Something went wrong.",
        },
      ]);

    } finally {
      setLoading(false);
    }
  }

  // Start new chat
  function startNewProject() {
    setMessages([]);
    setPrompt("");
  }

  return (
    <main className="flex min-h-screen bg-[#0c0d18] text-white">

      {/* SIDEBAR */}

      <aside className="w-64 border-r border-white/10 p-5">

        <h1 className="text-2xl font-bold text-violet-400">
          Horizon AI
        </h1>

        <p className="mb-10 mt-1 text-xs text-gray-400">
          Shopify Theme Developer
        </p>

        <button
          onClick={startNewProject}
          className="mb-8 flex w-full items-center gap-3 rounded-xl bg-violet-600 p-3"
        >
          <Plus size={18} />
          New Project
        </button>

        <nav className="space-y-6 text-sm text-gray-300">

          <div className="flex gap-3">
            <LayoutDashboard />
            Dashboard
          </div>

          <div className="flex gap-3">
            <FolderOpen />
            My Projects
          </div>

          <button
            type="button"
            onClick={() => document.getElementById("figma-workspace")?.scrollIntoView({ behavior: "smooth", block: "start" })}
            className="flex w-full items-center gap-3 text-left hover:text-white"
          >
            <Code2 />
            Local .fig Design
          </button>

          <div className="flex gap-3">
            <Code2 />
            Code Editor
          </div>

          <div className="flex gap-3">
            <Settings />
            Settings
          </div>

        </nav>

      </aside>

      {/* MAIN CONTENT */}

      <section className="flex flex-1 flex-col">

        {/* HEADER */}

        <header className="flex h-20 items-center justify-between border-b border-white/10 px-8">

          <div>
            <h2 className="text-xl font-semibold">
              Horizon AI Developer
            </h2>

            <p className="text-xs text-gray-400">
              Build Shopify themes with AI
            </p>
          </div>

          <span className="rounded-full bg-green-500/10 px-4 py-2 text-sm text-green-400">
            ● System Online
          </span>

        </header>



        {/* CHAT + PREVIEW */}

        <div className="grid min-h-0 flex-1 grid-cols-1 gap-6 p-6 xl:grid-cols-2">

          {/* AI CHAT */}

          <div className="flex min-h-[620px] min-w-0 flex-col rounded-2xl border border-white/10 bg-[#141525] p-6">

            <div className="mb-6 flex items-center gap-3">

              <Bot
                className="text-violet-400"
                size={28}
              />

              <div>
                <h3 className="font-semibold">
                  AI Assistant
                </h3>

                <p className="text-xs text-gray-400">
                  Your Horizon Theme Developer
                </p>
              </div>

            </div>

            {/* MESSAGES */}

            <div className="flex-1 space-y-4 overflow-y-auto rounded-xl bg-[#0c0d18] p-5">

              {!hydrated && (
                <p className="text-sm text-gray-400">
                  Loading chat history...
                </p>
              )}

              {hydrated && messages.length === 0 && (
                <div>

                  <Sparkles className="mb-3 text-violet-400" />

                  <h3 className="mb-2 text-lg font-semibold">
                    What should we build today?
                  </h3>

                  <p className="text-sm text-gray-400">
                    Describe your Shopify frontend requirements.
                  </p>

                </div>
              )}

              {messages.map((message, index) => (

                <div
                  key={index}
                  className={`rounded-xl p-4 ${
                    message.role === "user"
                      ? "ml-8 bg-violet-600/20"
                      : "mr-4 bg-white/5"
                  }`}
                >

                  <div className="mb-2 flex items-center gap-2 text-xs text-violet-300">

                    {message.role === "user" ? (
                      <User size={15} />
                    ) : (
                      <Bot size={15} />
                    )}

                    {message.role === "user"
                      ? "You"
                      : "Horizon AI"}

                  </div>

                  <p className="whitespace-pre-wrap text-sm leading-6">
                    {message.content}
                  </p>

                  {message.localFigDesign && (
                    <div className="mt-3 rounded-lg border border-violet-400/20 bg-violet-500/5 px-3 py-2 text-xs text-violet-200">
                      Local .fig source used: {message.localFigDesign.fileName} · {message.localFigDesign.pageNames.join(", ")} · {message.localFigDesign.nodeCount} nodes · {message.localFigDesign.imageCount} embedded image assets
                    </div>
                  )}

                  {message.themeEdit?.applied && (
                    <div className="mt-3 rounded-lg border border-violet-400/25 bg-violet-500/10 p-3 text-xs text-violet-100">
                      <p className="font-semibold text-violet-200">Theme files updated</p>
                      <ul className="mt-2 space-y-1">
                        {message.themeEdit.files.map((file) => (
                          <li key={file.path} className="break-all">
                            <span className="font-mono">{file.path}</span>
                            {typeof file.operations === "number" && <span className="ml-2 text-violet-300">({file.operations} change{file.operations === 1 ? "" : "s"})</span>}
                          </li>
                        ))}
                      </ul>
                      {message.themeEdit.previewNote && (
                        <p className="mt-2 leading-5 text-violet-200/80">{message.themeEdit.previewNote}</p>
                      )}
                    </div>
                  )}

                  {message.shopifyToolkit && (
                    <div className="mt-3 rounded-lg border border-emerald-400/20 bg-emerald-400/5 p-3 text-xs text-emerald-100/80">
                      <p className="font-semibold text-emerald-300">
                        Shopify AI Toolkit {message.shopifyToolkit.skillVersion || "version unknown"}
                      </p>
                      {message.shopifyToolkit.agentProfile && (
                        <p className="mt-1">Specialist: {message.shopifyToolkit.agentProfile}</p>
                      )}
                      {message.shopifyToolkit.topics && (
                        <p className="mt-1">Skills: {message.shopifyToolkit.topics.join(", ")}</p>
                      )}
                      {message.shopifyToolkit.mcp && (
                        <p className="mt-1">
                          Dev MCP: {message.shopifyToolkit.mcp.connected ? "connected" : message.shopifyToolkit.mcp.enabled ? "unavailable" : "disabled"}
                          {" · "}tools {message.shopifyToolkit.mcp.toolsDiscovered}/{message.shopifyToolkit.mcp.toolsExposed}
                          {" · "}calls {message.shopifyToolkit.mcp.callsMade}
                        </p>
                      )}
                      {message.shopifyToolkit.mcp?.lastError && (
                        <p className="mt-1 break-words text-amber-200">
                          MCP: {message.shopifyToolkit.mcp.lastError}
                        </p>
                      )}
                      {message.shopifyToolkit.mcp?.reason && (
                        <p className="mt-1 break-words text-amber-200">
                          {message.shopifyToolkit.mcp.reason}
                        </p>
                      )}
                    </div>
                  )}

                </div>

              ))}

              {loading && (
                <div className="animate-pulse text-sm text-violet-400">
                  Horizon AI is thinking...
                </div>
              )}

            </div>

            {/* INPUT */}

            <div className="mt-5 rounded-xl border border-white/10 p-3">

              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (
                    e.key === "Enter" &&
                    !e.shiftKey
                  ) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder="Ask anything about your Shopify Horizon theme..."
                className="w-full resize-none bg-transparent text-sm outline-none"
                rows={3}
              />

              <div className="flex justify-between pt-3">

                <Paperclip className="cursor-pointer text-gray-400" />

                <button
                  onClick={sendMessage}
                  disabled={
                    loading ||
                    !prompt.trim() ||
                    !hydrated
                  }
                  className="rounded-lg bg-violet-600 p-2 disabled:opacity-40"
                >
                  <Send size={18} />
                </button>

              </div>

            </div>

          </div>

          {/* LOCAL FIGMA .FIG READER */}

          <div id="figma-workspace" className="flex min-h-[620px] min-w-0 flex-col rounded-2xl border border-white/10 bg-[#141525] p-5">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <Code2 size={19} className="text-violet-300" />
                  <h3 className="font-semibold">Local Figma .fig</h3>
                </div>
                <p className="mt-1 text-xs text-gray-400">
                  Reads figma/design.fig directly from this repository. No Figma API or screenshot pipeline.
                </p>
              </div>
              <button
                type="button"
                onClick={() => void refreshLocalFigStatus()}
                disabled={localFigLoading}
                aria-label="Refresh local .fig status"
                className="rounded-lg p-2 text-gray-300 transition-colors hover:bg-white/10 disabled:opacity-40"
              >
                <RefreshCw size={17} className={localFigLoading ? "animate-spin" : ""} />
              </button>
            </div>

            {localFigStatus?.available && localFigStatus.design ? (
              <div className="flex-1 rounded-xl border border-emerald-400/20 bg-[#0c0d18] p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-emerald-300">{localFigStatus.design.fileName}</p>
                    <p className="mt-1 text-xs text-gray-400">{localFigStatus.design.filePath}</p>
                  </div>
                  <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-300">
                    Original .fig parsed
                  </span>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {[
                    ["Pages", localFigStatus.design.pageCount],
                    ["Nodes", localFigStatus.design.nodeCount],
                    ["Frames", localFigStatus.design.frameCount],
                    ["Image assets", localFigStatus.design.imageCount],
                  ].map(([label, value]) => (
                    <div key={String(label)} className="rounded-lg border border-white/10 bg-white/[0.03] p-3">
                      <p className="text-xl font-semibold text-white">{value}</p>
                      <p className="mt-1 text-[11px] text-gray-400">{label}</p>
                    </div>
                  ))}
                </div>

                <div className="mt-5">
                  <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-400">Pages</p>
                  <div className="flex flex-wrap gap-2">
                    {localFigStatus.design.pageNames.map((name) => (
                      <span key={name} className="rounded-full border border-violet-400/20 bg-violet-500/10 px-3 py-1.5 text-xs text-violet-200">
                        {name}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="mt-5 max-h-56 overflow-y-auto rounded-lg border border-white/10">
                  <div className="border-b border-white/10 px-3 py-2 text-xs font-semibold text-gray-300">Frames and components</div>
                  {localFigStatus.design.frames.length > 0 ? localFigStatus.design.frames.slice(0, 80).map((frame) => (
                    <div key={frame.id} className="flex items-center justify-between gap-3 border-b border-white/5 px-3 py-2 last:border-0">
                      <div className="min-w-0">
                        <p className="truncate text-xs text-white">{frame.name}</p>
                        <p className="truncate text-[10px] text-gray-500">{frame.page} · {frame.id}</p>
                      </div>
                      <p className="shrink-0 text-[10px] text-gray-400">
                        {frame.width ?? "?"} × {frame.height ?? "?"}
                      </p>
                    </div>
                  )) : (
                    <p className="px-3 py-4 text-xs text-gray-500">No frames were indexed in this file.</p>
                  )}
                </div>

                <p className="mt-4 text-xs leading-5 text-gray-500">
                  Horizon reads the native node tree, text, layout settings, fills, typography, vector metadata and original embedded image assets. It does not use a rendered screenshot as design input.
                </p>
              </div>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/20 bg-[#0c0d18] p-7 text-center">
                <Code2 size={46} className="mb-4 text-violet-400" />
                <h3 className="font-semibold">Local design.fig not ready</h3>
                <p className="mt-2 max-w-md text-sm leading-6 text-gray-400">
                  Put your original Figma export in the repository at <code className="text-violet-300">figma/design.fig</code>, then refresh this panel.
                </p>
                {localFigStatus?.error && (
                  <p role="alert" className="mt-4 max-w-lg rounded-lg border border-amber-400/20 bg-amber-400/5 px-3 py-2 text-xs leading-5 text-amber-200">
                    {localFigStatus.error}
                  </p>
                )}
                {!localFigStatus && (
                  <p className="mt-4 text-xs text-gray-500">{localFigLoading ? "Checking local .fig file…" : "Checking local file status…"}</p>
                )}
                <button
                  type="button"
                  onClick={() => void refreshLocalFigStatus()}
                  disabled={localFigLoading}
                  className="mt-5 rounded-lg bg-violet-600 px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-violet-500 disabled:opacity-40"
                >
                  Check .fig file
                </button>
              </div>
            )}
          </div>

        </div>

      </section>

    </main>
  );
}