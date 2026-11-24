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
  ExternalLink,
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

type Message = {
  role: "user" | "assistant";
  content: string;
  shopifyToolkit?: ShopifyToolkitStatus;
  themeEdit?: ThemeEditMessage;
  figmaDesign?: { fileName: string; nodeName: string; nodeId: string } | null;
};

type FigmaAgentStatus = "idle" | "checking" | "configured" | "connected" | "error";

const STORAGE_KEY = "horizon-ai-chat-history";
const FIGMA_STORAGE_KEY = "horizon-ai-figma-url";

type FigmaFileType = "design" | "proto" | "board" | "slides" | "deck";

type FigmaEmbedTarget = {
  originalUrl: string;
  embedUrl: string;
  fileType: FigmaFileType;
  fileKey: string;
};

function createFigmaEmbedTarget(value: string, mode: "design" | "dev"): FigmaEmbedTarget {
  let parsed: URL;
  try {
    parsed = new URL(value.trim());
  } catch {
    throw new Error("Paste a valid Figma design, prototype, FigJam, or Slides URL.");
  }

  const hostname = parsed.hostname.toLowerCase();
  const isFigmaHost = hostname === "figma.com" || hostname.endsWith(".figma.com");
  if (
    parsed.protocol !== "https:" ||
    !isFigmaHost ||
    parsed.username ||
    parsed.password ||
    parsed.port
  ) {
    throw new Error("Use a secure https://www.figma.com share URL.");
  }

  const match = parsed.pathname.match(/^\/(design|file|proto|board|slides|deck)\/([a-z0-9]{22,128})(?:\/.*)?$/i);
  if (!match) {
    throw new Error("Supported links: Figma Design, Prototype, FigJam board, Slides, or Deck share URLs.");
  }

  const requestedType = match[1].toLowerCase();
  const fileType = (requestedType === "file" ? "design" : requestedType) as FigmaFileType;
  const fileKey = match[2];
  const remainder = parsed.pathname.split("/").slice(3).filter(Boolean).join("/");
  const embedUrl = new URL(
    "https://embed.figma.com/" + fileType + "/" + fileKey + (remainder ? "/" + remainder : "")
  );

  parsed.searchParams.forEach((parameterValue, key) => {
    if (key !== "embed_host" && key !== "embed_origin" && key !== "url" && key !== "m" && key !== "mode") {
      embedUrl.searchParams.set(key, parameterValue);
    }
  });

  if (fileType === "design" && mode === "dev") {
    embedUrl.searchParams.set("mode", "dev");
  }

  embedUrl.searchParams.set("embed-host", "horizon-ai");
  embedUrl.searchParams.set("theme", "dark");
  embedUrl.searchParams.set("footer", "false");

  const originalUrl = parsed.origin + parsed.pathname + parsed.search;
  return {
    originalUrl,
    embedUrl: embedUrl.toString(),
    fileType,
    fileKey,
  };
}

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [figmaInput, setFigmaInput] = useState("");
  const [figmaOriginalUrl, setFigmaOriginalUrl] = useState("");
  const [figmaEmbedUrl, setFigmaEmbedUrl] = useState("");
  const [figmaFileType, setFigmaFileType] = useState<FigmaFileType | "">("");
  const [figmaFileKey, setFigmaFileKey] = useState("");
  const [figmaMode, setFigmaMode] = useState<"design" | "dev">("design");
  const [figmaRevision, setFigmaRevision] = useState(0);
  const [figmaError, setFigmaError] = useState("");
  const [figmaAgentStatus, setFigmaAgentStatus] = useState<FigmaAgentStatus>("idle");
  const [figmaAgentMessage, setFigmaAgentMessage] = useState("");
  const [figmaAgentFileName, setFigmaAgentFileName] = useState("");

  async function checkFigmaAgentConnection(url: string) {
    setFigmaAgentStatus("checking");
    setFigmaAgentMessage("Checking Figma API access for Horizon AI Agent...");
    setFigmaAgentFileName("");

    try {
      const response = await fetch("/api/figma/connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Figma Agent connection failed.");
      }

      setFigmaAgentStatus(data.cached ? "connected" : "configured");
      setFigmaAgentMessage(
        data.cached
          ? "A saved Figma design context is cached locally; no Figma API request was needed."
          : "Token is configured. File access will be checked once when the Agent first reads this design."
      );
      setFigmaAgentFileName(data.fileName || "");
    } catch (error) {
      setFigmaAgentStatus("error");
      setFigmaAgentMessage(error instanceof Error ? error.message : "Figma Agent connection failed.");
    }
  }

  function loadFigmaDesign(mode: "design" | "dev" = figmaMode) {
    const value = figmaInput.trim();
    if (!value) {
      setFigmaError("Paste your Figma share URL first.");
      return;
    }

    try {
      const target = createFigmaEmbedTarget(value, mode);
      setFigmaOriginalUrl(target.originalUrl);
      setFigmaEmbedUrl(target.embedUrl);
      setFigmaFileType(target.fileType);
      setFigmaFileKey(target.fileKey);
      setFigmaMode(mode);
      setFigmaError("");
      setFigmaRevision((revision) => revision + 1);
      localStorage.setItem(FIGMA_STORAGE_KEY, target.originalUrl);
      if (target.fileType === "design") {
        void checkFigmaAgentConnection(target.originalUrl);
      } else {
        setFigmaAgentStatus("error");
        setFigmaAgentMessage("Horizon AI Agent reads Figma Design files only. The embed preview can still show this file type.");
      }
    } catch (error) {
      setFigmaError(error instanceof Error ? error.message : "Unable to load this Figma link.");
    }
  }

  useEffect(() => {
    try {
      const savedFigmaUrl = localStorage.getItem(FIGMA_STORAGE_KEY);
      if (savedFigmaUrl) {
        const savedMode = /[?&](?:m|mode)=dev(?:&|$)/.test(savedFigmaUrl) ? "dev" : "design";
        const target = createFigmaEmbedTarget(savedFigmaUrl, savedMode);
        setFigmaInput(savedFigmaUrl);
        setFigmaOriginalUrl(target.originalUrl);
        setFigmaEmbedUrl(target.embedUrl);
        setFigmaFileType(target.fileType);
        setFigmaFileKey(target.fileKey);
        setFigmaMode(savedMode);
        if (target.fileType === "design") {
          void checkFigmaAgentConnection(target.originalUrl);
        } else {
          setFigmaAgentStatus("error");
          setFigmaAgentMessage("Horizon AI Agent reads Figma Design files only.");
        }
      }
    } catch (error) {
      console.warn("Could not restore the saved Figma design:", error);
    }
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
          figmaUrl: figmaFileType === "design" ? figmaOriginalUrl : "",
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

      if (data.figmaDesign) {
        setFigmaAgentStatus("connected");
        setFigmaAgentMessage("Figma design was read successfully and cached on this computer.");
        setFigmaAgentFileName(data.figmaDesign.fileName || "");
      }

      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: data.message || "No response received.",
          shopifyToolkit: data.shopifyToolkit,
          themeEdit: data.themeEdit,
          figmaDesign: data.figmaDesign ?? null,
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
            Figma Designs
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

                  {message.figmaDesign && (
                    <div className="mt-3 rounded-lg border border-violet-400/20 bg-violet-500/5 px-3 py-2 text-xs text-violet-200">
                      Figma reference used: {message.figmaDesign.fileName} · {message.figmaDesign.nodeName} · node {message.figmaDesign.nodeId}
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

          {/* FIGMA WORKSPACE */}

          <div id="figma-workspace" className="flex min-h-[620px] min-w-0 flex-col rounded-2xl border border-white/10 bg-[#141525] p-5">

            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <Code2 size={19} className="text-violet-300" />
                  <h3 className="font-semibold">Figma Designs</h3>
                </div>
                <p className="mt-1 truncate text-xs text-gray-400">
                  {figmaEmbedUrl
                    ? "Preview loaded · " + (figmaFileType === "proto" ? "Prototype" : figmaFileType === "board" ? "FigJam" : figmaFileType === "slides" || figmaFileType === "deck" ? "Slides" : "Design") + " · " + figmaFileKey
                    : "No Figma design connected"}
                </p>
                {figmaEmbedUrl && (
                  <p className={"mt-1 break-words text-[11px] " + (figmaAgentStatus === "connected" ? "text-emerald-300" : figmaAgentStatus === "checking" ? "text-violet-300" : "text-amber-200")}>
                    {figmaAgentStatus === "connected"
                      ? "Agent API connected · cached " + (figmaAgentFileName || figmaAgentMessage)
                      : figmaAgentStatus === "configured"
                        ? "Agent token configured · " + (figmaAgentFileName || figmaAgentMessage)
                        : figmaAgentStatus === "checking"
                          ? figmaAgentMessage
                          : figmaAgentStatus === "error"
                            ? "Preview only · " + figmaAgentMessage
                            : "Agent API connection not checked"}
                  </p>
                )}
              </div>

              <div className="flex items-center gap-2 text-gray-300">
                {figmaFileType === "design" && (
                  <>
                    <button
                      type="button"
                      onClick={() => loadFigmaDesign("design")}
                      aria-pressed={figmaMode === "design"}
                      className={"rounded-lg px-3 py-2 text-xs transition-colors " + (figmaMode === "design" ? "bg-violet-600/25 text-violet-300" : "hover:bg-white/10")}
                    >
                      Design
                    </button>
                    <button
                      type="button"
                      onClick={() => loadFigmaDesign("dev")}
                      aria-pressed={figmaMode === "dev"}
                      className={"rounded-lg px-3 py-2 text-xs transition-colors " + (figmaMode === "dev" ? "bg-violet-600/25 text-violet-300" : "hover:bg-white/10")}
                    >
                      Dev Mode
                    </button>
                  </>
                )}
                <button
                  type="button"
                  onClick={() => loadFigmaDesign()}
                  disabled={!figmaOriginalUrl}
                  aria-label="Refresh Figma embed"
                  className="rounded-lg p-2 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                >
                  <RefreshCw size={17} />
                </button>
                {figmaOriginalUrl && (
                  <a
                    href={figmaOriginalUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="Open design in Figma"
                    className="rounded-lg p-2 transition-colors hover:bg-white/10"
                  >
                    <ExternalLink size={17} />
                  </a>
                )}
              </div>
            </div>

            <form
              onSubmit={(event) => {
                event.preventDefault();
                loadFigmaDesign("design");
              }}
              className="mb-3 flex gap-2"
            >
              <input
                type="url"
                value={figmaInput}
                onChange={(event) => setFigmaInput(event.target.value)}
                placeholder="Paste Figma Design, Prototype, FigJam or Slides link"
                aria-label="Figma share URL"
                className="min-w-0 flex-1 rounded-lg border border-white/10 bg-[#0c0d18] px-3 py-2 text-xs text-white outline-none transition-colors focus:border-violet-500/60"
              />
              <button
                type="submit"
                className="shrink-0 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium transition-colors hover:bg-violet-500"
              >
                Load Figma
              </button>
            </form>

            {figmaError && (
              <p role="alert" className="mb-3 rounded-lg border border-red-400/20 bg-red-400/5 px-3 py-2 text-xs text-red-200">
                {figmaError}
              </p>
            )}

            {figmaEmbedUrl ? (
              <div className="min-h-[460px] flex-1 overflow-hidden rounded-xl border border-white/15 bg-[#0c0d18]">
                <iframe
                  key={figmaEmbedUrl + ":" + figmaRevision}
                  src={figmaEmbedUrl}
                  title={"Figma " + (figmaFileType || "design") + " workspace"}
                  className="h-full min-h-[460px] w-full border-0 bg-[#0c0d18]"
                  loading="lazy"
                  referrerPolicy="strict-origin-when-cross-origin"
                  allowFullScreen
                />
              </div>
            ) : (
              <div className="flex min-h-[460px] flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/20 bg-[#0c0d18] p-7 text-center">
                <Code2 size={46} className="mb-4 text-violet-400" />
                <h3 className="font-semibold">Connect your Figma workspace</h3>
                <p className="mt-2 max-w-md text-sm leading-6 text-gray-400">
                  Paste a Figma share link to open a live design, interactive prototype, FigJam board, or Slides file here.
                </p>
                <div className="mt-5 flex flex-wrap justify-center gap-2 text-xs">
                  <span className="rounded-full border border-white/10 px-3 py-1.5 text-gray-300">Figma Design</span>
                  <span className="rounded-full border border-white/10 px-3 py-1.5 text-gray-300">Prototype</span>
                  <span className="rounded-full border border-white/10 px-3 py-1.5 text-gray-300">FigJam</span>
                  <span className="rounded-full border border-white/10 px-3 py-1.5 text-gray-300">Slides</span>
                </div>
                <p className="mt-5 max-w-md text-xs leading-5 text-gray-500">
                  The file must be shared with your Figma account or have link access. Private organization files require you to be signed in with permission.
                </p>
              </div>
            )}

          </div>

        </div>

      </section>

    </main>
  );
}