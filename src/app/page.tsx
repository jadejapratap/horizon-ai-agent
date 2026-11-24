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
  Monitor,
  Smartphone,
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

type Message = {
  role: "user" | "assistant";
  content: string;
  shopifyToolkit?: ShopifyToolkitStatus;
};

const STORAGE_KEY = "horizon-ai-chat-history";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const configuredPreviewUrl = process.env.NEXT_PUBLIC_SHOPIFY_THEME_PREVIEW_URL?.trim() || "";
  const [previewInput, setPreviewInput] = useState(configuredPreviewUrl);
  const [previewUrl, setPreviewUrl] = useState(() => {
    if (!configuredPreviewUrl) return "";
    try {
      const parsed = new URL(configuredPreviewUrl);
      const localHosts = ["localhost", "127.0.0.1", "[::1]"];
      const isAllowedProtocol = parsed.protocol === "https:"
        || (parsed.protocol === "http:" && localHosts.includes(parsed.hostname));
      return isAllowedProtocol && !parsed.username && !parsed.password
        ? parsed.toString()
        : "";
    } catch {
      return "";
    }
  });
  const [previewMode, setPreviewMode] = useState<"desktop" | "mobile">("desktop");
  const [previewRevision, setPreviewRevision] = useState(0);
  const [previewError, setPreviewError] = useState("");

  function connectThemePreview() {
    const value = previewInput.trim();
    if (!value) {
      setPreviewError("Paste the storefront preview URL first.");
      return;
    }

    try {
      const parsed = new URL(value);
      const localHosts = ["localhost", "127.0.0.1", "[::1]"];
      const isAllowedProtocol = parsed.protocol === "https:"
        || (parsed.protocol === "http:" && localHosts.includes(parsed.hostname));
      if (!isAllowedProtocol || parsed.username || parsed.password) {
        throw new Error("Use an HTTPS storefront URL or a local Shopify CLI URL such as http://127.0.0.1:9292.");
      }
      if (parsed.origin === window.location.origin) {
        throw new Error("Enter the Shopify storefront preview URL, not the Horizon AI application URL.");
      }

      setPreviewUrl(parsed.toString());
      setPreviewError("");
      setPreviewRevision((revision) => revision + 1);
    } catch (error) {
      setPreviewError(error instanceof Error
        ? error.message
        : "Enter a valid Shopify storefront preview URL.");
    }
  }


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

      setMessages([
        ...updatedMessages,
        {
          role: "assistant",
          content: data.message || "No response received.",
          shopifyToolkit: data.shopifyToolkit,
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

          <div className="flex gap-3">
            <Code2 />
            Figma Designs
          </div>

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

          {/* LIVE PREVIEW */}

          <div className="flex min-h-[620px] min-w-0 flex-col rounded-2xl border border-white/10 bg-[#141525] p-5">

            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="min-w-0">
                <h3 className="font-semibold">Live Preview</h3>
                <p className="mt-1 truncate text-xs text-gray-400">
                  {previewUrl ? `Connected · ${new URL(previewUrl).host}` : "No storefront preview connected"}
                </p>
              </div>

              <div className="flex items-center gap-2 text-gray-300">
                <button
                  type="button"
                  onClick={() => setPreviewMode("desktop")}
                  aria-label="Desktop preview"
                  aria-pressed={previewMode === "desktop"}
                  className={`rounded-lg p-2 transition-colors ${previewMode === "desktop" ? "bg-violet-600/25 text-violet-300" : "hover:bg-white/10"}`}
                >
                  <Monitor size={18} />
                </button>
                <button
                  type="button"
                  onClick={() => setPreviewMode("mobile")}
                  aria-label="Mobile preview"
                  aria-pressed={previewMode === "mobile"}
                  className={`rounded-lg p-2 transition-colors ${previewMode === "mobile" ? "bg-violet-600/25 text-violet-300" : "hover:bg-white/10"}`}
                >
                  <Smartphone size={18} />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setPreviewError("");
                    setPreviewRevision((revision) => revision + 1);
                  }}
                  disabled={!previewUrl}
                  aria-label="Refresh theme preview"
                  className="rounded-lg p-2 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-30"
                >
                  <RefreshCw size={17} />
                </button>
                {previewUrl && (
                  <a
                    href={previewUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="Open storefront preview in a new tab"
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
                connectThemePreview();
              }}
              className="mb-3 flex gap-2"
            >
              <input
                type="url"
                value={previewInput}
                onChange={(event) => setPreviewInput(event.target.value)}
                placeholder="Paste Shopify theme preview URL (e.g. http://127.0.0.1:9292)"
                aria-label="Shopify storefront preview URL"
                className="min-w-0 flex-1 rounded-lg border border-white/10 bg-[#0c0d18] px-3 py-2 text-xs text-white outline-none transition-colors focus:border-violet-500/60"
              />
              <button
                type="submit"
                className="shrink-0 rounded-lg bg-violet-600 px-3 py-2 text-xs font-medium transition-colors hover:bg-violet-500"
              >
                Load Preview
              </button>
            </form>

            {previewError && (
              <p role="alert" className="mb-3 rounded-lg border border-red-400/20 bg-red-400/5 px-3 py-2 text-xs text-red-200">
                {previewError}
              </p>
            )}

            {previewUrl ? (
              <div className="flex min-h-[460px] flex-1 items-start justify-center overflow-auto rounded-xl border border-white/15 bg-[#0c0d18] p-3">
                <div
                  className={`h-[calc(100vh-330px)] min-h-[430px] overflow-hidden rounded-lg bg-white shadow-2xl transition-[width] duration-300 ${previewMode === "mobile" ? "w-[390px] max-w-full" : "w-full"}`}
                >
                  <iframe
                    key={`${previewUrl}:${previewRevision}`}
                    src={previewUrl}
                    title="Shopify storefront theme preview"
                    className="h-full w-full border-0 bg-white"
                    loading="lazy"
                    referrerPolicy="no-referrer"
                    sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-downloads"
                    onError={() => setPreviewError("The storefront preview could not be loaded. Check the URL and whether the preview host allows iframe embedding.")}
                  />
                </div>
              </div>
            ) : (
              <div className="flex min-h-[460px] flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-white/20 bg-[#0c0d18] p-7 text-center">
                <Code2 size={44} className="mb-4 text-violet-400" />
                <h3 className="font-semibold">Connect a real Shopify theme preview</h3>
                <p className="mt-2 max-w-md text-sm text-gray-400">
                  Start Shopify CLI preview for the theme in this repository, then paste the local preview URL above.
                </p>
                <code className="mt-5 max-w-full overflow-x-auto rounded-lg border border-white/10 bg-black/30 px-4 py-3 text-left text-xs text-violet-200">
                  shopify theme dev --path themes/horizon --store your-store.myshopify.com
                </code>
                <p className="mt-4 max-w-md text-xs leading-5 text-gray-500">
                  Shopify CLI authentication is required. Keep the CLI preview running while using Horizon AI. For a remote deployment, use an HTTPS preview URL reachable from the browser.
                </p>
              </div>
            )}

          </div>
          </div>

        </div>

      </section>

    </main>
  );
}