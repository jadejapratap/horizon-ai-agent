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
  const [testSecret, setTestSecret] = useState("");
  const [testRunning, setTestRunning] = useState(false);
  const [testStatus, setTestStatus] = useState("");

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

  // Run the sticky-header diagnostic and publish its structured log to GitHub.
  async function runStickyHeaderTest() {
    if (!testSecret.trim() || testRunning) return;
    setTestRunning(true);
    setTestStatus("Running agent and publishing runtime log...");
    try {
      const response = await fetch("/api/test-runner", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ secret: testSecret, prompt: `Explain how the Horizon sticky header works. Trace how sections/header.liquid controls the sticky header setting, how assets/base.css handles sticky states, and how assets/header.js observes the header.

Provide exact source evidence for every claim. Keep each claim atomic. Preserve the exact conditions, selectors, and operations in the answer. Do not make assumptions or include claims without verified source evidence.` }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || `Test failed (HTTP ${response.status})`);
      const summary = data.result?.evidenceSummary;
      const coverage = data.result?.retrievalCoverage;
      const details = [
        `Sticky Header test completed. HTTP ${data.httpStatus}; duration ${Math.round(data.durationMs / 1000)}s.`,
        summary ? `Evidence claims: ${summary.verified}/${summary.total} verified; ${summary.unverified} unverified.` : "",
        coverage ? `Retrieval coverage: ${JSON.stringify(coverage)}` : "",
        `GitHub log: ${data.githubLogUrl}`,
      ].filter(Boolean).join("\\n");
      setTestStatus(details);
    } catch (error) {
      setTestStatus(error instanceof Error ? error.message : "Test runner failed.");
    } finally {
      setTestRunning(false);
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

        <div className="mx-8 mt-5 flex flex-wrap items-center gap-3 rounded-xl border border-violet-500/30 bg-violet-500/5 p-4">
          <div className="min-w-52 flex-1">
            <p className="text-sm font-semibold text-violet-300">Runtime Test Runner</p>
            <p className="text-xs text-gray-400">Run the Sticky Header test and publish its structured log to GitHub.</p>
          </div>
          <input
            type="password"
            value={testSecret}
            onChange={(e) => setTestSecret(e.target.value)}
            placeholder="Test runner secret"
            autoComplete="off"
            className="min-w-52 rounded-lg border border-white/10 bg-[#0c0d18] px-3 py-2 text-sm outline-none"
          />
          <button
            onClick={runStickyHeaderTest}
            disabled={!testSecret.trim() || testRunning}
            className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium disabled:opacity-40"
          >
            {testRunning ? "Running..." : "Run Sticky Header Test"}
          </button>
          {testStatus && <p className="w-full whitespace-pre-wrap break-words text-xs text-gray-300">{testStatus}</p>}
        </div>

        {/* CHAT + PREVIEW */}

        <div className="grid flex-1 grid-cols-2 gap-6 p-8">

          {/* AI CHAT */}

          <div className="flex flex-col rounded-2xl border border-white/10 bg-[#141525] p-6">

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

          <div className="rounded-2xl border border-white/10 bg-[#141525] p-6">

            <div className="mb-6 flex justify-between">

              <h3 className="font-semibold">
                Live Preview
              </h3>

              <div className="flex gap-4 text-gray-400">

                <Monitor />

                <Smartphone />

              </div>

            </div>

            <div className="flex h-[75%] flex-col items-center justify-center rounded-xl border border-dashed border-white/20 bg-[#0c0d18]">

              <Code2
                size={48}
                className="mb-4 text-violet-400"
              />

              <h3 className="font-semibold">
                Preview Workspace
              </h3>

              <p className="mt-2 text-sm text-gray-400">
                Shopify theme preview will appear here.
              </p>

            </div>

          </div>

        </div>

      </section>

    </main>
  );
}