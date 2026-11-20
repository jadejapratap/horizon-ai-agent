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

type Message = {
  role: "user" | "assistant";
  content: string;
};

const STORAGE_KEY = "horizon-ai-chat-history";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [hydrated, setHydrated] = useState(false);

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