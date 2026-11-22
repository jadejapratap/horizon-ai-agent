import { NextRequest, NextResponse } from "next/server";
import { timingSafeEqual } from "node:crypto";

export const runtime = "nodejs";
export const maxDuration = 180;

const REPOSITORY = "jadejapratap/horizon-ai-agent";
const BRANCH = "Horizon-ai";
const LOG_PATH = "agent-debug.log";
const STICKY_HEADER_PROMPT = `Explain how the Horizon sticky header works. Trace how sections/header.liquid controls the sticky header setting, how assets/base.css handles sticky states, and how assets/header.js observes the header.

Provide exact source evidence for every claim. Keep each claim atomic. Preserve the exact conditions, selectors, and operations in the answer. Do not make assumptions or include claims without verified source evidence.`;

function isAuthorized(candidate: unknown): boolean {
  const expected = process.env.TEST_RUNNER_SECRET;
  if (!expected || typeof candidate !== "string") return false;
  const a = Buffer.from(candidate);
  const b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}

async function saveLogToGitHub(log: string): Promise<string> {
  const token = process.env.GITHUB_TOKEN;
  if (!token) throw new Error("GITHUB_TOKEN is not configured on the server.");

  const apiUrl = `https://api.github.com/repos/${REPOSITORY}/contents/${LOG_PATH}`;
  const headers = {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
  };

  const existing = await fetch(`${apiUrl}?ref=${encodeURIComponent(BRANCH)}`, {
    headers,
    cache: "no-store",
  });

  let sha: string | undefined;
  if (existing.ok) {
    const current = await existing.json();
    sha = typeof current.sha === "string" ? current.sha : undefined;
  } else if (existing.status !== 404) {
    throw new Error(`GitHub log lookup failed (${existing.status}): ${(await existing.text()).slice(0, 500)}`);
  }

  const update = await fetch(apiUrl, {
    method: "PUT",
    headers,
    body: JSON.stringify({
      message: "Update Horizon Agent runtime test log",
      content: Buffer.from(log, "utf8").toString("base64"),
      branch: BRANCH,
      ...(sha ? { sha } : {}),
    }),
  });

  if (!update.ok) {
    throw new Error(`GitHub log upload failed (${update.status}): ${(await update.text()).slice(0, 700)}`);
  }

  return `https://github.com/${REPOSITORY}/blob/${BRANCH}/${LOG_PATH}`;
}

export async function POST(request: NextRequest) {
  const startedAt = new Date().toISOString();
  const started = Date.now();

  try {
    const body = await request.json();
    if (!isAuthorized(body.secret)) {
      return NextResponse.json({ success: false, error: "Unauthorized. Check TEST_RUNNER_SECRET." }, { status: 401 });
    }

    const prompt = typeof body.prompt === "string" && body.prompt.trim()
      ? body.prompt.trim()
      : STICKY_HEADER_PROMPT;

    if (prompt.length > 12000) {
      return NextResponse.json({ success: false, error: "Prompt is too long (maximum 12,000 characters)." }, { status: 400 });
    }

    const agentUrl = new URL("/api/agent", request.url);
    const agentResponse = await fetch(agentUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, history: [] }),
      cache: "no-store",
      signal: AbortSignal.timeout(170000),
    });

    const responseText = await agentResponse.text();
    let agentResult: unknown;
    try {
      agentResult = JSON.parse(responseText);
    } catch {
      agentResult = { rawResponse: responseText.slice(0, 30000) };
    }

    const logObject = {
      type: "horizon-agent-runtime-test",
      startedAt,
      completedAt: new Date().toISOString(),
      durationMs: Date.now() - started,
      prompt,
      httpStatus: agentResponse.status,
      agentResult,
    };

    const githubLogUrl = await saveLogToGitHub(JSON.stringify(logObject, null, 2) + "\n");

    return NextResponse.json({
      success: agentResponse.ok,
      httpStatus: agentResponse.status,
      durationMs: logObject.durationMs,
      githubLogUrl,
      result: agentResult,
    }, { status: agentResponse.ok ? 200 : 502 });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown test-runner error";
    return NextResponse.json({ success: false, error: message }, { status: 500 });
  }
}
