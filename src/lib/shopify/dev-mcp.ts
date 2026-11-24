import { createHash } from "node:crypto";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

type GeminiRequest = {
  model: string;
  contents: any;
  config?: Record<string, any>;
};

type McpToolInfo = {
  name: string;
  description?: string;
  inputSchema?: Record<string, any>;
};

type McpConnection = {
  client: Client;
  tools: McpToolInfo[];
  aliases: Map<string, string>;
  aliasByToolName: Map<string, string>;
};

type McpStatus = {
  enabled: boolean;
  connected: boolean;
  toolsDiscovered: number;
  toolsExposed: number;
  callsMade: number;
  lastError?: string;
  reason?: string;
};

type GeminiToolCall = {
  name: string;
  args?: Record<string, unknown>;
  id?: string;
};

type GenerateFunction = (request: any) => Promise<any>;

const MAX_GEMINI_TOOL_DECLARATIONS = 32;
const MAX_MCP_CALLS_PER_REQUEST = 6;
const MAX_TOOL_ROUNDS = 3;
const DEFAULT_TOOL_TIMEOUT_MS = 20000;
const MAX_TOOL_RESULT_CHARS = 24000;

const runtimeGlobal = globalThis as typeof globalThis & {
  __horizonShopifyMcpConnection?: Promise<McpConnection>;
};

function parseArgs(value: string | undefined, fallback: string[]): string[] {
  if (!value?.trim()) return fallback;
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
      return parsed;
    }
  } catch {
    // A malformed args setting must not silently become a shell command.
  }
  throw new Error("SHOPIFY_DEV_MCP_ARGS must be a JSON array of strings.");
}

function safeChildEnvironment(): Record<string, string> {
  const allowed = [
    "PATH", "Path", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
    "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATHEXT", "COMSPEC",
    "NPM_CONFIG_CACHE", "npm_config_cache", "CI",
  ];
  const env: Record<string, string> = {};
  for (const key of allowed) {
    const value = process.env[key];
    if (typeof value === "string") env[key] = value;
  }
  // Do not inherit GEMINI_API_KEY, GITHUB_TOKEN, Shopify OAuth tokens,
  // database credentials, or the full parent environment.
  env.OPT_OUT_INSTRUMENTATION = "true";
  env.DO_NOT_TRACK = "1";
  return env;
}

function getServerCommand(): { command: string; args: string[] } {
  const packageName = process.env.SHOPIFY_DEV_MCP_PACKAGE || "@shopify/dev-mcp@1.19.0";
  const configuredCommand = process.env.SHOPIFY_DEV_MCP_COMMAND;
  const configuredArgs = parseArgs(process.env.SHOPIFY_DEV_MCP_ARGS, ["-y", packageName]);

  if (configuredCommand) {
    return {
      command: configuredCommand,
      args: configuredArgs,
    };
  }

  if (process.platform === "win32") {
    return {
      command: process.env.ComSpec || "cmd.exe",
      args: ["/d", "/s", "/c", ["npx", ...configuredArgs].join(" ")],
    };
  }

  return { command: "npx", args: configuredArgs };
}

function buildToolAliases(tools: McpToolInfo[]): {
  aliases: Map<string, string>;
  aliasByToolName: Map<string, string>;
} {
  const aliases = new Map<string, string>();
  const aliasByToolName = new Map<string, string>();
  for (const tool of tools) {
    const base = toGeminiFunctionName(tool.name);
    let alias = base;
    if (aliases.has(alias) && aliases.get(alias) !== tool.name) {
      const suffix = createHash("sha256").update(tool.name).digest("hex").slice(0, 10);
      alias = `${base.slice(0, 53)}_${suffix}`;
    }
    aliases.set(alias, tool.name);
    aliasByToolName.set(tool.name, alias);
  }
  return { aliases, aliasByToolName };
}

async function refreshMcpTools(connection: McpConnection, timeoutMs: number): Promise<void> {
  const discovered = await withTimeout(
    connection.client.listTools(),
    timeoutMs,
    "Shopify Dev MCP tool discovery"
  );
  connection.tools = (discovered.tools ?? []) as McpToolInfo[];
  const mapped = buildToolAliases(connection.tools);
  connection.aliases = mapped.aliases;
  connection.aliasByToolName = mapped.aliasByToolName;
}

async function createMcpConnection(): Promise<McpConnection> {
  const command = getServerCommand();
  const transport = new StdioClientTransport({
    ...command,
    env: safeChildEnvironment(),
    stderr: "inherit",
  } as any);
  const client = new Client({
    name: "horizon-shopify-agent",
    version: "0.1.0",
  });

  await client.connect(transport);
  const discovered = await client.listTools();
  const tools = (discovered.tools ?? []) as McpToolInfo[];
  const mapped = buildToolAliases(tools);

  return {
    client,
    tools,
    aliases: mapped.aliases,
    aliasByToolName: mapped.aliasByToolName,
  };
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

async function getMcpConnection(): Promise<McpConnection> {
  if (!runtimeGlobal.__horizonShopifyMcpConnection) {
    runtimeGlobal.__horizonShopifyMcpConnection = createMcpConnection().catch((error) => {
      runtimeGlobal.__horizonShopifyMcpConnection = undefined;
      throw error;
    });
  }
  return withTimeout(
    runtimeGlobal.__horizonShopifyMcpConnection,
    Number(process.env.SHOPIFY_DEV_MCP_CONNECT_TIMEOUT_MS || 15000),
    "Shopify Dev MCP connection"
  );
}

function toGeminiFunctionName(toolName: string): string {
  const normalized = toolName.replace(/[^A-Za-z0-9_]/g, "_");
  const base = `shopify_mcp_${normalized}`;
  if (base.length <= 64) return base;
  const suffix = createHash("sha256").update(toolName).digest("hex").slice(0, 10);
  return `${base.slice(0, 53)}_${suffix}`;
}

function cleanSchema(schema: any, depth = 0): Record<string, any> {
  if (!schema || typeof schema !== "object" || Array.isArray(schema) || depth > 12) {
    return { type: "object", properties: {} };
  }

  const output: Record<string, any> = {};
  const allowed = new Set([
    "type", "description", "enum", "required", "properties", "items",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minItems", "maxItems", "minLength", "maxLength", "pattern",
    "format", "nullable",
  ]);

  for (const [key, value] of Object.entries(schema)) {
    if (!allowed.has(key)) continue;
    if (key === "properties" && value && typeof value === "object" && !Array.isArray(value)) {
      output.properties = Object.fromEntries(
        Object.entries(value as Record<string, unknown>).map(([name, property]) => [
          name,
          cleanSchema(property, depth + 1),
        ])
      );
    } else if (key === "items") {
      output.items = cleanSchema(value, depth + 1);
    } else if (key === "type" && Array.isArray(value)) {
      const types = value.filter((item) => item !== "null");
      output.type = types[0] || "object";
      if (types.length !== value.length) output.nullable = true;
    } else if (key === "enum" && Array.isArray(value)) {
      output.enum = value;
    } else if (key === "required" && Array.isArray(value)) {
      output.required = value.filter((item) => typeof item === "string");
    } else if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
      output[key] = value;
    }
  }

  if (!output.type) output.type = output.properties ? "object" : "object";
  if (output.type === "object" && !output.properties) output.properties = {};
  return output;
}

function buildGeminiDeclarations(connection: McpConnection): Array<Record<string, any>> {
  return connection.tools
    .slice(0, MAX_GEMINI_TOOL_DECLARATIONS)
    .map((tool) => ({
      name: connection.aliasByToolName.get(tool.name) || toGeminiFunctionName(tool.name),
      description: (tool.description || `Shopify Dev MCP tool: ${tool.name}`).slice(0, 900),
      parametersJsonSchema: cleanSchema(tool.inputSchema),
    }));
}

function getFunctionCalls(response: any): GeminiToolCall[] {
  const candidateParts = response?.candidates?.[0]?.content?.parts;
  const calls = Array.isArray(candidateParts)
    ? candidateParts.filter((part: any) => part?.functionCall).map((part: any) => part.functionCall)
    : [];
  const fallback = Array.isArray(response?.functionCalls) ? response.functionCalls : [];
  return (calls.length ? calls : fallback)
    .filter((call: any) => call && typeof call.name === "string")
    .map((call: any) => ({
      name: call.name,
      args: call.args && typeof call.args === "object" ? call.args : {},
      id: typeof call.id === "string" ? call.id : undefined,
    }));
}

function makeMcpOutput(result: any): string {
  const value = result?.structuredContent ?? result?.content ?? result;
  const normalized = typeof value === "string" ? value : JSON.stringify(value);
  return normalized.length > MAX_TOOL_RESULT_CHARS
    ? normalized.slice(0, MAX_TOOL_RESULT_CHARS) + "\n[Tool output truncated by Horizon Agent.]"
    : normalized;
}

async function callMcpToolWithRetry(
  client: Client,
  name: string,
  args: Record<string, unknown>,
  timeoutMs: number
): Promise<any> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      return await withTimeout(
        client.callTool({ name, arguments: args }),
        timeoutMs,
        `Shopify Dev MCP tool ${name}`
      );
    } catch (error) {
      lastError = error;
      if (attempt === 2) break;
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

export async function generateContentWithShopifyMcpTools(
  request: GeminiRequest,
  generateContent: GenerateFunction
): Promise<{ response: any; status: McpStatus }> {
  const enabled = process.env.SHOPIFY_DEV_MCP_ENABLED === "true";
  const status: McpStatus = {
    enabled,
    connected: false,
    toolsDiscovered: 0,
    toolsExposed: 0,
    callsMade: 0,
  };

  const generateWithoutTools = async () => ({
    response: await generateContent(request),
    status,
  });

  if (!enabled) {
    status.reason = "SHOPIFY_DEV_MCP_ENABLED is not true.";
    return generateWithoutTools();
  }

  if (
    (process.env.VERCEL === "1" || process.env.AWS_LAMBDA_FUNCTION_NAME) &&
    process.env.SHOPIFY_DEV_MCP_ALLOW_SERVERLESS !== "true"
  ) {
    status.reason = "Local stdio MCP is disabled on serverless hosts by default; use a persistent Node.js host or a remote MCP proxy.";
    return generateWithoutTools();
  }

  let connection: McpConnection;
  try {
    connection = await getMcpConnection();
    await refreshMcpTools(
      connection,
      Number(process.env.SHOPIFY_DEV_MCP_CONNECT_TIMEOUT_MS || 15000)
    );
    status.connected = true;
    status.toolsDiscovered = connection.tools.length;
  } catch (error) {
    status.lastError = error instanceof Error ? error.message.slice(0, 500) : String(error).slice(0, 500);
    return generateWithoutTools();
  }

  const declarations = buildGeminiDeclarations(connection);
  status.toolsExposed = declarations.length;
  if (declarations.length === 0) {
    status.reason = "Shopify Dev MCP connected but returned no tools.";
    return generateWithoutTools();
  }

  const timeoutMs = Math.max(
    1000,
    Math.min(60000, Number(process.env.SHOPIFY_DEV_MCP_TOOL_TIMEOUT_MS || DEFAULT_TOOL_TIMEOUT_MS))
  );
  let contents = Array.isArray(request.contents) ? [...request.contents] : request.contents;
  let response: any = await generateContent({
    ...request,
    config: {
      ...(request.config || {}),
      tools: [{ functionDeclarations: declarations }],
      toolConfig: { functionCallingConfig: { mode: "AUTO" } },
    },
  });

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    const calls = getFunctionCalls(response);
    if (calls.length === 0) return { response, status };

    const modelContent = response?.candidates?.[0]?.content;
    if (modelContent && Array.isArray(contents)) contents.push(modelContent);

    const functionResponses: Array<Record<string, any>> = [];
    for (const call of calls) {
      if (status.callsMade >= MAX_MCP_CALLS_PER_REQUEST) {
        functionResponses.push({
          functionResponse: {
            ...(call.id ? { id: call.id } : {}),
            name: call.name,
            response: { error: "Horizon Agent MCP call limit reached for this request." },
          },
        });
        continue;
      }

      const actualName = connection.aliases.get(call.name);
      if (!actualName) {
        functionResponses.push({
          functionResponse: {
            ...(call.id ? { id: call.id } : {}),
            name: call.name,
            response: { error: "Unknown Shopify Dev MCP tool alias." },
          },
        });
        continue;
      }

      status.callsMade += 1;
      const started = Date.now();
      try {
        const result = await callMcpToolWithRetry(
          connection.client,
          actualName,
          call.args || {},
          timeoutMs
        );
        console.info("SHOPIFY DEV MCP TOOL COMPLETED", {
          tool: actualName,
          durationMs: Date.now() - started,
          isError: Boolean(result?.isError),
        });
        functionResponses.push({
          functionResponse: {
            ...(call.id ? { id: call.id } : {}),
            name: call.name,
            response: result?.isError
              ? { error: makeMcpOutput(result) }
              : { output: makeMcpOutput(result) },
          },
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        status.lastError = message.slice(0, 500);
        console.error("SHOPIFY DEV MCP TOOL FAILED", {
          tool: actualName,
          durationMs: Date.now() - started,
          error: status.lastError,
        });
        functionResponses.push({
          functionResponse: {
            ...(call.id ? { id: call.id } : {}),
            name: call.name,
            response: { error: status.lastError },
          },
        });
      }
    }

    if (Array.isArray(contents)) {
      contents.push({ role: "user", parts: functionResponses });
    }

    response = await generateContent({
      ...request,
      contents,
      config: {
        ...(request.config || {}),
        tools: [{ functionDeclarations: declarations }],
        toolConfig: { functionCallingConfig: { mode: "AUTO" } },
      },
    });
  }

  // If Gemini keeps requesting tools, ask once for a final user-facing response
  // without exposing more tool calls in this turn.
  const finalConfig = { ...(request.config || {}) };
  delete finalConfig.tools;
  delete finalConfig.toolConfig;
  response = await generateContent({
    ...request,
    contents,
    config: finalConfig,
  });

  return { response, status };
}
