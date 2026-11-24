# Horizon AI — Shopify Development Agent

Horizon AI is the existing Next.js + Gemini-powered Shopify development agent. It retains the Horizon theme source reader, architecture/dependency analysis, evidence verification, and runtime test runner. The Shopify integration extends this same Gemini orchestration; it does not replace Gemini or create a second agent.

## Current architecture

- **Gemini orchestration:** `src/app/api/agent/route.ts` remains the primary API route and continues to use the existing `@google/genai` client and retry handling.
- **Horizon theme analysis:** the existing source discovery, Liquid/JSON/JavaScript/CSS retrieval, dependency tracing, source evidence verification, and response formatting remain in place.
- **Shopify skill loader:** `src/lib/shopify/skill-loader.ts` selects Shopify AI Toolkit topics from the user's prompt and loads only the relevant official references from `shopify-toolkit/skills/shopify`.
- **Shopify Dev MCP bridge:** `src/lib/shopify/dev-mcp.ts` starts the official Shopify Dev MCP server over stdio, discovers its tools at runtime, maps the schemas to Gemini function declarations, executes Gemini-requested MCP tools, and returns results to Gemini for follow-up reasoning.
- **Knowledge fallback:** Shopify API/documentation tasks that do not match a local Horizon theme file still run through the same Gemini API with selected official Shopify skill references and MCP tools (when enabled).

## Shopify AI Toolkit content

This repository vendors the consolidated Shopify skill and its Markdown topic references from Shopify's official [Shopify AI Toolkit](https://github.com/Shopify/Shopify-AI-Toolkit). The copied upstream skill metadata reports version **1.20.0**. The content is under the MIT license; see `shopify-toolkit/LICENSE`.

The loader routes tasks into relevant topics such as Admin GraphQL, Liquid, Shopify Functions, metafields/metaobjects, Polaris surfaces, Shopify CLI, Customer Account API, Flow, Hydrogen, Storefront GraphQL, ShopifyQL, and app development. It does not install or assume compatibility with Claude/Codex/Cursor plugins. Their host-specific shell hooks are not part of this Gemini server integration.

## Setup

Requirements:

- Node.js 20 or newer.
- A Gemini API key.
- `npx` available on the machine that runs the Next.js server.
- A persistent Node.js host for local stdio MCP. Serverless platforms do not enable local stdio MCP by default.

1. Install dependencies and refresh the lockfile:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env.local` and set `GEMINI_API_KEY`.

3. Enable Shopify Dev MCP on a compatible self-hosted Node.js instance:

   ```env
   SHOPIFY_DEV_MCP_ENABLED=true
   SHOPIFY_DEV_MCP_PACKAGE=@shopify/dev-mcp@1.19.0
   ```

   The adapter launches the MCP server using `npx -y @shopify/dev-mcp@1.19.0` by default. It discovers available tools dynamically; tool names and JSON input schemas are not hard-coded.

4. Start the application:

   ```bash
   npm run dev
   ```

   Or use `npm run build` followed by `npm run start` for production.

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Existing Gemini API key | Required |
| `SHOPIFY_DEV_MCP_ENABLED` | Enable the Shopify Dev MCP process and Gemini tool bridge | `false` |
| `SHOPIFY_DEV_MCP_PACKAGE` | npm package/version launched by npx | `@shopify/dev-mcp@1.19.0` |
| `SHOPIFY_DEV_MCP_COMMAND` | Override the executable used to start MCP | platform default |
| `SHOPIFY_DEV_MCP_ARGS` | Optional JSON array of executable arguments | `["-y","@shopify/dev-mcp@1.19.0"]` |
| `SHOPIFY_DEV_MCP_CONNECT_TIMEOUT_MS` | MCP startup timeout | `15000` |
| `SHOPIFY_DEV_MCP_TOOL_TIMEOUT_MS` | Per-tool timeout | `20000` |
| `SHOPIFY_DEV_MCP_ALLOW_SERVERLESS` | Explicitly allow stdio MCP on a serverless host | `false` |
| `TEST_RUNNER_SECRET` | Existing runtime test runner authorization | Optional |
| `GITHUB_TOKEN` | Existing runtime test log upload to GitHub | Optional |

The MCP child process receives a restricted environment allowlist. It is not passed `GEMINI_API_KEY`, GitHub credentials, database secrets, or Shopify OAuth tokens. The adapter sets Shopify Toolkit telemetry opt-out environment variables for the child process.

## What this integration currently provides

- Official Shopify AI Toolkit skill version and topic references dynamically loaded according to the request.
- Shopify Dev MCP stdio client using the Model Context Protocol SDK.
- Runtime MCP tool discovery via `listTools()`.
- Dynamic MCP-to-Gemini function declaration mapping.
- Multi-round Gemini function calling and MCP result handoff.
- Bounded tool execution (maximum 6 calls and 3 tool rounds per generation), tool timeouts, one retry for thrown MCP tool errors, and graceful fallback to Gemini without MCP if the service is unavailable.
- Shopify Toolkit reference files included in Next.js output file tracing for the agent route.

## Important boundaries / not yet implemented

**Shopify Dev MCP is not Shopify Admin OAuth.** Its tools provide Shopify developer documentation, API schema discovery, and supported code validation. They do not authorize access to a merchant's store or perform product/order/customer CRUD.

The following remain separate implementation phases and must not be assumed to work yet:

- Shopify app OAuth installation/callback, encrypted access-token storage, per-shop sessions, uninstall/revocation handling, and least-privilege scope approval.
- Admin GraphQL operation clients for product, order, fulfillment/refund, customer/B2B company, inventory, shipping, store settings, navigation, and theme publish operations.
- Explicit approval workflows, dry-run execution, audit storage, and role/permission checks for destructive or sensitive merchant operations.
- Automated Shopify CLI upload/preview/publish and OAuth-connected store preview. The dashboard can embed a real preview URL started separately with Shopify CLI.
- Full Figma OAuth sign-in, file browsing, comments, write operations, and team-level account management remain separate. Local design-to-code uses a server-side Figma personal access token with read-only `file_content:read` scope; the token is never sent to the browser.

Do not put merchant access tokens into MCP process variables. Admin API access must be built separately using Shopify OAuth and the exact scopes required for each supported operation. Refunds, cancellations, deletions, sensitive settings, and publishing require explicit approval.

## Automatic local theme editing

Storefront modification prompts sent through the dashboard are routed to the writable theme-edit engine. The engine scans the local themes/horizon source, selects relevant Liquid/JSON/JavaScript/CSS files, asks Gemini for exact source replacements, checks every replacement against the current file, validates supported syntax, and applies the complete patch to the local theme. If a source match is missing, ambiguous, stale, or fails validation, no files are written. Multi-file writes roll back if a later write fails.

Theme editing is deliberately limited to the persistent local repository under themes/horizon; it does not edit the Horizon app backend, store data, credentials, locale files, or Shopify Admin. Run Horizon AI with npm run dev in the local repository and keep shopify theme dev --path themes/horizon --store your-store.myshopify.com running. Shopify CLI syncs local theme changes to its preview. A deployed serverless instance cannot persist these workspace edits and returns an explicit error instead.

# Horizon AI — Shopify Development Agent

Horizon AI is the existing Next.js + Gemini-powered Shopify development agent. It retains the Horizon theme source reader, architecture/dependency analysis, evidence verification, and runtime test runner. The Shopify integration extends this same Gemini orchestration; it does not replace Gemini or create a second agent.

## Current architecture

- **Gemini orchestration:** `src/app/api/agent/route.ts` remains the primary API route and continues to use the existing `@google/genai` client and retry handling.
- **Horizon theme analysis:** the existing source discovery, Liquid/JSON/JavaScript/CSS retrieval, dependency tracing, source evidence verification, and response formatting remain in place.
- **Shopify skill loader:** `src/lib/shopify/skill-loader.ts` selects Shopify AI Toolkit topics from the user's prompt and loads only the relevant official references from `shopify-toolkit/skills/shopify`.
- **Shopify Dev MCP bridge:** `src/lib/shopify/dev-mcp.ts` starts the official Shopify Dev MCP server over stdio, discovers its tools at runtime, maps the schemas to Gemini function declarations, executes Gemini-requested MCP tools, and returns results to Gemini for follow-up reasoning.
- **Knowledge fallback:** Shopify API/documentation tasks that do not match a local Horizon theme file still run through the same Gemini API with selected official Shopify skill references and MCP tools (when enabled).

## Shopify AI Toolkit content

This repository vendors the consolidated Shopify skill and its Markdown topic references from Shopify's official [Shopify AI Toolkit](https://github.com/Shopify/Shopify-AI-Toolkit). The copied upstream skill metadata reports version **1.20.0**. The content is under the MIT license; see `shopify-toolkit/LICENSE`.

The loader routes tasks into relevant topics such as Admin GraphQL, Liquid, Shopify Functions, metafields/metaobjects, Polaris surfaces, Shopify CLI, Customer Account API, Flow, Hydrogen, Storefront GraphQL, ShopifyQL, and app development. It does not install or assume compatibility with Claude/Codex/Cursor plugins. Their host-specific shell hooks are not part of this Gemini server integration.

## Setup

Requirements:

- Node.js 20 or newer.
- A Gemini API key.
- `npx` available on the machine that runs the Next.js server.
- A persistent Node.js host for local stdio MCP. Serverless platforms do not enable local stdio MCP by default.

1. Install dependencies and refresh the lockfile:

   ```bash
   npm install
   ```

2. Copy `.env.example` to `.env.local` and set `GEMINI_API_KEY`.

3. Enable Shopify Dev MCP on a compatible self-hosted Node.js instance:

   ```env
   SHOPIFY_DEV_MCP_ENABLED=true
   SHOPIFY_DEV_MCP_PACKAGE=@shopify/dev-mcp@1.19.0
   ```

   The adapter launches the MCP server using `npx -y @shopify/dev-mcp@1.19.0` by default. It discovers available tools dynamically; tool names and JSON input schemas are not hard-coded.

4. Start the application:

   ```bash
   npm run dev
   ```

   Or use `npm run build` followed by `npm run start` for production.

### Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `GEMINI_API_KEY` | Existing Gemini API key | Required |
| `SHOPIFY_DEV_MCP_ENABLED` | Enable the Shopify Dev MCP process and Gemini tool bridge | `false` |
| `SHOPIFY_DEV_MCP_PACKAGE` | npm package/version launched by npx | `@shopify/dev-mcp@1.19.0` |
| `SHOPIFY_DEV_MCP_COMMAND` | Override the executable used to start MCP | platform default |
| `SHOPIFY_DEV_MCP_ARGS` | Optional JSON array of executable arguments | `["-y","@shopify/dev-mcp@1.19.0"]` |
| `SHOPIFY_DEV_MCP_CONNECT_TIMEOUT_MS` | MCP startup timeout | `15000` |
| `SHOPIFY_DEV_MCP_TOOL_TIMEOUT_MS` | Per-tool timeout | `20000` |
| `SHOPIFY_DEV_MCP_ALLOW_SERVERLESS` | Explicitly allow stdio MCP on a serverless host | `false` |
| `TEST_RUNNER_SECRET` | Existing runtime test runner authorization | Optional |
| `GITHUB_TOKEN` | Existing runtime test log upload to GitHub | Optional |

The MCP child process receives a restricted environment allowlist. It is not passed `GEMINI_API_KEY`, GitHub credentials, database secrets, or Shopify OAuth tokens. The adapter sets Shopify Toolkit telemetry opt-out environment variables for the child process.

## What this integration currently provides

- Official Shopify AI Toolkit skill version and topic references dynamically loaded according to the request.
- Shopify Dev MCP stdio client using the Model Context Protocol SDK.
- Runtime MCP tool discovery via `listTools()`.
- Dynamic MCP-to-Gemini function declaration mapping.
- Multi-round Gemini function calling and MCP result handoff.
- Bounded tool execution (maximum 6 calls and 3 tool rounds per generation), tool timeouts, one retry for thrown MCP tool errors, and graceful fallback to Gemini without MCP if the service is unavailable.
- Shopify Toolkit reference files included in Next.js output file tracing for the agent route.

## Important boundaries / not yet implemented

**Shopify Dev MCP is not Shopify Admin OAuth.** Its tools provide Shopify developer documentation, API schema discovery, and supported code validation. They do not authorize access to a merchant's store or perform product/order/customer CRUD.

The following remain separate implementation phases and must not be assumed to work yet:

- Shopify app OAuth installation/callback, encrypted access-token storage, per-shop sessions, uninstall/revocation handling, and least-privilege scope approval.
- Admin GraphQL operation clients for product, order, fulfillment/refund, customer/B2B company, inventory, shipping, store settings, navigation, and theme publish operations.
- Explicit approval workflows, dry-run execution, audit storage, and role/permission checks for destructive or sensitive merchant operations.
- Automated Shopify CLI upload/preview/publish and OAuth-connected store preview. The dashboard can embed a real preview URL started separately with Shopify CLI.
- Figma API integration, SRS ingestion, spreadsheet import, and production QA agents.

Do not put merchant access tokens into MCP process variables. Admin API access must be built separately using Shopify OAuth and the exact scopes required for each supported operation. Refunds, cancellations, deletions, sensitive settings, and publishing require explicit approval.

## Dashboard Figma workspace

The dashboard's right-hand workspace embeds Figma files using Figma Embed Kit 2.0. Paste a Figma share URL and select **Load Figma**. The browser preview supports Design, Prototype, FigJam, Slides, and Deck links. Design links are also checked against the Figma REST API so the dashboard distinguishes a visible iframe preview from a design that the Agent can actually read.

For Agent design-to-code, set `FIGMA_ACCESS_TOKEN` in the local `.env.local` file. Create a Figma personal access token under **Figma account Settings → Security → Personal access tokens** and grant the read-only `file_content:read` scope. The token is used only on the server and is never sent to the browser or included in Gemini prompts. Keep the token private and restart the Next.js server after adding it. The API connection check reports missing token, insufficient scope, invalid token, and inaccessible file errors separately.

When a design edit is requested, Horizon sends the active Figma Design URL to the server. The server reads the file's layer hierarchy and exports the selected homepage frame screenshot through the Figma REST API; both are provided to the theme-edit planner as design reference. If the URL has no `node-id`, the Agent selects the most likely homepage-sized frame by its name and dimensions. For the most reliable target, select the actual homepage frame in Figma and copy its link before loading it in Horizon.

This uses the Figma REST API's [personal access token authentication](https://developers.figma.com/docs/rest-api/personal-access-tokens/) and [file/image endpoints](https://developers.figma.com/docs/rest-api/file-endpoints/). The authenticated account must be able to view the file, and the token must include `file_content:read`. The embed viewer can still work when the Agent API connection is not configured, but in that case the Agent will not claim to have read the design or silently guess at the visual target.

## Dashboard Shopify theme preview

The dashboard Live Preview embeds a real storefront URL; it does not fabricate a preview from Liquid source files. To preview this repository's local theme, authenticate Shopify CLI and run this command from the repository root:

```bash
shopify theme dev --path themes/horizon --store your-store.myshopify.com
```

Paste the local preview URL printed by Shopify CLI (commonly `http://127.0.0.1:9292`) into the Live Preview panel. Keep the CLI process running while viewing the dashboard. The app and the browser must be able to reach that preview URL. Alternatively, set `NEXT_PUBLIC_SHOPIFY_THEME_PREVIEW_URL` before starting or building the app. Only a storefront URL belongs in this public variable; never put Shopify access tokens or secrets in it. The current repository does not yet implement Shopify Admin OAuth or automatic theme-preview process management.

## Existing runtime test runner

The existing `/api/test-runner` endpoint remains available. It requires `TEST_RUNNER_SECRET` and `GITHUB_TOKEN` to publish its test log. The Shopify knowledge/MCP route can also be tested directly through the normal chat API.

## Upstream references

- [Shopify AI Toolkit](https://github.com/Shopify/Shopify-AI-Toolkit)
- [Shopify AI Toolkit documentation](https://shopify.dev/docs/api/ai-toolkit)
- [Shopify Dev MCP npm package](https://www.npmjs.com/package/@shopify/dev-mcp)
- [Shopify Dev MCP setup](https://shopify.dev/docs/apps/build/devmcp)
- [Shopify Admin GraphQL API](https://shopify.dev/docs/api/admin-graphql)

The dashboard Live Preview embeds a real storefront URL; it does not fabricate a preview from Liquid source files. To preview this repository's local theme, authenticate Shopify CLI and run this command from the repository root:

```bash
shopify theme dev --path themes/horizon --store your-store.myshopify.com
```

Paste the local preview URL printed by Shopify CLI (commonly `http://127.0.0.1:9292`) into the Live Preview panel. Keep the CLI process running while viewing the dashboard. The app and the browser must be able to reach that preview URL. Alternatively, set `NEXT_PUBLIC_SHOPIFY_THEME_PREVIEW_URL` before starting or building the app. Only a storefront URL belongs in this public variable; never put Shopify access tokens or secrets in it. The current repository does not yet implement Shopify Admin OAuth or automatic theme-preview process management.

## Existing runtime test runner

The existing `/api/test-runner` endpoint remains available. It requires `TEST_RUNNER_SECRET` and `GITHUB_TOKEN` to publish its test log. The Shopify knowledge/MCP route can also be tested directly through the normal chat API.

## Upstream references

- [Shopify AI Toolkit](https://github.com/Shopify/Shopify-AI-Toolkit)
- [Shopify AI Toolkit documentation](https://shopify.dev/docs/api/ai-toolkit)
- [Shopify Dev MCP npm package](https://www.npmjs.com/package/@shopify/dev-mcp)
- [Shopify Dev MCP setup](https://shopify.dev/docs/apps/build/devmcp)
- [Shopify Admin GraphQL API](https://shopify.dev/docs/api/admin-graphql)
