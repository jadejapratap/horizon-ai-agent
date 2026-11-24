import { createHash, randomUUID } from "node:crypto";
import { Buffer } from "node:buffer";
import { lstat, mkdir, readFile, readdir, realpath, rename, stat, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { GoogleGenAI } from "@google/genai";
import ts from "typescript";
import { loadShopifySkillContext, resolveShopifyAgentProfile } from "@/lib/shopify/skill-loader";
import type { FigmaDesignContext } from "@/lib/figma-connection";

const THEME_ROOT = path.resolve(process.cwd(), "themes", "horizon");
const MODEL = "gemini-3.5-flash-lite";
const MAX_PROMPT_CHARS = 5000;
const MAX_FILES_IN_PLAN = 8;
const MAX_OPERATIONS_PER_FILE = 10;
const MAX_FILE_CHARS = 140000;
const MAX_SOURCE_CONTEXT_CHARS = 105000;
const MAX_SOURCE_FILE_CHARS = 18000;
const ALLOWED_EXTENSIONS = new Set([".liquid", ".json", ".js", ".css"]);
const STOP_WORDS = new Set(["the", "and", "for", "with", "want", "need", "please", "make", "show", "add", "create", "build", "implement", "change", "modify", "update", "remove", "delete", "replace", "style", "fix", "this", "that", "from", "into", "your", "our", "you", "can", "could", "would", "should", "have", "has", "are", "is", "to", "in", "on", "of", "it", "my", "me", "a", "an", "be", "by", "as", "at", "or", "if", "when", "only", "both", "all"]);
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

export type ThemeEditSummary = {
  applied: boolean;
  summary: string;
  files: Array<{ path: string; operations: number; reasons: string[] }>;
  assets: Array<{ path: string; nodeId: string; nodeName: string; mimeType: string; bytes: number; created: boolean }>;
  previewNote: string;
};

type SourceFile = {
  path: string;
  fullPath: string;
  content: string;
  sha256: string;
  score: number;
  excerpt: string;
  complete: boolean;
};
type EditOperation = { oldText: string; newText: string; reason?: string };
type ProposedFileChange = { path: string; create: boolean; content?: string; edits: EditOperation[] };
type EditPlan = { summary: string; changes: ProposedFileChange[] };
type PreparedChange = {
  path: string;
  fullPath: string;
  originalContent: string | null;
  nextContent: string;
  operations: number;
  reasons: string[];
};

export class ThemeEditEngineError extends Error {
  status: number;
  constructor(message: string, status = 422) {
    super(message);
    this.name = "ThemeEditEngineError";
    this.status = status;
  }
}

let editQueue: Promise<void> = Promise.resolve();

async function withThemeEditLock<T>(task: () => Promise<T>): Promise<T> {
  const previous = editQueue;
  let release!: () => void;
  editQueue = new Promise<void>((resolve) => { release = resolve; });
  await previous;
  try {
    return await task();
  } finally {
    release();
  }
}

function isInsideRoot(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative);
}

function normalizeThemePath(value: string): string {
  return value.replace(/\\/g, "/").replace(/^\.\/+/, "");
}

function validateThemePath(value: string): string {
  const normalized = normalizeThemePath(value);
  const segments = normalized.split("/");
  const extension = path.posix.extname(normalized).toLowerCase();
  if (
    !normalized || normalized.includes("\0") || path.posix.isAbsolute(normalized) ||
    segments.some((segment) => !segment || segment === "." || segment === "..") ||
    !ALLOWED_EXTENSIONS.has(extension) ||
    normalized === "config/settings_data.json" || normalized.startsWith("locales/")
  ) {
    throw new ThemeEditEngineError("Unsupported or protected theme path: " + value, 403);
  }
  const fullPath = path.resolve(THEME_ROOT, normalized);
  if (!isInsideRoot(THEME_ROOT, fullPath)) {
    throw new ThemeEditEngineError("Theme edit path escaped the Horizon theme directory.", 403);
  }
  return normalized;
}

async function collectThemeFiles(directory: string, root: string, files: string[], directories: Set<string>): Promise<void> {
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith(".") || entry.name === "locales") continue;
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      directories.add(path.relative(root, fullPath).replace(/\\/g, "/"));
      await collectThemeFiles(fullPath, root, files, directories);
    } else if (
      entry.isFile() &&
      ALLOWED_EXTENSIONS.has(path.extname(entry.name).toLowerCase()) &&
      path.relative(root, fullPath).replace(/\\/g, "/") !== "config/settings_data.json"
    ) {
      files.push(path.relative(root, fullPath).replace(/\\/g, "/"));
    }
  }
}

function getPromptTerms(prompt: string): string[] {
  return [...new Set((prompt.toLowerCase().match(/[a-z0-9_]+/g) || [])
    .filter((term) => term.length > 2 && !STOP_WORDS.has(term)))];
}

function countOccurrences(source: string, term: string, cap = 8): number {
  let count = 0;
  let offset = 0;
  while (count < cap) {
    const index = source.indexOf(term, offset);
    if (index < 0) break;
    count += 1;
    offset = index + term.length;
  }
  return count;
}

function scoreThemeFile(prompt: string, file: string, content: string, terms: string[]): number {
  const lowerPath = file.toLowerCase();
  const lowerContent = content.toLowerCase();
  let score = 0;
  for (const term of terms) {
    if (lowerPath.includes(term)) score += 16;
    score += Math.min(6, countOccurrences(lowerContent, term, 6));
  }
  if (/product[\s-]?cards?/.test(prompt) && /product-card|card-gallery|product-grid|product-list|main-collection|price\.liquid/.test(lowerPath)) score += 55;
  if (/compare.?at|discount|sale price|percentage|price/.test(prompt)) {
    if (/price|product-card|product-badges/.test(lowerPath)) score += 45;
    if (lowerContent.includes("compare_at_price")) score += 35;
    if (lowerContent.includes("show_compare_price")) score += 18;
  }
  if (/badge|label|sale tag/.test(prompt) && /badge|product-card|card-gallery/.test(lowerPath)) score += 35;
  if (/header|navigation|menu|drawer/.test(prompt) && /header|navigation|menu|drawer/.test(lowerPath)) score += 45;
  if (/collection/.test(prompt) && /main-collection|collection|product-grid|product-list/.test(lowerPath)) score += 30;
  if (/homepage|home page/.test(prompt) && (lowerPath === "templates/index.json" || /product-list|featured-collection/.test(lowerPath))) score += 35;
  if (/cart drawer|cart/.test(prompt) && /cart/.test(lowerPath)) score += 40;
  if (/mobile|responsive/.test(prompt) && /base\.css|component|product-card/.test(lowerPath)) score += 12;
  return score;
}

function buildRelevantExcerpt(content: string, prompt: string, maxChars: number): { content: string; complete: boolean } {
  if (content.length <= maxChars) return { content, complete: true };
  const terms = [...getPromptTerms(prompt), "compare_at_price", "show_compare_price", "product-card", "price-item", "render", "stylesheet"];
  const positions: number[] = [];
  const lower = content.toLowerCase();
  for (const term of [...new Set(terms)]) {
    let offset = 0;
    let found = 0;
    while (found < 12) {
      const index = lower.indexOf(term.toLowerCase(), offset);
      if (index < 0) break;
      positions.push(index);
      offset = index + term.length;
      found += 1;
    }
  }
  positions.sort((a, b) => a - b);
  const ranges: Array<{ start: number; end: number }> = [];
  for (const position of positions) {
    const start = Math.max(0, position - 650);
    const end = Math.min(content.length, position + 1050);
    const previous = ranges[ranges.length - 1];
    if (previous && start <= previous.end + 120) {
      previous.end = Math.max(previous.end, end);
    } else {
      ranges.push({ start, end });
    }
  }
  let remaining = maxChars;
  const chunks: string[] = [];
  for (const range of ranges) {
    if (remaining <= 0) break;
    const chunk = content.slice(range.start, range.end).slice(0, remaining);
    if (chunk) {
      chunks.push("--- Exact source excerpt (character offsets " + range.start + "-" + (range.start + chunk.length) + ") ---\n" + chunk);
      remaining -= chunk.length;
    }
  }
  if (chunks.length === 0) chunks.push(content.slice(0, maxChars));
  return { content: chunks.join("\n\n"), complete: false };
}

async function discoverSourceContext(
  prompt: string,
  figmaContext?: FigmaDesignContext
): Promise<{ files: SourceFile[]; manifest: string[]; directories: string[] }> {
  const root = await realpath(THEME_ROOT);
  const manifest: string[] = [];
  const directories = new Set<string>();
  await collectThemeFiles(root, root, manifest, directories);
  if (manifest.length === 0) {
    throw new ThemeEditEngineError("No editable Shopify theme source files were found in themes/horizon.", 404);
  }

  const terms = getPromptTerms(prompt);
  const isHeroOrBannerRequest = /\b(hero|banner)\b/i.test(prompt);
  const figmaFrameLooksLikeHomepage = /\b(home|homepage|landing|storefront)\b/i.test(
    figmaContext?.nodeName ?? ""
  );
  const isHomepageCompositionRequest =
    /\b(homepage|home page|landing page|landing desktop|storefront home)\b/i.test(prompt) ||
    (isHeroOrBannerRequest && figmaFrameLooksLikeHomepage);
  const requiredHomepageFiles = isHomepageCompositionRequest
    ? new Set(["templates/index.json"])
    : new Set<string>();
  if (isHeroOrBannerRequest) {
    requiredHomepageFiles.add("sections/hero.liquid");
  }

  const results = await Promise.all(manifest.map(async (relativePath) => {
    const fullPath = path.resolve(root, relativePath);
    try {
      const metadata = await stat(fullPath);
      if (!metadata.isFile() || metadata.size > 250000) return null;
      const content = await readFile(fullPath, "utf8");
      const isRequiredSourceFile = requiredHomepageFiles.has(relativePath);
      const score = scoreThemeFile(prompt.toLowerCase(), relativePath, content, terms) +
        (isRequiredSourceFile ? 1000 : 0);
      if (score <= 0 && !isRequiredSourceFile) return null;
      return {
        path: relativePath,
        fullPath,
        content,
        sha256: createHash("sha256").update(content).digest("hex"),
        score,
      };
    } catch {
      return null;
    }
  }));

  const loaded: SourceFile[] = [];
  for (const result of results) {
    if (result) loaded.push({ ...result, excerpt: "", complete: false });
  }

  for (const requiredPath of requiredHomepageFiles) {
    if (!loaded.some((file) => file.path === requiredPath)) {
      throw new ThemeEditEngineError(
        "The required Shopify homepage composition file was not found or could not be read: " + requiredPath + ". No files were changed.",
        404
      );
    }
  }

  loaded.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path));

  const selected: SourceFile[] = [];
  let usedChars = 0;
  for (const candidate of loaded) {
    if (selected.length >= 12 || usedChars >= MAX_SOURCE_CONTEXT_CHARS) break;
    const budget = Math.min(MAX_SOURCE_FILE_CHARS, MAX_SOURCE_CONTEXT_CHARS - usedChars);
    if (budget < 1200) break;
    const excerpt = buildRelevantExcerpt(candidate.content, prompt, budget);
    candidate.excerpt = excerpt.content;
    candidate.complete = excerpt.complete;
    selected.push(candidate);
    usedChars += excerpt.content.length;
  }
  if (selected.length === 0) {
    throw new ThemeEditEngineError("I could not identify relevant editable theme source files. No files were changed.", 422);
  }
  return {
    files: selected,
    manifest: manifest.slice(0, 700),
    directories: [...directories].slice(0, 250),
  };
}

function parseModelJson(value: string): EditPlan {
  const trimmed = value.trim()
    .replace(/^\u0060{3}(?:json)?\s*/i, "")
    .replace(/\s*\u0060{3}$/, "");
  const start = trimmed.indexOf("{");
  const end = trimmed.lastIndexOf("}");
  if (start < 0 || end <= start) {
    throw new ThemeEditEngineError("Gemini did not return a structured theme edit plan.", 502);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed.slice(start, end + 1));
  } catch {
    throw new ThemeEditEngineError("Gemini returned invalid JSON for the theme edit plan. No files were changed.", 502);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new ThemeEditEngineError("Gemini returned an invalid theme edit plan.", 502);
  }
  const plan = parsed as Record<string, unknown>;
  if (typeof plan.summary !== "string" || !Array.isArray(plan.changes)) {
    throw new ThemeEditEngineError("Theme edit plan is missing summary or changes.", 502);
  }
  return { summary: plan.summary.slice(0, 1200), changes: plan.changes as ProposedFileChange[] };
}

function stripLeadingJsonComment(content: string): string {
  return content.replace(/^\uFEFF?\s*\/\*[\s\S]*?\*\/\s*/, "").trim();
}

function validateJsonContent(file: string, content: string): void {
  if (file.endsWith(".json")) {
    const parsed = JSON.parse(stripLeadingJsonComment(content)) as unknown;

    if (file === "templates/index.json" && parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      const sections = (parsed as Record<string, unknown>).sections;
      if (sections && typeof sections === "object" && !Array.isArray(sections)) {
        for (const [sectionId, value] of Object.entries(sections as Record<string, unknown>)) {
          if (!value || typeof value !== "object" || Array.isArray(value)) continue;
          const section = value as Record<string, unknown>;
          if (section.type !== "hero" || !section.settings || typeof section.settings !== "object" || Array.isArray(section.settings)) continue;
          const settings = section.settings as Record<string, unknown>;
          const rawHeight = settings.section_height_custom;
          const customHeight = typeof rawHeight === "number"
            ? rawHeight
            : typeof rawHeight === "string" && rawHeight.trim()
              ? Number(rawHeight)
              : NaN;
          if (Number.isFinite(customHeight) && (customHeight < 0 || customHeight > 100)) {
            throw new ThemeEditEngineError(
              "Shopify validation: hero section '" + sectionId + "' has section_height_custom=" +
                String(rawHeight) + ", but sections/hero.liquid allows only 0–100. No files were changed.",
              422
            );
          }
        }
      }
    }
    return;
  }
  if (file.endsWith(".liquid")) {
    const schemaPattern = /{%-?\s*schema\s*-?%}([\s\S]*?){%-?\s*endschema\s*-?%}/g;
    for (const match of content.matchAll(schemaPattern)) JSON.parse(match[1].trim());
  }
}

function validateLiquidBlockTags(file: string, content: string): void {
  if (!file.endsWith(".liquid")) return;
  const source = content
    .replace(/{%-?\s*comment\s*-?%}[\s\S]*?{%-?\s*endcomment\s*-?%}/g, "")
    .replace(/{%-?\s*raw\s*-?%}[\s\S]*?{%-?\s*endraw\s*-?%}/g, "");
  const blockNames = new Set(["if", "unless", "case", "for", "tablerow", "capture", "form", "paginate", "style", "stylesheet", "javascript", "schema", "doc"]);
  const stack: string[] = [];
  const tagPattern = /{%-?\s*([a-z_]+)\b[\s\S]*?-?%}/gi;
  for (const match of source.matchAll(tagPattern)) {
    const tag = match[1].toLowerCase();
    if (tag.startsWith("end")) {
      const closing = tag.slice(3);
      if (blockNames.has(closing)) {
        const opening = stack.pop();
        if (opening !== closing) {
          throw new Error("Liquid block mismatch in " + file + ": found end" + closing + " while expecting " + (opening ? "end" + opening : "an opening block") + ".");
        }
      }
      continue;
    }
    if (blockNames.has(tag)) stack.push(tag);
  }
  if (stack.length > 0) throw new Error("Liquid block is not closed in " + file + ": " + stack[stack.length - 1] + ".");
}

function validateCssStructure(file: string, content: string): void {
  if (!file.endsWith(".css")) return;
  const stack: string[] = [];
  let quote = "";
  let inComment = false;
  let escaped = false;
  for (let index = 0; index < content.length; index += 1) {
    const char = content[index];
    const next = content[index + 1];
    if (inComment) {
      if (char === "*" && next === "/") { inComment = false; index += 1; }
      continue;
    }
    if (quote) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === quote) quote = "";
      continue;
    }
    if (char === "/" && next === "*") { inComment = true; index += 1; continue; }
    if (char === "\"" || char === "'") { quote = char; continue; }
    if (char === "{" || char === "(" || char === "[") { stack.push(char); continue; }
    if (char === "}" || char === ")" || char === "]") {
      const expected = char === "}" ? "{" : char === ")" ? "(" : "[";
      if (stack.pop() !== expected) throw new Error("Unbalanced CSS delimiter in " + file + ".");
    }
  }
  if (quote || inComment || stack.length > 0) throw new Error("Unclosed CSS quote, comment, or block in " + file + ".");
}

function validateJavaScriptSyntax(file: string, content: string): void {
  if (!file.endsWith(".js")) return;
  const result = ts.transpileModule(content, {
    fileName: file,
    reportDiagnostics: true,
    compilerOptions: { allowJs: true, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  });
  const errors = (result.diagnostics || [])
    .filter((diagnostic) => diagnostic.category === ts.DiagnosticCategory.Error)
    .slice(0, 5)
    .map((diagnostic) => ts.flattenDiagnosticMessageText(diagnostic.messageText, " "));
  if (errors.length > 0) throw new Error("JavaScript syntax validation failed for " + file + ": " + errors.join(" | "));
}

function validateEditedContent(file: string, content: string): void {
  if (content.length > MAX_FILE_CHARS) {
    throw new ThemeEditEngineError("Edited file exceeds the " + MAX_FILE_CHARS + " character safety limit: " + file, 413);
  }
  try {
    validateJsonContent(file, content);
    validateLiquidBlockTags(file, content);
    validateCssStructure(file, content);
    validateJavaScriptSyntax(file, content);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new ThemeEditEngineError("Validation failed. No theme files were changed. " + message, 422);
  }
}

async function resolveSafeThemeTarget(relativePath: string, allowCreate: boolean): Promise<{ fullPath: string; exists: boolean }> {
  const normalized = validateThemePath(relativePath);
  const root = await realpath(THEME_ROOT);
  const fullPath = path.resolve(root, normalized);
  if (!isInsideRoot(root, fullPath)) throw new ThemeEditEngineError("Theme edit path is outside themes/horizon.", 403);

  const realParent = await realpath(path.dirname(fullPath));
  if (!isInsideRoot(root, realParent) && realParent !== root) {
    throw new ThemeEditEngineError("Theme edit directory resolves outside themes/horizon.", 403);
  }

  try {
    const metadata = await lstat(fullPath);
    if (metadata.isSymbolicLink() || !metadata.isFile()) {
      throw new ThemeEditEngineError("Theme edits are limited to regular files: " + normalized, 403);
    }
    if (metadata.size > 250000) throw new ThemeEditEngineError("Target theme file is too large to edit safely: " + normalized, 413);
    return { fullPath, exists: true };
  } catch (error) {
    if (error instanceof ThemeEditEngineError) throw error;
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "ENOENT") throw error;
    if (!allowCreate) throw new ThemeEditEngineError("Theme file no longer exists: " + normalized, 409);
    return { fullPath, exists: false };
  }
}

function applyExactEdits(file: string, original: string, edits: EditOperation[]): { content: string; reasons: string[] } {
  if (!Array.isArray(edits) || edits.length === 0 || edits.length > MAX_OPERATIONS_PER_FILE) {
    throw new ThemeEditEngineError("Invalid edit operation count for " + file + ".", 422);
  }
  let content = original;
  const reasons: string[] = [];
  for (const edit of edits) {
    if (!edit || typeof edit.oldText !== "string" || !edit.oldText || typeof edit.newText !== "string" || edit.oldText.length > 12000 || edit.newText.length > 20000) {
      throw new ThemeEditEngineError("An edit operation for " + file + " is missing a valid exact source/replacement.", 422);
    }
    const first = content.indexOf(edit.oldText);
    if (first < 0) {
      throw new ThemeEditEngineError("Patch did not match the current source in " + file + ". The source may have changed; no files were written.", 409);
    }
    if (content.indexOf(edit.oldText, first + edit.oldText.length) >= 0) {
      throw new ThemeEditEngineError("Patch is ambiguous because its exact source appears more than once in " + file + ". No files were written.", 409);
    }
    content = content.slice(0, first) + edit.newText + content.slice(first + edit.oldText.length);
    reasons.push(typeof edit.reason === "string" ? edit.reason.slice(0, 240) : "Applied source patch");
  }
  return { content, reasons };
}

async function atomicWrite(fullPath: string, content: string): Promise<void> {
  const temporaryPath = path.join(path.dirname(fullPath), ".horizon-ai-" + randomUUID() + ".tmp");
  try {
    await writeFile(temporaryPath, content, { encoding: "utf8", flag: "wx" });
    await rename(temporaryPath, fullPath);
  } catch (error) {
    await unlink(temporaryPath).catch(() => undefined);
    throw error;
  }
}

type PersistedFigmaAsset = {
  path: string;
  fullPath: string;
  nodeId: string;
  nodeName: string;
  mimeType: string;
  bytes: number;
  created: boolean;
};

async function persistFigmaImageAssets(context?: FigmaDesignContext): Promise<PersistedFigmaAsset[]> {
  if (!context?.assets?.length) return [];

  const root = await realpath(THEME_ROOT);
  const assetsDirectory = path.resolve(root, "assets");
  await mkdir(assetsDirectory, { recursive: true });
  const realAssetsDirectory = await realpath(assetsDirectory);
  if (!isInsideRoot(root, realAssetsDirectory)) {
    throw new ThemeEditEngineError("The Shopify theme assets directory resolves outside themes/horizon.", 403);
  }

  const persisted: PersistedFigmaAsset[] = [];
  let totalBytes = 0;
  try {
    for (const asset of context.assets) {
      if (
        !asset ||
        !/^figma-[0-9-]+-[a-f0-9]{10}\.(?:png|jpg|webp|gif|svg)$/i.test(asset.fileName) ||
        !/^image\/(?:png|jpeg|webp|gif|svg\+xml)$/i.test(asset.mimeType)
      ) {
        throw new ThemeEditEngineError("Figma returned an unsupported or unsafe image asset name/type. No theme files were changed.", 422);
      }

      const imageBuffer = Buffer.from(asset.imageData, "base64");
      if (imageBuffer.length === 0 || imageBuffer.length > 8 * 1024 * 1024) {
        throw new ThemeEditEngineError("Figma image asset is empty or exceeds the 8 MB per-image limit: " + asset.fileName, 413);
      }
      totalBytes += imageBuffer.length;
      if (totalBytes > 20 * 1024 * 1024) {
        throw new ThemeEditEngineError("Figma hero image assets exceed the 20 MB total import limit. No theme files were changed.", 413);
      }

      const fullPath = path.resolve(realAssetsDirectory, asset.fileName);
      if (!isInsideRoot(realAssetsDirectory, fullPath)) {
        throw new ThemeEditEngineError("Figma image asset path escaped the Shopify assets directory.", 403);
      }

      let created = false;
      try {
        const existing = await readFile(fullPath);
        const existingHash = createHash("sha256").update(existing).digest("hex");
        const incomingHash = createHash("sha256").update(imageBuffer).digest("hex");
        if (existingHash !== incomingHash) {
          throw new ThemeEditEngineError(
            "A different file already exists at " + asset.fileName + ". The existing theme asset was not overwritten.",
            409
          );
        }
      } catch (error) {
        if (error instanceof ThemeEditEngineError) throw error;
        const code = error && typeof error === "object"
          ? (error as NodeJS.ErrnoException).code
          : undefined;
        if (code !== "ENOENT") throw error;

        const temporaryPath = path.join(realAssetsDirectory, ".horizon-ai-figma-" + randomUUID() + ".tmp");
        try {
          await writeFile(temporaryPath, imageBuffer, { flag: "wx" });
          await rename(temporaryPath, fullPath);
          created = true;
        } catch (writeError) {
          await unlink(temporaryPath).catch(() => undefined);
          throw writeError;
        }
      }

      persisted.push({
        path: "assets/" + asset.fileName,
        fullPath,
        nodeId: asset.nodeId,
        nodeName: asset.nodeName,
        mimeType: asset.mimeType,
        bytes: imageBuffer.length,
        created,
      });
    }
  } catch (error) {
    await Promise.all(
      persisted.filter((asset) => asset.created).map((asset) => unlink(asset.fullPath).catch(() => undefined))
    );
    throw error;
  }

  return persisted;
}

async function rollbackCreatedFigmaAssets(assets: PersistedFigmaAsset[]): Promise<void> {
  await Promise.all(
    assets.filter((asset) => asset.created).map((asset) => unlink(asset.fullPath).catch((error) => {
      console.error("FIGMA ASSET ROLLBACK FAILED:", asset.path, error);
    }))
  );
}

async function applyPreparedChanges(changes: PreparedChange[]): Promise<void> {
  const written: PreparedChange[] = [];
  try {
    for (const change of changes) {
      if (change.originalContent !== null) {
        const latest = await readFile(change.fullPath, "utf8");
        if (latest !== change.originalContent) {
          throw new ThemeEditEngineError("The source changed while the patch was being prepared: " + change.path + ". No further files were changed.", 409);
        }
      } else {
        await stat(path.dirname(change.fullPath));
      }
      await atomicWrite(change.fullPath, change.nextContent);
      written.push(change);
    }
  } catch (error) {
    for (const change of written.reverse()) {
      try {
        if (change.originalContent === null) await unlink(change.fullPath);
        else await atomicWrite(change.fullPath, change.originalContent);
      } catch (rollbackError) {
        console.error("THEME EDIT ROLLBACK FAILED", {
          file: change.path,
          error: rollbackError instanceof Error ? rollbackError.message : String(rollbackError),
        });
      }
    }
    throw error;
  }
}

function makeEditPlanSchema(): Record<string, unknown> {
  return {
    type: "OBJECT",
    properties: {
      summary: { type: "STRING" },
      changes: {
        type: "ARRAY",
        items: {
          type: "OBJECT",
          properties: {
            path: { type: "STRING" },
            create: { type: "BOOLEAN" },
            content: { type: "STRING" },
            edits: {
              type: "ARRAY",
              items: {
                type: "OBJECT",
                properties: {
                  oldText: { type: "STRING" },
                  newText: { type: "STRING" },
                  reason: { type: "STRING" },
                },
                required: ["oldText", "newText", "reason"],
              },
            },
          },
          required: ["path", "create", "edits"],
        },
      },
    },
    required: ["summary", "changes"],
  };
}

async function generateEditPlan(
  prompt: string,
  history: any[],
  files: SourceFile[],
  manifest: string[],
  directories: string[],
  figmaContext?: FigmaDesignContext
): Promise<EditPlan> {
  const skillContext = await loadShopifySkillContext(prompt);
  const specialist = resolveShopifyAgentProfile(prompt);
  const boundedHistory = Array.isArray(history)
    ? history
        .filter((item: any) => item && typeof item.content === "string" && ["user", "assistant"].includes(item.role))
        .slice(-4)
        .map((item: any) => ({
          role: item.role === "assistant" ? "model" : "user",
          parts: [{ text: item.content.slice(-1800) }],
        }))
    : [];

  const fileContext = files.map((file) => ({
    path: file.path,
    sha256: file.sha256,
    complete: file.complete,
    source: file.excerpt,
  }));
  const requestText = [
    "USER REQUEST:",
    prompt,
    "",
    "ACTIVE SHOPIFY SPECIALIST:",
    specialist.name,
    specialist.instructions,
    "",
    "OFFICIAL SHOPIFY AI TOOLKIT REFERENCES:",
    skillContext.context.slice(0, 9000),
    "",
    "EDITABLE HORIZON THEME FILE MANIFEST:",
    JSON.stringify(manifest),
    "",
    "EXISTING THEME DIRECTORIES:",
    JSON.stringify(directories),
    "",
    "SOURCE FILES RETRIEVED FOR THIS REQUEST:",
    JSON.stringify(fileContext),
    "",
    "FIGMA DESIGN REFERENCE:",
    figmaContext
      ? [
          "File: " + figmaContext.fileName,
          "Homepage/frame: " + figmaContext.nodeName,
          "Node ID: " + figmaContext.nodeId,
          "The accompanying image is the actual Figma frame screenshot. Match its visible layout, visual hierarchy, colors, typography, spacing, and image placement.",
          "The following layer hierarchy and properties were extracted from the same frame:",
          figmaContext.summary,
        ].join("\n")
      : "No authenticated Figma design context was provided for this request.",
    "ORIGINAL FIGMA IMAGE ASSETS AVAILABLE IN themes/horizon/assets:",
    JSON.stringify((figmaContext?.assets ?? []).map((asset) => ({
      nodeId: asset.nodeId,
      nodeName: asset.nodeName,
      path: "assets/" + asset.fileName,
      fileName: asset.fileName,
      mimeType: asset.mimeType,
      width: asset.width,
      height: asset.height,
    }))),
  ].join("\n");

  const userParts: Array<Record<string, unknown>> = [{ text: requestText }];
  if (figmaContext) {
    userParts.push({
      inlineData: {
        mimeType: figmaContext.imageMimeType,
        data: figmaContext.imageData,
      },
    });
  }

  const response = await ai.models.generateContent({
    model: MODEL,
    contents: [...boundedHistory, { role: "user", parts: userParts }],
    config: {
      responseMimeType: "application/json",
      responseSchema: makeEditPlanSchema(),
      systemInstruction: [
        "You are Horizon AI, an autonomous Shopify Horizon theme code editor.",
        "Implement the user's requested storefront change in the actual supplied theme source. You are not in read-only analysis mode. Return a concrete edit plan, not an explanation-only answer.",
        "When FIGMA DESIGN REFERENCE is present, use its actual screenshot as the visual target and the layer hierarchy as supporting evidence. Recreate the design with native Shopify Liquid, CSS, and JavaScript; do not use the screenshot itself as a page background or substitute for implementation assets.",
        "When ORIGINAL FIGMA IMAGE ASSETS are listed, those original image files will be saved locally under themes/horizon/assets before the source patch is applied. Use the exact listed filenames and node names. Never use temporary Figma URLs. For the native hero section, use its image_1_asset and image_2_asset settings for these local files; do not put a theme asset filename into an image_picker setting. Preserve the Figma left/right image mapping, add meaningful alt text, and set image_1_width from the relative Figma frame widths (first image width divided by the sum of both image widths, rounded to a whole percent within the setting's 20–80 range).",
        "Understand behavior by tracing the supplied Liquid, CSS, JavaScript, JSON, render calls, symbols, and dependencies. Do not choose a file based only on its filename.",
        "For homepage and landing-page requests, treat templates/index.json as the Shopify page-composition entrypoint. Inspect its existing section instances and order, preserve stable section/block IDs where practical, and make only grounded changes to the supplied JSON and its inspected section implementations.",
        "The sections/hero.liquid schema caps section_height_custom at 100. Never place a value below 0 or above 100 in templates/index.json, even when the field is hidden by visible_if.",
        "Only edit existing source files included in SOURCE FILES RETRIEVED FOR THIS REQUEST, or create a new file inside an existing directory listed in EXISTING THEME DIRECTORIES.",
        "Never edit files outside themes/horizon. Never edit config/settings_data.json, locale files, credentials, environment files, or application/backend files.",
        "For an existing file, return create=false and one or more exact oldText/newText replacements. Every oldText must be copied byte-for-byte from the supplied source and be unique in the file at the time that operation is applied.",
        "For a new file, return create=true, content with the complete new file, and an empty edits array. Do not create a file if modifying an existing shared implementation is the correct solution.",
        "Do not return unchanged files. Do not return an operation with empty oldText. Do not use ellipses or placeholders in source text.",
        "Keep the patch minimal, preserve unrelated functionality and existing design, and implement all requested parts together across the minimum correct files.",
        "Use Shopify Liquid money filters and existing theme conventions. Do not invent product data or modify merchant data.",
        "For JSON files, preserve valid JSON and any existing Shopify header comment. For Liquid files, preserve valid Liquid block structure and schema JSON. For JS/CSS, preserve syntax and existing conventions.",
        "If the request cannot be implemented safely from the supplied source, return changes=[] and explain the concrete missing source in summary.",
        "Return only the JSON object required by the response schema. Do not claim that a file has been changed; the server applies your proposed edits.",
      ].join("\n"),
    },
  } as any);

  const responseText = response.text?.trim() || "";
  if (!responseText) throw new ThemeEditEngineError("Gemini returned an empty theme edit plan. No files were changed.", 502);
  return parseModelJson(responseText);
}

async function prepareChanges(plan: EditPlan, sourceFiles: SourceFile[], directories: string[]): Promise<PreparedChange[]> {
  if (!Array.isArray(plan.changes) || plan.changes.length === 0) {
    throw new ThemeEditEngineError("No safe code changes were proposed. " + (plan.summary || "No source files were changed."), 422);
  }
  if (plan.changes.length > MAX_FILES_IN_PLAN) {
    throw new ThemeEditEngineError("The edit plan exceeds the " + MAX_FILES_IN_PLAN + " file safety limit. No files were changed.", 413);
  }

  const sourceByPath = new Map<string, SourceFile>(sourceFiles.map((file) => [file.path, file] as const));
  const directorySet = new Set(directories);
  const seenPaths = new Set<string>();
  const prepared: PreparedChange[] = [];

  for (const proposed of plan.changes) {
    if (!proposed || typeof proposed.path !== "string" || typeof proposed.create !== "boolean") {
      throw new ThemeEditEngineError("Gemini returned a malformed file change. No files were changed.", 502);
    }
    const relativePath = validateThemePath(proposed.path);
    if (seenPaths.has(relativePath)) throw new ThemeEditEngineError("The edit plan repeats the same file: " + relativePath, 422);
    seenPaths.add(relativePath);

    const target = await resolveSafeThemeTarget(relativePath, proposed.create);
    if (proposed.create) {
      const parentDirectory = path.posix.dirname(relativePath);
      if (!directorySet.has(parentDirectory)) throw new ThemeEditEngineError("New files may only be created in an existing theme directory: " + parentDirectory, 403);
      if (target.exists) throw new ThemeEditEngineError("The proposed new theme file already exists: " + relativePath, 409);
      if (typeof proposed.content !== "string" || !proposed.content.trim() || proposed.content.length > MAX_FILE_CHARS) {
        throw new ThemeEditEngineError("New file content is empty or exceeds the safety limit: " + relativePath, 422);
      }
      validateEditedContent(relativePath, proposed.content);
      prepared.push({
        path: relativePath,
        fullPath: target.fullPath,
        originalContent: null,
        nextContent: proposed.content,
        operations: 1,
        reasons: ["Create " + relativePath],
      });
      continue;
    }

    if (!target.exists) throw new ThemeEditEngineError("The target file does not exist: " + relativePath, 404);
    const source = sourceByPath.get(relativePath);
    if (!source) throw new ThemeEditEngineError("Gemini proposed editing a file that was not included in this task's inspected source context: " + relativePath, 403);

    const originalContent = await readFile(target.fullPath, "utf8");
    const currentHash = createHash("sha256").update(originalContent).digest("hex");
    if (currentHash !== source.sha256) {
      throw new ThemeEditEngineError("The source changed after inspection: " + relativePath + ". Re-run the request so the patch is based on the latest file.", 409);
    }

    const applied = applyExactEdits(relativePath, originalContent, proposed.edits);
    if (applied.content === originalContent) throw new ThemeEditEngineError("The edit plan made no effective change to " + relativePath + ".", 422);
    validateEditedContent(relativePath, applied.content);
    prepared.push({
      path: relativePath,
      fullPath: target.fullPath,
      originalContent,
      nextContent: applied.content,
      operations: proposed.edits.length,
      reasons: applied.reasons,
    });
  }
  return prepared;
}

function buildEditResponse(
  summary: string,
  files: PreparedChange[],
  assets: PersistedFigmaAsset[]
): ThemeEditSummary {
  return {
    applied: true,
    summary,
    files: files.map((file) => ({ path: file.path, operations: file.operations, reasons: file.reasons })),
    assets: assets.map((asset) => ({
      path: asset.path,
      nodeId: asset.nodeId,
      nodeName: asset.nodeName,
      mimeType: asset.mimeType,
      bytes: asset.bytes,
      created: asset.created,
    })),
    previewNote: "The local theme files and any referenced Figma image assets have been updated. Shopify CLI theme dev should sync the changed files; refresh the storefront preview if it does not reload automatically.",
  };
}

export function isThemeEditRequest(prompt: string): boolean {
  const value = prompt.trim();
  if (!value) return false;
  if (/^(?:how|why|what|where|when|which|explain|describe|analyze|analyse|review|inspect|find|show me|tell me how|can you explain|could you explain)\b/i.test(value)) return false;

  const action = "(?:add|create|build|implement|import|download|fetch|save|sync|change|modify|update|remove|delete|replace|rework|redesign|recreate|replicate|restyle|style|fix|make|display|hide|show|enable|disable|insert|adjust|move|increase|decrease)";
  const directRequest = new RegExp(
    "^(?:please\\s+)?" +
    "(?:" + action + "\\b|(?:can|could)\\s+you\\s+(?:please\\s+)?" + action + "\\b|i\\s+(?:want|need)\\s+you\\s+to\\s+" + action + "\\b)",
    "i"
  ).test(value);
  const actionAfterFeature = new RegExp(
    "\\b(?:product cards?|product grid|theme|storefront|homepage|home page|figma design|figma images?|design assets?|hero|header|footer|collection page|cart drawer|product page|navigation|menu|section|snippet|price|logo|button|banner)\\b[\\s\\S]{0,140}\\b" + action + "\\b",
    "i"
  ).test(value);
  const gujaratiAction = /(?:ઉમેરો|ઉમેર|બદલો|દૂર કરો|દેખાડો|છુપાવો|બનાવો)/.test(value);

  return directRequest || actionAfterFeature || gujaratiAction;
}

export async function executeThemeEditRequest(
  prompt: string,
  history: any[] = [],
  figmaContext?: FigmaDesignContext
): Promise<ThemeEditSummary> {
  if (process.env.VERCEL === "1" || process.env.AWS_LAMBDA_FUNCTION_NAME) {
    throw new ThemeEditEngineError(
      "Automatic theme file editing requires the persistent local Horizon repository. Run Horizon AI with npm run dev in the theme workspace; deployed serverless instances cannot persist Shopify theme file changes.",
      409
    );
  }
  if (prompt.length > MAX_PROMPT_CHARS) {
    throw new ThemeEditEngineError("The theme edit request is too long. Please split it into smaller changes.", 413);
  }

  return withThemeEditLock(async () => {
    const sourceContext = await discoverSourceContext(prompt, figmaContext);
    console.info("HORIZON THEME EDIT SOURCE DISCOVERY", {
      manifestFiles: sourceContext.manifest.length,
      selectedFiles: sourceContext.files.map((file) => ({ path: file.path, score: file.score, complete: file.complete, sourceCharacters: file.excerpt.length })),
    });

    const plan = await generateEditPlan(prompt, history, sourceContext.files, sourceContext.manifest, sourceContext.directories, figmaContext);
    console.info("HORIZON THEME EDIT PLAN", {
      summary: plan.summary,
      files: plan.changes.map((change) => ({ path: change.path, create: change.create, operations: change.edits?.length ?? 0 })),
    });

    const prepared = await prepareChanges(plan, sourceContext.files, sourceContext.directories);
    const persistedFigmaAssets = await persistFigmaImageAssets(figmaContext);
    try {
      await applyPreparedChanges(prepared);
    } catch (error) {
      await rollbackCreatedFigmaAssets(persistedFigmaAssets);
      throw error;
    }
    console.info("HORIZON THEME EDIT APPLIED", {
      files: prepared.map((change) => ({ path: change.path, operations: change.operations })),
      assets: persistedFigmaAssets.map((asset) => ({ path: asset.path, bytes: asset.bytes, created: asset.created })),
    });
    return buildEditResponse(plan.summary, prepared, persistedFigmaAssets);
  });
}
