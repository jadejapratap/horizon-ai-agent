// v37: adds iterative explicit-only chunk-budget utilization and strict #updateScrollState evidence verification; preserves v36 scope isolation.
// v50: verifies product-page quantity-break branch scope from contiguous source context and authorizes explicit verified caller-to-target render claims without requiring the caller path in the sentence.
import { NextRequest, NextResponse } from "next/server";
import { GoogleGenAI } from "@google/genai";
import { readFile, readdir, stat } from "fs/promises";
import path from "path";

// --------------------------------------------------
// CONFIGURATION
// --------------------------------------------------
// v22: verify default Liquid case branches from the operation's active branch,
// require complete outer product-card evidence for product-page quantity breaks,
// strip model-generated coverage headings, and expose traceability mismatch tokens.
// v28: prioritize schema-definition source files, retrieve exact global-setting JSON objects,
// suppress unrelated fallback chunks for narrow requests, and normalize generic "references" wording.
// v29: allocate small exact setting-object ranges as required requests, and restrict targeted
// setting retrieval to IDs explicitly named by the user's prompt when such IDs are present.

const THEME_DIR = path.resolve(process.cwd(), "themes", "horizon");

const MAX_FILES = 30;
const MAX_FILE_SIZE = 18000;

const MAX_CONTEXT_SIZE = 80000;
const INITIAL_CONTEXT_BUDGET = 52000;
const ON_DEMAND_CONTEXT_BUDGET = 28000;

const MAX_ON_DEMAND_FILES = 5;
const MAX_ON_DEMAND_FILE_SIZE = 26000;
const MAX_CHUNK_SIZE = 8000;


const MIN_DEPENDENCY_SCORE = 15;

const GEMINI_RETRY_ATTEMPTS = 3;
const GEMINI_RETRY_BASE_DELAY = 1000;

const MODEL = "gemini-3.5-flash-lite";

const ALLOWED_EXTENSIONS = [
  ".liquid",
  ".json",
  ".js",
  ".css",
];

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY || "",
});

// --------------------------------------------------
// TYPES
// --------------------------------------------------
type Dependency = {
  from: string;
  to: string;
  relationship: string;
  evidence: string;
  relevanceScore: number;
};

type VerifiedDependency = Dependency & {
  verified: boolean;
  verificationEvidence: string;

  // Actual section/block files that own the setting.
  settingOwners?: string[];
};

type ThemeFile = {
  file: string;
  content: string;
  score: number;
};

type AnswerClaim = {
  id: string;
  claim: string;
  file: string;
  // Gemini may provide several exact excerpts when one atomic claim needs
  // evidence from more than one source location. A string remains accepted
  // for compatibility with older model responses.
  evidence: string | string[];
};

type VerifiedAnswerClaim = AnswerClaim & {
  verified: boolean;
  verificationReason: string;
  verificationMode: "direct_source" | "dependency_relationship" | "unverified";
};

type DependencyCategory = "high" | "medium" | "low";

type ThemeContextResult = {
  context: string;
  filesIncludedInContext: string[];
  filesSkippedFromContext: string[];
  includedSourceContent: Map<string, string>;
  truncatedFiles: string[];
  filePriorities: Map<string, number>;
};

type SchemaSetting = {
  id: string;
  type?: string;
};

type GeminiRequest = {
  model: string;
  contents: any;
  config?: Record<string, unknown>;
};


type RetrievedRange = {
  startOffset: number;
  endOffset: number;
};

type FileRetrievalState = {
  file: string;
  totalLength: number;
  retrievedRanges: RetrievedRange[];
  nextOffset: number;
  hasMore: boolean;
  isComplete: boolean;
};

type RetrievedChunk = {
  file: string;
  content: string;
  startOffset: number;
  endOffset: number;
  totalLength: number;
  hasMore: boolean;
  isComplete: boolean;
};

type SharedRetrievalState = {
  totalBudget: number;
  remainingBudget: number;
  usedBudget: number;
  maxOnDemandFiles: number;
  uniqueFiles: Set<string>;
  fileStates: Map<string, FileRetrievalState>;
  retrievedChunks: RetrievedChunk[];
  fullContents: Map<string, string>;
  processedDependencyRanges: Map<string, RetrievedRange[]>;
  // When present, every content read and chunk retrieval is confined to this allowlist.
  allowedFiles?: Set<string> | null;
};

type PlannerChunkRequest = {
  file: string;
  start_offset: number | null;
  max_chars: number;
  // Exact source-object retrievals must not be dropped by the generic tiny-fragment threshold.
  must_retrieve?: boolean;
};

// --------------------------------------------------
// GEMINI RETRY MANAGER
// --------------------------------------------------

function getGeminiErrorStatus(error: unknown): number | null {
  if (!error || typeof error !== "object") {
    return null;
  }

  const item = error as Record<string, unknown>;

  const status =
    item.status ??
    item.statusCode ??
    item.code;

  if (typeof status === "number") {
    return status;
  }

  if (typeof status === "string") {
    const parsed = Number(status);

    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  return null;
}

function isRetryableGeminiError(error: unknown): boolean {
  const status = getGeminiErrorStatus(error);

  if (
    status === 429 ||
    status === 500 ||
    status === 502 ||
    status === 503 ||
    status === 504
  ) {
    return true;
  }

  const message =
    error instanceof Error
      ? error.message.toLowerCase()
      : String(error).toLowerCase();

  return (
    message.includes("high demand") ||
    message.includes("overloaded") ||
    message.includes("temporarily unavailable") ||
    message.includes("rate limit") ||
    message.includes("resource exhausted") ||
    message.includes("503") ||
    message.includes("429")
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function generateContentWithRetry(
  request: GeminiRequest
): Promise<any> {
  let lastError: unknown;

  for (
    let attempt = 1;
    attempt <= GEMINI_RETRY_ATTEMPTS;
    attempt++
  ) {
    try {
      console.log(
        `GEMINI REQUEST ATTEMPT ${attempt}/${GEMINI_RETRY_ATTEMPTS}`
      );

      return await ai.models.generateContent(request as any);
    } catch (error) {
      lastError = error;

      const retryable = isRetryableGeminiError(error);

      console.error(
        "GEMINI REQUEST FAILED:",
        {
          attempt,
          retryable,
          status: getGeminiErrorStatus(error),
          message:
            error instanceof Error
              ? error.message
              : String(error),
        }
      );

      if (
        !retryable ||
        attempt === GEMINI_RETRY_ATTEMPTS
      ) {
        break;
      }

      const delay =
        GEMINI_RETRY_BASE_DELAY *
        Math.pow(2, attempt - 1);

      console.log(
        `Retrying Gemini request in ${delay}ms`
      );

      await sleep(delay);
    }
  }

  throw lastError;
}

// --------------------------------------------------
// PATH HELPERS
// --------------------------------------------------

function normalizePath(file: string): string {
  return file
    .replace(/\\/g, "/")
    .replace(/^\/+/, "");
}



function isSafeThemePath(file: string): boolean {
  const normalized = normalizePath(file);

  if (
    !normalized ||
    normalized.includes("\0")
  ) {
    return false;
  }

  const fullPath = path.resolve(
    THEME_DIR,
    normalized
  );

  const relativePath = path.relative(
    THEME_DIR,
    fullPath
  );

  return (
    relativePath !== "" &&
    !relativePath.startsWith("..") &&
    !path.isAbsolute(relativePath)
  );
}

async function readThemeFile(file: string): Promise<string> {
  const normalized = normalizePath(file);

  if (!isSafeThemePath(normalized)) {
    throw new Error("Invalid theme file path");
  }

  return readFile(
    path.resolve(THEME_DIR, normalized),
    "utf8"
  );
}

async function fileExists(file: string): Promise<boolean> {
  try {
    const normalized = normalizePath(file);

    if (!isSafeThemePath(normalized)) {
      return false;
    }

    const fileStat = await stat(
      path.resolve(THEME_DIR, normalized)
    );

    return fileStat.isFile();
  } catch {
    return false;
  }
}

// --------------------------------------------------
// EXPLICIT USER-REQUESTED FILE SCOPE (v35)
// --------------------------------------------------

type ExplicitFileRequest = {
  enabled: boolean;
  detectedPaths: string[];
  canonicalPaths: string[];
  invalidPaths: Array<{ path: string; reason: string }>;
};

function resolveExplicitFileRequest(
  prompt: string,
  themeFileManifest: string[]
): ExplicitFileRequest {
  // Match paths from supported Horizon theme roots, including traversal
  // attempts. Both slash styles are accepted and normalized afterwards.
  const pathPattern = /((?:(?:\.\.?)[\\/])*(?:(?:assets|blocks|config|layout|sections|snippets|templates)[\\/][A-Za-z0-9_@-]+(?:[\\/][A-Za-z0-9_@-]+)*\.(?:liquid|json|js|css)|(?:(?:\.\.?)[\\/]){1,}[A-Za-z0-9_@-]+(?:[\\/][A-Za-z0-9_@-]+)*\.(?:liquid|json|js|css)))/gi;
  const occurrences: Array<{ path: string; index: number }> = [];
  const detected = new Set<string>();

  for (const match of prompt.matchAll(pathPattern)) {
    const candidate = (match[1] ?? match[0])
      .trim()
      .replace(/^[\s"'`([{<]+/, "")
      .replace(/[\s"'`),;.!?}>\]]+$/, "")
      .replace(/\\/g, "/");
    if (!candidate) continue;
    detected.add(candidate);
    occurrences.push({ path: candidate, index: match.index ?? 0 });
  }

  const scopeRestriction = /\b(?:only|just|exactly|exclusively)\b/i;
  const hasDirectFileAnalysisIntent = occurrences.some(({ index, path: candidate }) => {
    const beforePath = prompt.slice(Math.max(0, index - 100), index);
    const afterPath = prompt.slice(index + candidate.length, Math.min(prompt.length, index + candidate.length + 100));
    const localContext = `${beforePath} ${afterPath}`;
    const directFileAction =
      /\b(?:inspect|analy[sz]e|review|explain|describe|read|open|trace|debug|understand|summari[sz]e|check|examine|study|investigate|compare|diff|look\s+at|go\s+through|use)\b(?:\s+[A-Za-z0-9_#.-]+){0,5}\s*$/i.test(beforePath) ||
      /\b(?:file|source|code|method|function|class|path)\b.{0,45}$/i.test(beforePath);
    return directFileAction || scopeRestriction.test(localContext);
  });
  const enabled = detected.size > 0 && hasDirectFileAnalysisIntent;

  if (!enabled) {
    return {
      enabled: false,
      detectedPaths: [...detected],
      canonicalPaths: [],
      invalidPaths: [],
    };
  }

  const canonicalByLowerPath = new Map(
    themeFileManifest.map((file) => [normalizePath(file).toLowerCase(), normalizePath(file)])
  );
  const canonicalPaths = new Set<string>();
  const invalidPaths: Array<{ path: string; reason: string }> = [];

  for (const candidate of detected) {
    const normalized = normalizePath(candidate).replace(/^(?:\.\/)+/, "");

    if (!isSafeThemePath(normalized)) {
      invalidPaths.push({ path: candidate, reason: "Unsafe path: traversal or path outside the Horizon theme." });
      continue;
    }

    const root = normalized.split("/")[0]?.toLowerCase();
    if (!["assets", "blocks", "config", "layout", "sections", "snippets", "templates"].includes(root)) {
      invalidPaths.push({ path: candidate, reason: "Unsupported theme path root." });
      continue;
    }

    if (!ALLOWED_EXTENSIONS.includes(path.extname(normalized).toLowerCase())) {
      invalidPaths.push({ path: candidate, reason: "Unsupported theme file extension." });
      continue;
    }

    const canonical = canonicalByLowerPath.get(normalized.toLowerCase());
    if (!canonical) {
      invalidPaths.push({ path: candidate, reason: "File does not exist in the Horizon theme manifest." });
      continue;
    }

    canonicalPaths.add(canonical);
  }

  return {
    enabled: true,
    detectedPaths: [...detected],
    canonicalPaths: [...canonicalPaths],
    invalidPaths,
  };
}

function isPathAllowedByScope(
  file: string,
  allowedFiles?: Set<string> | null
): boolean {
  if (!allowedFiles) return true;
  const normalized = normalizePath(file).toLowerCase();
  for (const allowed of allowedFiles) {
    if (normalizePath(allowed).toLowerCase() === normalized) return true;
  }
  return false;
}

// --------------------------------------------------
// FIND NEXT AVAILABLE GAP
// --------------------------------------------------

function findNextGap(
  ranges: RetrievedRange[],
  start: number,
  totalLength: number
): number {
  if (
    !Number.isFinite(totalLength) ||
    totalLength <= 0
  ) {
    return 0;
  }

  if (!Number.isFinite(start)) {
    start = 0;
  }

  let cursor = Math.max(
    0,
    Math.min(Math.floor(start), totalLength)
  );

  if (cursor >= totalLength) {
    return totalLength;
  }

  const sorted = mergeRanges(ranges);

  for (const range of sorted) {
    if (range.endOffset <= cursor) {
      continue;
    }

    if (range.startOffset > cursor) {
      return cursor;
    }

    cursor = Math.max(cursor, range.endOffset);

    if (cursor >= totalLength) {
      return totalLength;
    }
  }

  return cursor;
}

// --------------------------------------------------
// CHUNK RETRIEVAL — RANGE UTILITIES
// --------------------------------------------------

function mergeRanges(
  ranges: RetrievedRange[]
): RetrievedRange[] {
  if (ranges.length === 0) {
    return [];
  }

  const sorted = ranges
    .filter(
      range =>
        Number.isFinite(range.startOffset) &&
        Number.isFinite(range.endOffset) &&
        range.startOffset >= 0 &&
        range.endOffset > range.startOffset
    )
    .map(range => ({
      startOffset: Math.floor(range.startOffset),
      endOffset: Math.floor(range.endOffset),
    }))
    .filter(
      range =>
        range.endOffset > range.startOffset
    )
    .sort(
      (a, b) =>
        a.startOffset - b.startOffset
    );

  if (sorted.length === 0) {
    return [];
  }

  const merged: RetrievedRange[] = [
    { ...sorted[0] },
  ];

  for (let i = 1; i < sorted.length; i++) {
    const last = merged[merged.length - 1];
    const current = sorted[i];

    if (current.startOffset <= last.endOffset) {
      last.endOffset = Math.max(
        last.endOffset,
        current.endOffset
      );
    } else {
      merged.push({ ...current });
    }
  }

  return merged;
}

// --------------------------------------------------
// COMPUTE NEXT OFFSET
// --------------------------------------------------

function computeNextOffset(
  ranges: RetrievedRange[]
): number {
  if (ranges.length === 0) {
    return 0;
  }

  const sorted = mergeRanges(ranges);

  let cursor = 0;

  for (const range of sorted) {
    if (range.startOffset > cursor) {
      return cursor;
    }

    cursor = Math.max(
      cursor,
      range.endOffset
    );
  }

  return cursor;
}

function isRangeFullyRetrieved(
  ranges: RetrievedRange[],
  start: number,
  end: number
): boolean {
  return ranges.some(r => r.startOffset <= start && r.endOffset >= end);
}

// --------------------------------------------------
// SHARED RETRIEVAL STATE
// --------------------------------------------------

function createRetrievalState(
  initialUsed: number,
  onDemandBudget: number,
  maxOnDemandFiles: number,
  allowedFiles: Set<string> | null = null
): SharedRetrievalState {
  return {
    totalBudget: initialUsed + onDemandBudget,
    usedBudget: initialUsed,
    remainingBudget: onDemandBudget,
    maxOnDemandFiles,
    uniqueFiles: new Set(),
    fileStates: new Map(),
    retrievedChunks: [],
    fullContents: new Map(),
    processedDependencyRanges: new Map(),
    allowedFiles,
  };
}

async function ensureFullContent(
  state: SharedRetrievalState,
  file: string
): Promise<string | null> {
  const normalizedFile = normalizePath(file);
  if (!isPathAllowedByScope(normalizedFile, state.allowedFiles)) {
    console.warn("EXPLICIT FILE SCOPE BLOCKED CONTENT READ:", normalizedFile);
    return null;
  }
  if (state.fullContents.has(normalizedFile)) return state.fullContents.get(normalizedFile)!;
  try {
    const content = await readThemeFile(normalizedFile);
    state.fullContents.set(normalizedFile, content);
    return content;
  } catch {
    return null;
  }
}



// --------------------------------------------------
// CORE CHUNK RETRIEVAL (single budget, single source of truth)
// --------------------------------------------------

function retrieveChunk(
  state: SharedRetrievalState,
  filePath: string,
  fullContent: string,
  requestedStart: number | null,
  requestedMaxChars: number = MAX_CHUNK_SIZE
): RetrievedChunk | null {
  const totalLength = fullContent.length;

  // ----------------------------------------------
  // BASIC VALIDATION
  // ----------------------------------------------

  if (!filePath || totalLength === 0) return null;
  if (state.remainingBudget <= 0) return null;
  if (!isPathAllowedByScope(filePath, state.allowedFiles)) {
    console.warn("EXPLICIT FILE SCOPE BLOCKED CHUNK RETRIEVAL:", normalizePath(filePath));
    return null;
  }

  console.log("CHUNK RETRIEVAL REQUEST:", {
    file: filePath,
    requestedStart,
    requestedMaxChars,
    remainingBudget: state.remainingBudget,
  });

  // ----------------------------------------------
  // EXISTING FILE STATE
  // ----------------------------------------------

  const existingFileState = state.fileStates.get(filePath);

  // Prevent retrieving from a changed version of the file.
  if (
    existingFileState &&
    existingFileState.totalLength !== totalLength
  ) {
    console.error("CHUNK RETRIEVAL FILE LENGTH MISMATCH:", {
      file: filePath,
      existingLength: existingFileState.totalLength,
      currentLength: totalLength,
    });

    return null;
  }

  const retrievedRanges =
    existingFileState?.retrievedRanges ?? [];

  const desiredStart =
    requestedStart !== null &&
      Number.isFinite(requestedStart) &&
      Number.isInteger(requestedStart) &&
      requestedStart >= 0
      ? requestedStart
      : existingFileState?.nextOffset ?? 0;

  // ----------------------------------------------
  // FIND NEXT AVAILABLE GAP
  // ----------------------------------------------

  const startOffset = findNextGap(
    retrievedRanges,
    desiredStart,
    totalLength
  );

  if (
    !Number.isFinite(startOffset) ||
    startOffset < 0 ||
    startOffset >= totalLength
  ) {
    console.log("CHUNK RETRIEVAL SKIPPED:", {
      file: filePath,
      reason: "No available source range",
      desiredStart,
    });

    return null;
  }

  // ----------------------------------------------
  // PER-FILE RETRIEVAL LIMIT
  // ----------------------------------------------

  const alreadyRetrieved = state.retrievedChunks
    .filter(chunk => chunk.file === filePath)
    .reduce(
      (total, chunk) => total + chunk.content.length,
      0
    );

  const remainingFileCapacity =
    MAX_ON_DEMAND_FILE_SIZE - alreadyRetrieved;

  if (remainingFileCapacity <= 0) {
    console.log("CHUNK RETRIEVAL SKIPPED:", {
      file: filePath,
      reason: "Per-file retrieval limit reached",
      alreadyRetrieved,
    });

    return null;
  }

  // ----------------------------------------------
  // UNIQUE FILE LIMIT
  // ----------------------------------------------

  const isNewFile = !state.uniqueFiles.has(filePath);

  if (
    isNewFile &&
    state.uniqueFiles.size >= state.maxOnDemandFiles
  ) {
    console.log("CHUNK RETRIEVAL SKIPPED:", {
      file: filePath,
      reason: "Unique file limit reached",
      uniqueFiles: state.uniqueFiles.size,
      maxOnDemandFiles: state.maxOnDemandFiles,
    });

    return null;
  }

  // ----------------------------------------------
  // CHUNK SIZE
  // ----------------------------------------------

  const safeRequestedMaxChars =
    Number.isFinite(requestedMaxChars)
      ? Math.floor(requestedMaxChars)
      : MAX_CHUNK_SIZE;

  const maxChars = Math.max(
    1,
    Math.min(
      safeRequestedMaxChars,
      MAX_CHUNK_SIZE,
      remainingFileCapacity
    )
  );

  const size = Math.min(
    maxChars,
    state.remainingBudget,
    totalLength - startOffset
  );

  if (!Number.isFinite(size) || size <= 0) {
    console.log("CHUNK RETRIEVAL SKIPPED:", {
      file: filePath,
      reason: "No available retrieval budget",
      remainingBudget: state.remainingBudget,
    });

    return null;
  }

  // ----------------------------------------------
  // EXTRACT CHUNK
  // ----------------------------------------------

  const endOffset = startOffset + size;

  const content = fullContent.substring(
    startOffset,
    endOffset
  );

  const actual = content.length;

  if (actual === 0) return null;

  const actualEndOffset = startOffset + actual;

  // ----------------------------------------------
  // CREATE / UPDATE FILE STATE
  // ----------------------------------------------

  const fileState: FileRetrievalState =
    existingFileState ?? {
      file: filePath,
      totalLength,
      retrievedRanges: [],
      nextOffset: 0,
      hasMore: true,
      isComplete: false,
    };

  fileState.retrievedRanges = mergeRanges([
    ...fileState.retrievedRanges,
    {
      startOffset,
      endOffset: actualEndOffset,
    },
  ]);

  fileState.nextOffset = computeNextOffset(
    fileState.retrievedRanges
  );

  fileState.hasMore =
    fileState.nextOffset < totalLength;

  fileState.isComplete = !fileState.hasMore;

  state.fileStates.set(filePath, fileState);

  // ----------------------------------------------
  // REGISTER FILE
  // ----------------------------------------------

  if (isNewFile) {
    state.uniqueFiles.add(filePath);
  }

  if (!state.fullContents.has(filePath)) {
    state.fullContents.set(filePath, fullContent);
  }

  // ----------------------------------------------
  // UPDATE SHARED BUDGET
  // ----------------------------------------------

  state.remainingBudget -= actual;
  state.usedBudget += actual;

  // ----------------------------------------------
  // BUILD RESULT
  // ----------------------------------------------

  const chunk: RetrievedChunk = {
    file: filePath,
    content,
    startOffset,
    endOffset: actualEndOffset,
    totalLength,
    hasMore: fileState.hasMore,
    isComplete: fileState.isComplete,
  };

  state.retrievedChunks.push(chunk);

  // ----------------------------------------------
  // LOG RESULT
  // ----------------------------------------------

  console.log("CHUNK RETRIEVAL RESULT:", {
    file: chunk.file,
    startOffset: chunk.startOffset,
    endOffset: chunk.endOffset,
    totalLength: chunk.totalLength,
    contentLength: chunk.content.length,
    fileRetrievedCharacters: alreadyRetrieved + actual,
    fileRemainingCapacity: Math.max(
      0,
      MAX_ON_DEMAND_FILE_SIZE -
      alreadyRetrieved -
      actual
    ),
    hasMore: chunk.hasMore,
    isComplete: chunk.isComplete,
    remainingBudget: state.remainingBudget,
    usedBudget: state.usedBudget,
    uniqueFiles: state.uniqueFiles.size,
  });


  console.log("CHUNK OFFSET VALIDATION:", {
    file: filePath,
    totalLength,
    requestedStart,
    requestedMaxChars,
  });

  return chunk;
}

// --------------------------------------------------
// MANIFEST + DERIVED VIEWS
// --------------------------------------------------

function buildPlannerFileManifest(
  state: SharedRetrievalState,
  relevantFiles: ThemeFile[],
  dynamicFilePriorities: Map<string, number>
): {
  files: Array<{
    file: string;
    totalLength: number;
    priorityScore: number;
    includedInContext: boolean;
    retrievedRanges: RetrievedRange[];
    nextOffset: number;
    hasMore: boolean;
    isComplete: boolean;
  }>;
} {
  const priorityByFile = new Map<string, number>(
    relevantFiles.map((item) => [item.file, item.score])
  );

  // Dynamic priorities include search relevance, verified dependency
  // importance, relationship type, and graph connectivity.
  for (const [file, priority] of dynamicFilePriorities) {
    priorityByFile.set(
      file,
      Math.max(priorityByFile.get(file) ?? 0, priority)
    );
  }

  const files = Array.from(state.fullContents.entries())
    .filter(([file]) => isPathAllowedByScope(file, state.allowedFiles))
    .map(([file, content]) => {
      const fileState = state.fileStates.get(file);
      const retrievedRanges = mergeRanges(
        fileState?.retrievedRanges ?? []
      );

      const nextOffset = fileState
        ? computeNextOffset(retrievedRanges)
        : 0;

      const isComplete =
        fileState?.isComplete ?? false;

      return {
        file,
        totalLength: content.length,
        priorityScore: priorityByFile.get(file) ?? 0,
        includedInContext: retrievedRanges.length > 0,
        retrievedRanges,
        nextOffset,
        hasMore: nextOffset < content.length,
        isComplete,
      };
    })
    .sort((a, b) =>
      b.priorityScore - a.priorityScore || a.file.localeCompare(b.file)
    );

  return { files };
}

// --------------------------------------------------
// BUILD INCLUDED SOURCE CONTENT FROM SHARED STATE
// --------------------------------------------------

function buildIncludedSourceContentFromState(
  state: SharedRetrievalState
): Map<string, string> {
  const result = new Map<string, string>();

  for (const [file, fs] of state.fileStates.entries()) {
    if (!isPathAllowedByScope(file, state.allowedFiles)) continue;
    const full = state.fullContents.get(file);

    if (full === undefined) continue;

    const sorted = mergeRanges(fs.retrievedRanges)
      .map(range => ({
        startOffset: Math.max(0, range.startOffset),
        endOffset: Math.min(full.length, range.endOffset),
      }))
      .filter(
        range => range.endOffset > range.startOffset
      );

    const merged = sorted
      .map(range =>
        full.substring(
          range.startOffset,
          range.endOffset
        )
      )
      .join(
        "\n\n[UNRETRIEVED SOURCE RANGE GAP]\n\n"
      );

    result.set(file, merged);
  }

  return result;
}

// --------------------------------------------------
// DEPENDENCY EXPANSION — RANGE-AWARE
// --------------------------------------------------

function extractNewContentForDependencies(
  state: SharedRetrievalState,
  chunk: RetrievedChunk
): string {
  const processed =
    state.processedDependencyRanges.get(chunk.file) ?? [];

  const chunkRange: RetrievedRange = {
    startOffset: chunk.startOffset,
    endOffset: chunk.endOffset,
  };

  // ----------------------------------------------
  // VALIDATE CHUNK
  // ----------------------------------------------

  if (
    !chunk.file ||
    !Number.isInteger(chunkRange.startOffset) ||
    !Number.isInteger(chunkRange.endOffset) ||
    chunkRange.startOffset < 0 ||
    chunkRange.endOffset <= chunkRange.startOffset ||
    chunkRange.endOffset > chunk.totalLength ||
    chunk.content.length !==
    chunkRange.endOffset - chunkRange.startOffset
  ) {
    return "";
  }

  // ----------------------------------------------
  // FIND UNPROCESSED RANGES
  // ----------------------------------------------

  const newRanges: RetrievedRange[] = [];

  let cursor = chunkRange.startOffset;

  const sorted = mergeRanges(processed);

  for (const range of sorted) {
    if (range.endOffset <= cursor) {
      continue;
    }

    if (range.startOffset >= chunkRange.endOffset) {
      break;
    }

    if (range.startOffset > cursor) {
      newRanges.push({
        startOffset: cursor,
        endOffset: Math.min(
          range.startOffset,
          chunkRange.endOffset
        ),
      });
    }

    cursor = Math.max(cursor, range.endOffset);

    if (cursor >= chunkRange.endOffset) {
      break;
    }
  }

  if (cursor < chunkRange.endOffset) {
    newRanges.push({
      startOffset: cursor,
      endOffset: chunkRange.endOffset,
    });
  }

  if (newRanges.length === 0) {
    return "";
  }

  // ----------------------------------------------
  // EXTRACT ONLY NEW CONTENT
  // ----------------------------------------------

  const newContent = newRanges
    .map(range =>
      chunk.content.substring(
        range.startOffset - chunk.startOffset,
        range.endOffset - chunk.startOffset
      )
    )
    .filter(content => content.length > 0)
    .join("\n\n[UNPROCESSED SOURCE RANGE GAP]\n\n");

  // ----------------------------------------------
  // RETURN NEW CONTENT
  // ----------------------------------------------

  return newContent;
}

// --------------------------------------------------
// GET ALL THEME FILES
// --------------------------------------------------

async function getThemeFiles(
  directory = THEME_DIR,
  prefix = ""
): Promise<string[]> {
  const entries = await readdir(directory, {
    withFileTypes: true,
  });

  const files: string[] = [];

  for (const entry of entries) {
    if (
      [
        "node_modules",
        ".git",
        ".next",
        "locales",
      ].includes(entry.name)
    ) {
      continue;
    }

    const relativePath = prefix
      ? `${prefix}/${entry.name}`
      : entry.name;

    const fullPath = path.join(
      directory,
      entry.name
    );

    if (entry.isDirectory()) {
      const nestedFiles = await getThemeFiles(
        fullPath,
        relativePath
      );

      files.push(...nestedFiles);
      continue;
    }

    if (
      ALLOWED_EXTENSIONS.includes(
        path.extname(entry.name)
      )
    ) {
      files.push(normalizePath(relativePath));
    }
  }

  return files;
}

// --------------------------------------------------
// SEARCH TERMS
// --------------------------------------------------

function getSearchTerms(prompt: string): string[] {
  const terms = prompt
    .toLowerCase()
    .replace(/[^a-z0-9_\s-]/g, " ")
    .split(/\s+/)
    .filter((term) => term.length > 2);

  const expanded = new Set(terms);

  const synonyms: Record<string, string[]> = {
    header: [
      "header",
      "header-logo",
      "header-row",
      "header-menu",
    ],

    logo: [
      "logo",
      "logo_height",
      "logo_height_mobile",
      "inverse_logo",
      "header-logo",
    ],

    size: [
      "size",
      "height",
      "width",
      "font_size",
      "image_width",
      "image-size",
      "aspect-ratio",
    ],

    height: [
      "height",
      "logo_height",
      "logo_height_mobile",
      "image-height",
      "max-height",
    ],

    width: [
      "width",
      "image_width",
      "logo_width",
      "image-width",
      "max-width",
    ],

    mobile: [
      "mobile",
      "logo_height_mobile",
      "responsive",
    ],

    desktop: [
      "desktop",
      "logo_height",
      "header",
    ],

    product: [
      "product",
      "product-information",
      "product-details",
      "product-media",
      "product-card",
      "product-grid",
      "card-product",
      "card-gallery",
    ],

    card: [
      "card",
      "product-card",
      "card-product",
      "card-gallery",
      "card-media",
      "card-image",
    ],

    image: [
      "image",
      "image-ratio",
      "image_ratio",
      "image-size",
      "image-width",
      "image-height",
      "media",
      "aspect-ratio",
      "aspect_ratio",
      "object-fit",
      "object-position",
    ],

    collection: [
      "collection",
      "collection-grid",
      "collection-list",
      "collection-page",
      "main-collection",
      "product-grid",
      "product-card",
      "filters",
    ],

    grid: [
      "grid",
      "product-grid",
      "grid-item",
      "grid-template-columns",
      "grid-column",
      "grid-row",
    ],

    color: [
      "color",
      "background",
      "foreground",
      "text-color",
    ],

    font: [
      "font",
      "font-family",
      "font-size",
      "typography",
    ],

    spacing: [
      "spacing",
      "padding",
      "margin",
      "gap",
    ],

    cart: [
      "cart",
      "cart-drawer",
      "cart-items",
      "cart-total",
    ],

    menu: [
      "menu",
      "navigation",
      "header-menu",
      "mega-menu",
    ],
  };

  for (const term of terms) {
    for (const synonym of synonyms[term] || []) {
      expanded.add(synonym);
    }
  }

  return Array.from(expanded);
}

// --------------------------------------------------
// ARCHITECTURE INTENT + EXPLICIT ROOT FILES
// --------------------------------------------------

function isProductCardArchitectureIntent(prompt: string): boolean {
  const lower = prompt.toLowerCase();

  // A mention of "product cards" as a pricing display context is not
  // enough to switch the whole retrieval pipeline into product-card mode.
  // Require an explicit card-layout/component/gallery task instead.
  return (
    /\bcard[\s_-]*gallery\b/i.test(lower) ||
    /\bproduct[\s_-]*(?:grid|list)\b/i.test(lower) ||
    /\bproduct[\s_-]*cards?\s+(?:layout|gallery|component|block|architecture|design|images?|media|hover|quick[\s_-]*add|badges?)\b/i.test(lower) ||
    /\b(?:analy[sz]e|explain|review|inspect|trace|debug|modify|build|implement|update)\s+(?:the\s+)?(?:shopify\s+)?product[\s_-]*cards?\b/i.test(lower)
  );
}

function getArchitectureRootFiles(prompt: string): string[] {
  const roots: string[] = [];
  const domains = detectCoverageDomains(prompt);

  if (domains.includes("sticky-header")) {
    roots.push(
      "sections/header.liquid",
      "assets/base.css"
    );
  } else if (domains.includes("header")) {
    roots.push(
      "sections/header.liquid",
      "blocks/_header-menu.liquid",
      "snippets/header-drawer.liquid",
      "snippets/mega-menu-list.liquid",
      "snippets/overflow-list.liquid"
    );
  }

  if (domains.includes("pricing")) {
    roots.push(
      "snippets/price.liquid",
      "snippets/price-styles.liquid",
      "snippets/unit-price.liquid"
    );
  }

  if (isProductCardArchitectureIntent(prompt)) {
    roots.push(
      "sections/product-list.liquid",
      "blocks/_product-card.liquid",
      "blocks/_product-card-gallery.liquid",
      "snippets/card-gallery.liquid",
      "snippets/product-card.liquid",
      "snippets/product-media.liquid",
      "snippets/product-media-container-styles.liquid"
    );
  }

  return [...new Set(roots)];
}

// --------------------------------------------------
// SEARCH THEME
// --------------------------------------------------

async function searchTheme(
  prompt: string
): Promise<ThemeFile[]> {
  const files = await getThemeFiles();

  const terms = getSearchTerms(prompt);

  const promptLower = prompt.toLowerCase();

  const isProductCardIntent =
    isProductCardArchitectureIntent(prompt);

  const isProductImageIntent =
    isProductCardIntent &&
    /\bimage(s)?\b|\bmedia\b|\bphoto(s)?\b|\brender(ing)?\b/i.test(
      promptLower
    );

  const results: ThemeFile[] = [];

  for (const file of files) {
    try {
      const fullPath = path.resolve(
        THEME_DIR,
        file
      );

      const fileStat = await stat(fullPath);

      if (
        !fileStat.isFile() ||
        fileStat.size > 500000
      ) {
        continue;
      }

      const content = await readFile(
        fullPath,
        "utf8"
      );

      const filename = file.toLowerCase();

      const searchableContent =
        `${file}\n${content}`.toLowerCase();

      let score = 0;

      for (const term of terms) {
        if (filename.includes(term)) {
          score += 10;
        }

        const escapedTerm = term.replace(
          /[.*+?^${}()|[\]\\]/g,
          "\\$&"
        );

        const matches =
          searchableContent.match(
            new RegExp(escapedTerm, "g")
          ) || [];

        score += Math.min(matches.length, 10);
      }

      if (isProductCardIntent) {
        if (
          filename === "sections/product-list.liquid"
        ) {
          score += 300;
        }

        if (
          filename === "blocks/_product-card.liquid"
        ) {
          score += 180;
        }

        if (
          filename === "blocks/_product-card-gallery.liquid"
        ) {
          score += 200;
        }

        if (
          filename === "snippets/card-gallery.liquid"
        ) {
          score += 180;
        }

        if (
          filename === "snippets/product-media.liquid"
        ) {
          score += 130;
        }

        if (
          filename ===
          "snippets/product-media-container-styles.liquid"
        ) {
          score += 120;
        }

        if (
          filename.includes("header") ||
          filename.includes("filters")
        ) {
          score -= 120;
        }

        if (
          !isProductImageIntent &&
          filename ===
          "snippets/product-media-gallery-content.liquid"
        ) {
          score -= 100;
        }
      }

      if (score > 0) {
        results.push({
          file,
          content,
          score,
        });
      }
    } catch {
      continue;
    }
  }

  return results.sort(
    (a, b) => b.score - a.score
  );
}

// --------------------------------------------------
// DEPENDENCY RELEVANCE
// --------------------------------------------------

function calculateDependencyRelevance(
  prompt: string,
  from: string,
  to: string,
  relationship: string,
  evidence: string
): number {
  const words: string[] =
    prompt.toLowerCase().match(/[a-z_]+/g) ?? [];

  const stopWords = new Set([
    "i",
    "want",
    "to",
    "change",
    "update",
    "the",
    "a",
    "an",
    "in",
    "on",
    "for",
    "my",
    "is",
    "are",
    "how",
  ]);

  const contextWords = new Set([
    "header",
    "section",
    "theme",
    "page",
    "block",
  ]);

  const primaryTerms = [
    ...new Set(
      words.filter(
        (word) =>
          !stopWords.has(word) &&
          !contextWords.has(word)
      )
    ),
  ];

  const contextTerms = [
    ...new Set(
      words.filter((word) =>
        contextWords.has(word)
      )
    ),
  ];

  const targetText = to.toLowerCase();
  const sourceText = from.toLowerCase();
  const evidenceText = evidence.toLowerCase();

  let score = 0;

  if (
    relationship === "uses global setting" ||
    relationship === "uses section setting" ||
    relationship === "uses block setting" ||
    relationship === "defines setting"
  ) {
    score += 25;
  } else if (relationship === "renders snippet") {
    score += 18;
  } else if (
    relationship === "renders theme block" ||
    relationship === "renders section"
  ) {
    score += 15;
  }

  for (const term of primaryTerms) {
    if (targetText.includes(term)) {
      score += 20;
    }

    if (evidenceText.includes(term)) {
      score += 10;
    }

    if (sourceText.includes(term)) {
      score += 3;
    }
  }

  const promptHasLogo = words.includes("logo");

  const dependencyText = [
    from,
    to,
    evidence,
  ].join(" ").toLowerCase();

  if (promptHasLogo) {
    if (dependencyText.includes("logo")) {
      score += 20;
    } else {
      score -= 15;
    }
  }

  for (const term of contextTerms) {
    if (targetText.includes(term)) {
      score += 4;
    }

    if (evidenceText.includes(term)) {
      score += 2;
    }
  }

  return Math.max(0, Math.min(score, 100));
}

// --------------------------------------------------
// SCHEMA PARSING
// --------------------------------------------------

function extractSchemaJson(content: string): unknown | null {
  const match = content.match(
    /\{%[-]?\s*schema\s*[-]?%\}([\s\S]*?)\{%[-]?\s*endschema\s*[-]?%\}/i
  );

  if (!match) {
    return null;
  }

  try {
    return JSON.parse(match[1].trim());
  } catch {
    return null;
  }
}

function collectSchemaSettingIds(
  value: unknown,
  result = new Set<string>()
): Set<string> {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectSchemaSettingIds(item, result);
    }

    return result;
  }

  if (
    typeof value !== "object" ||
    value === null
  ) {
    return result;
  }

  const object = value as Record<string, unknown>;

  if (
    typeof object.id === "string" &&
    object.id.trim()
  ) {
    result.add(object.id);
  }

  for (const child of Object.values(object)) {
    if (
      typeof child === "object" &&
      child !== null
    ) {
      collectSchemaSettingIds(child, result);
    }
  }

  return result;
}

function verifySettingDefinition(
  settingId: string,
  schemaContent: string
): boolean {
  try {
    const parsed = JSON.parse(schemaContent);

    const settingIds = collectSchemaSettingIds(parsed);

    return settingIds.has(settingId);
  } catch {
    return false;
  }
}

// --------------------------------------------------
// SCHEMA BLOCK EXTRACTION
// --------------------------------------------------

function collectPresetBlockTypes(
  value: unknown,
  result: Set<string>
): void {
  if (Array.isArray(value)) {
    for (const item of value) {
      collectPresetBlockTypes(item, result);
    }

    return;
  }

  if (
    typeof value !== "object" ||
    value === null
  ) {
    return;
  }

  const object = value as Record<string, unknown>;

  if (
    typeof object.type === "string" &&
    object.type.startsWith("_")
  ) {
    result.add(object.type);
  }

  if (object.blocks) {
    collectPresetBlockTypes(
      object.blocks,
      result
    );
  }
}

function extractSchemaBlockDependencies(
  file: string,
  content: string,
  prompt: string
): Dependency[] {
  const dependencies: Dependency[] = [];

  const schema = extractSchemaJson(content);

  if (
    typeof schema !== "object" ||
    schema === null ||
    Array.isArray(schema)
  ) {
    return dependencies;
  }

  const schemaObject =
    schema as Record<string, unknown>;

  const blockTypes = new Set<string>();

  if (Array.isArray(schemaObject.presets)) {
    for (const preset of schemaObject.presets) {
      if (
        typeof preset === "object" &&
        preset !== null
      ) {
        const presetObject =
          preset as Record<string, unknown>;

        collectPresetBlockTypes(
          presetObject.blocks,
          blockTypes
        );
      }
    }
  }

  for (const blockType of blockTypes) {
    const evidence = `"type": "${blockType}"`;

    dependencies.push({
      from: file,
      to: `blocks/${blockType}.liquid`,
      relationship: "defines preset theme block",
      evidence,
      relevanceScore: calculateDependencyRelevance(
        prompt,
        file,
        `blocks/${blockType}.liquid`,
        "defines preset theme block",
        evidence
      ),
    });
  }

  return dependencies;
}


// --------------------------------------------------
// EXTRACT DEPENDENCIES
// --------------------------------------------------

function extractDependencies(
  file: string,
  content: string,
  prompt: string
): Dependency[] {
  const dependencies: Dependency[] = [];

  const add = (
    to: string,
    relationship: string,
    evidence: string
  ) => {
    if (to === file) return;

    const relevanceScore =
      calculateDependencyRelevance(
        prompt,
        file,
        to,
        relationship,
        evidence
      );

    dependencies.push({
      from: file,
      to,
      relationship,
      evidence: evidence.trim(),
      relevanceScore,
    });
  };

  const cleanContent = content
    .replace(
      /\{%-?\s*comment\s*-?%\}[\s\S]*?\{%-?\s*endcomment\s*-?%\}/g,
      ""
    )
    .replace(/<!--[\s\S]*?-->/g, "");

  // -------------------------------------
  // GLOBAL SETTINGS
  // -------------------------------------

  const globalSettings =
    /(?<![\w.])settings\.([a-zA-Z0-9_]+)/g;

  for (const match of cleanContent.matchAll(globalSettings)) {
    add(
      `setting:global.${match[1]}`,
      "uses global setting",
      match[0]
    );
  }

  // -------------------------------------
  // SECTION SETTINGS
  // -------------------------------------

  const sectionSettings =
    /\bsection\.settings\.([a-zA-Z0-9_]+)/g;

  for (const match of cleanContent.matchAll(sectionSettings)) {
    add(
      `setting:section.${match[1]}`,
      "uses section setting",
      match[0]
    );
  }

  // -------------------------------------
  // BLOCK SETTINGS
  // -------------------------------------

  const blockSettings =
    /\bblock\.settings\.([a-zA-Z0-9_]+)/g;

  for (const match of cleanContent.matchAll(blockSettings)) {
    add(
      `setting:block.${match[1]}`,
      "uses block setting",
      match[0]
    );
  }

  // -------------------------------------
  // BLOCK SETTINGS VIA block_settings
  // -------------------------------------

  const blockSettingsVariable =
    /\bblock_settings\.([a-zA-Z0-9_]+)/g;

  for (const match of cleanContent.matchAll(
    blockSettingsVariable
  )) {
    add(
      `setting:block.${match[1]}`,
      "uses block setting",
      match[0]
    );
  }

  // -------------------------------------
  // RENDERED SNIPPETS
  // -------------------------------------

  const renderRegex =
    /\{%-?\s*render\s+['"]([^'"]+)['"]/g;

  for (const match of cleanContent.matchAll(renderRegex)) {
    add(
      `snippets/${match[1]}.liquid`,
      "renders snippet",
      match[0]
    );
  }

  // -------------------------------------
  // THEME BLOCKS
  // -------------------------------------

  const blockRegex =
    /content_for\s+['"]block['"]\s*,\s*type:\s*['"]([^'"]+)['"]/g;

  for (const match of cleanContent.matchAll(blockRegex)) {
    add(
      `blocks/${match[1]}.liquid`,
      "renders theme block",
      match[0]
    );
  }

  // -------------------------------------
  // PRESET BLOCKS
  // -------------------------------------

  dependencies.push(
    ...extractSchemaBlockDependencies(
      file,
      cleanContent,
      prompt
    )
  );

  // -------------------------------------
  // SECTION REFERENCES
  // -------------------------------------

  const sectionRegex =
    /\{%-?\s*section\s+['"]([^'"]+)['"]/g;

  for (const match of cleanContent.matchAll(sectionRegex)) {
    add(
      `sections/${match[1]}.liquid`,
      "renders section",
      match[0]
    );
  }

  return uniqueDependencies(dependencies);
}
// --------------------------------------------------
// UNIQUE DEPENDENCIES
// --------------------------------------------------

function uniqueDependencies(
  dependencies: Dependency[]
): Dependency[] {
  const map = new Map<string, Dependency>();

  for (const dependency of dependencies) {
    const key = [
      dependency.from,
      dependency.to,
      dependency.relationship,
      dependency.evidence,
    ].join("|");

    const existing = map.get(key);

    if (
      !existing ||
      dependency.relevanceScore > existing.relevanceScore
    ) {
      map.set(key, dependency);
    }
  }

  return Array.from(map.values());
}

// --------------------------------------------------
// FIND SETTING OWNERS
// --------------------------------------------------

async function findSettingOwners(
  dependency: Dependency,
  allDependencies: Dependency[]
): Promise<string[]> {
  const settingType =
    dependency.relationship === "uses section setting"
      ? "section"
      : "block";

  const settingId = dependency.to.replace(
    `setting:${settingType}.`,
    ""
  );

  const ownerPrefix =
    settingType === "section"
      ? "sections/"
      : "blocks/";

  const visited = new Set<string>();

  const queue: string[] = [dependency.from];

  const owners = new Set<string>();

  while (queue.length > 0) {
    const current = queue.shift()!;

    if (visited.has(current)) {
      continue;
    }

    visited.add(current);

    // Direct owner may be the current source file.
    if (current.startsWith(ownerPrefix)) {
      try {
        const content = await readThemeFile(current);

        const schema = extractSchemaJson(content);

        if (
          schema &&
          collectSchemaSettingIds(schema).has(settingId)
        ) {
          owners.add(current);
        }
      } catch {
        // Ignore unreadable owner.
      }
    }

    // Find parent files that render the current file.
    const parents = allDependencies.filter(
      (item) =>
        item.to === current &&
        [
          "renders snippet",
          "renders theme block",
          "renders section",
        ].includes(item.relationship)
    );

    for (const parent of parents) {
      if (!visited.has(parent.from)) {
        queue.push(parent.from);
      }
    }
  }

  return [...owners];
}


// --------------------------------------------------
// VERIFY DEPENDENCY
// --------------------------------------------------

async function verifyDependency(
  dependency: Dependency,
  schemaContent?: string,
  allDependencies: Dependency[] = []
): Promise<VerifiedDependency> {
  const sourceExists = await fileExists(
    dependency.from
  );

  if (!sourceExists) {
    return {
      ...dependency,
      verified: false,
      verificationEvidence: "Source file does not exist",
    };
  }

  let sourceContent: string;

  try {
    sourceContent = await readThemeFile(
      dependency.from
    );
  } catch {
    return {
      ...dependency,
      verified: false,
      verificationEvidence: "Unable to read source file",
    };
  }

  // -------------------------------------
  // SETTING REFERENCES
  // -------------------------------------

  if (dependency.relationship.startsWith("uses ")) {
    const referenceExists =
      sourceContent.includes(dependency.evidence);

    if (!referenceExists) {
      return {
        ...dependency,
        verified: false,
        verificationEvidence:
          `Reference not found in source: ${dependency.evidence}`,
      };
    }

    // -------------------------------------
    // GLOBAL SETTINGS
    // -------------------------------------

    if (
      dependency.relationship === "uses global setting"
    ) {
      const settingId = dependency.to.replace(
        "setting:global.",
        ""
      );

      const definitionExists = schemaContent
        ? verifySettingDefinition(
          settingId,
          schemaContent
        )
        : false;

      return {
        ...dependency,
        verified: definitionExists,
        verificationEvidence: definitionExists
          ? `Global setting "${settingId}" verified in config/settings_schema.json`
          : `Global setting "${settingId}" was not found in config/settings_schema.json`,
      };
    }

    // -------------------------------------
    // SECTION / BLOCK SETTINGS
    // -------------------------------------

    const owners = await findSettingOwners(
      dependency,
      allDependencies
    );

    const verified = owners.length > 0;

    return {
      ...dependency,
      verified,
      settingOwners: owners,
      verificationEvidence: verified
        ? `Setting "${dependency.to}" verified in owner file(s): ${owners.join(", ")}`
        : `No reachable ${dependency.relationship === "uses section setting" ? "section" : "block"} owner schema defines setting "${dependency.to}"`,
    };
  }

  // -------------------------------------
  // GLOBAL SETTING DEFINITIONS
  // -------------------------------------

  if (dependency.relationship === "defines setting") {
    const settingId = dependency.to.replace(
      "setting:global.",
      ""
    );

    const definitionExists = schemaContent
      ? verifySettingDefinition(
        settingId,
        schemaContent
      )
      : false;

    const referenceExists =
      sourceContent.includes(dependency.evidence);

    const verified =
      definitionExists && referenceExists;

    return {
      ...dependency,
      verified,
      verificationEvidence: verified
        ? `Global setting "${settingId}" verified in config/settings_schema.json`
        : `Global setting "${settingId}" definition or source evidence was not verified`,
    };
  }

  // -------------------------------------
  // FILE RELATIONSHIPS
  // -------------------------------------

  const targetExists = await fileExists(
    dependency.to
  );

  const referenceExists =
    sourceContent.includes(dependency.evidence);

  const verified =
    targetExists && referenceExists;

  return {
    ...dependency,
    verified,
    verificationEvidence: verified
      ? `Exact reference verified in ${dependency.from}; target exists: ${dependency.to}`
      : !targetExists
        ? `Target file does not exist: ${dependency.to}`
        : `Exact reference not found in ${dependency.from}`,
  };
}

// --------------------------------------------------
// DEPENDENCY CLASSIFICATION
// --------------------------------------------------

function classifyDependency(
  dependency: VerifiedDependency
): DependencyCategory {
  if (dependency.relevanceScore >= 60) {
    return "high";
  }

  if (dependency.relevanceScore >= 35) {
    return "medium";
  }

  return "low";
}

// --------------------------------------------------
// FILTER RELEVANT DEPENDENCIES
// --------------------------------------------------

function filterRelevantDependencies(
  dependencies: VerifiedDependency[],
  initialFiles: ThemeFile[]
): VerifiedDependency[] {
  const rootPaths = new Set(
    initialFiles.map((file) => file.file)
  );

  const reachableFiles = new Set(rootPaths);

  let changed = true;

  // -------------------------------------
  // BUILD REACHABLE FILE GRAPH
  // -------------------------------------

  while (changed) {
    changed = false;

    for (const dependency of dependencies) {
      if (!dependency.verified) {
        continue;
      }

      const isFileRelationship = [
        "renders snippet",
        "renders theme block",
        "renders section",
        "defines preset theme block",
      ].includes(dependency.relationship);

      if (!isFileRelationship) {
        continue;
      }

      if (
        reachableFiles.has(dependency.from) &&
        !reachableFiles.has(dependency.to)
      ) {
        reachableFiles.add(dependency.to);
        changed = true;
      }
    }
  }

  // -------------------------------------
  // FILTER DEPENDENCIES
  // -------------------------------------

  return dependencies.filter((dependency) => {
    if (!dependency.verified) {
      return false;
    }

    // Preserve all root dependencies.
    if (rootPaths.has(dependency.from)) {
      return true;
    }

    // Preserve rendering relationships in reachable graph.
    if (
      [
        "renders snippet",
        "renders theme block",
        "renders section",
        "defines preset theme block",
      ].includes(dependency.relationship)
    ) {
      return reachableFiles.has(dependency.from);
    }

    // Preserve settings used by reachable files.
    if (
      dependency.relationship.startsWith("uses ")
    ) {
      return reachableFiles.has(dependency.from);
    }

    // Preserve definitions referenced by reachable files.
    if (dependency.relationship === "defines setting") {


      const referencedSetting = dependencies.some(
        (item) =>
          item.verified &&
          item.relationship.startsWith("uses ") &&
          item.to === dependency.to &&
          reachableFiles.has(item.from)
      );

      return referencedSetting;
    }

    return false;
  });
}

// --------------------------------------------------
// BUILD SETTING DEFINITION DEPENDENCIES
// --------------------------------------------------

function buildSettingDefinitionDependencies(
  dependencies: Dependency[],
  schemaContent: string,
  prompt: string
): Dependency[] {
  const result: Dependency[] = [];

  const settingIds = new Set<string>();

  for (const dependency of dependencies) {
    if (
      !dependency.to.startsWith("setting:global.")
    ) {
      continue;
    }

    const settingId = dependency.to.replace(
      "setting:global.",
      ""
    );

    if (settingId) {
      settingIds.add(settingId);
    }
  }

  for (const settingId of settingIds) {
    const exists = verifySettingDefinition(
      settingId,
      schemaContent
    );

    if (!exists) {
      continue;
    }

    const evidence = `"id": "${settingId}"`;

    result.push({
      from: "config/settings_schema.json",
      to: `setting:global.${settingId}`,
      relationship: "defines setting",
      evidence,
      relevanceScore: calculateDependencyRelevance(
        prompt,
        "config/settings_schema.json",
        `setting:global.${settingId}`,
        "defines setting",
        evidence
      ),
    });
  }

  return result;
}

// --------------------------------------------------
// FILTER RELEVANT THEME FILES
// --------------------------------------------------

function filterRelevantThemeFiles(
  files: ThemeFile[],
  initialFiles: ThemeFile[],
  dependencies: VerifiedDependency[],
  explicitFileAllowlist?: Set<string> | null
): ThemeFile[] {
  if (explicitFileAllowlist) {
    return files.filter((file) => isPathAllowedByScope(file.file, explicitFileAllowlist));
  }

  const relevantPaths = new Set<string>();

  for (const file of initialFiles) {
    relevantPaths.add(file.file);
  }

  relevantPaths.add("config/settings_schema.json");

  for (const dependency of dependencies) {
    if (!dependency.verified) {
      continue;
    }

    relevantPaths.add(dependency.from);

    if (
      dependency.to.startsWith("snippets/") ||
      dependency.to.startsWith("blocks/") ||
      dependency.to.startsWith("sections/")
    ) {
      relevantPaths.add(dependency.to);
    }
  }

  return files.filter((file) =>
    relevantPaths.has(file.file)
  );
}

// --------------------------------------------------
// BUILD THEME CONTEXT - FAIR MANDATORY ALLOCATION
// --------------------------------------------------

function buildThemeContext(
  files: ThemeFile[],
  dependencies: Dependency[],
  contextBudget = INITIAL_CONTEXT_BUDGET,
  mandatoryFilePaths: string[] = [],
  prompt = ""
): ThemeContextResult {
  let context = "";

  const filesIncludedInContext: string[] = [];
  const filesSkippedFromContext: string[] = [];

  const includedSourceContent =
    new Map<string, string>();

  const truncatedFiles: string[] = [];

  // --------------------------------------------------
  // EXPLICIT REQUEST-SCOPED MANDATORY FILES
  // --------------------------------------------------

  const availablePaths = new Set(
    files.map((file) => file.file)
  );

  // A file is mandatory only because the request explicitly selected it
  // as an architecture root, never merely because it happened to be loaded
  // as a related dependency.
  const mandatoryFiles = new Set(
    mandatoryFilePaths.filter((filePath) =>
      availablePaths.has(filePath)
    )
  );

  // --------------------------------------------------
  // PRIORITY MAPS
  // --------------------------------------------------

  const fileRelevance = new Map<string, number>();
  const fileRolePriority = new Map<string, number>();

  const incomingDependencies = new Map<string, number>();
  const outgoingDependencies = new Map<string, number>();

  // --------------------------------------------------
  // ANALYZE DEPENDENCIES
  // --------------------------------------------------

  for (const dep of dependencies) {
    fileRelevance.set(
      dep.from,
      Math.max(
        fileRelevance.get(dep.from) ?? 0,
        dep.relevanceScore
      )
    );

    outgoingDependencies.set(
      dep.from,
      (outgoingDependencies.get(dep.from) ?? 0) + 1
    );

    const dependencyTargetIsFile =
      dep.to.startsWith("snippets/") ||
      dep.to.startsWith("blocks/") ||
      dep.to.startsWith("sections/") ||
      dep.to.startsWith("config/");

    if (dependencyTargetIsFile) {
      fileRelevance.set(
        dep.to,
        Math.max(
          fileRelevance.get(dep.to) ?? 0,
          dep.relevanceScore
        )
      );

      incomingDependencies.set(
        dep.to,
        (incomingDependencies.get(dep.to) ?? 0) + 1
      );
    }

    // --------------------------------------------------
    // RELATIONSHIP PRIORITY
    // --------------------------------------------------

    let rolePriority = 0;

    if (dep.relationship === "renders theme block") {
      rolePriority = 180;
    } else if (
      dep.relationship === "defines preset theme block"
    ) {
      rolePriority = 160;
    } else if (
      dep.relationship === "renders snippet"
    ) {
      rolePriority = 80;
    } else if (
      dep.relationship === "uses block setting"
    ) {
      rolePriority = 60;
    } else if (
      dep.relationship === "uses global setting"
    ) {
      rolePriority = 40;
    } else if (
      dep.relationship === "defines setting"
    ) {
      rolePriority = isSettingsSchemaIntent(prompt) ? 420 : 30;
    }

    // A "defines setting" edge points from the schema file to a synthetic
    // setting identifier. The actual source file receiving definition
    // priority is therefore dependency.from, not dependency.to.
    const rolePriorityPath = dep.relationship === "defines setting"
      ? dep.from
      : dep.to;
    const rolePriorityPathIsFile =
      rolePriorityPath.startsWith("snippets/") ||
      rolePriorityPath.startsWith("blocks/") ||
      rolePriorityPath.startsWith("sections/") ||
      rolePriorityPath.startsWith("config/");

    if (rolePriorityPathIsFile) {
      fileRolePriority.set(
        rolePriorityPath,
        Math.max(
          fileRolePriority.get(rolePriorityPath) ?? 0,
          rolePriority
        )
      );
    }
  }

  // --------------------------------------------------
  // DYNAMIC FILE PRIORITY
  // --------------------------------------------------

  function getFilePriority(file: ThemeFile): number {
    const filePath = file.file;

    const searchScore = file.score;

    const dependencyScore =
      fileRelevance.get(filePath) ?? 0;

    const rolePriority =
      fileRolePriority.get(filePath) ?? 0;

    const incomingCount =
      incomingDependencies.get(filePath) ?? 0;

    const outgoingCount =
      outgoingDependencies.get(filePath) ?? 0;

    const incomingPriority =
      Math.min(incomingCount, 5) * 15;

    const outgoingPriority =
      Math.min(outgoingCount, 5) * 5;

    let pathPriority = 0;

    if (filePath.startsWith("blocks/")) {
      pathPriority = 35;
    } else if (filePath.startsWith("snippets/")) {
      pathPriority = 25;
    } else if (filePath.startsWith("sections/")) {
      pathPriority = 20;
    }

    if (
      filePath === "config/settings_schema.json"
    ) {
      pathPriority = 50;
    }

    const mandatoryPriority =
      mandatoryFiles.has(filePath)
        ? 10000
        : 0;

    return (
      mandatoryPriority +
      searchScore +
      dependencyScore * 1.5 +
      rolePriority +
      incomingPriority +
      outgoingPriority +
      pathPriority
    );
  }

  // --------------------------------------------------
  // SORT FILES
  // --------------------------------------------------

  const prioritizedFiles = [...files].sort(
    (a, b) =>
      getFilePriority(b) - getFilePriority(a)
  );

  console.log(
    "\n========== DYNAMIC FILE PRIORITIES =========="
  );

  console.table(
    prioritizedFiles.map((file) => ({
      file: file.file,

      mandatory:
        mandatoryFiles.has(file.file),

      searchScore: file.score,

      dependencyScore:
        fileRelevance.get(file.file) ?? 0,

      incomingDependencies:
        incomingDependencies.get(file.file) ?? 0,

      outgoingDependencies:
        outgoingDependencies.get(file.file) ?? 0,

      finalPriority:
        getFilePriority(file),
    }))
  );

  // --------------------------------------------------
  // FILE SECTION HELPERS
  // --------------------------------------------------

  function getFileHeader(filePath: string): string {
    return `
==================================================
FILE: ${filePath}
==================================================
`;
  }

  const footer = "\n";

  function getHeaderSize(filePath: string): number {
    return (
      getFileHeader(filePath).length +
      footer.length
    );
  }

  // --------------------------------------------------
  // APPEND FILE TO CONTEXT
  // --------------------------------------------------

  function appendFile(
    file: ThemeFile,
    contentLimit: number
  ): boolean {
    const header = getFileHeader(file.file);

    const originalContent = file.content;

    const limitedContent = originalContent.slice(
      0,
      MAX_FILE_SIZE
    );

    const content = limitedContent.slice(
      0,
      contentLimit
    );

    const section =
      header + content + footer;

    if (
      context.length + section.length >
      contextBudget
    ) {
      return false;
    }

    context += section;

    filesIncludedInContext.push(file.file);

    includedSourceContent.set(
      file.file,
      content
    );

    if (
      content.length < originalContent.length
    ) {
      truncatedFiles.push(file.file);
    }

    return true;
  }

  // --------------------------------------------------
  // STEP 1: SEPARATE MANDATORY AND OPTIONAL FILES
  // --------------------------------------------------

  const mandatoryFileObjects =
    prioritizedFiles.filter((file) =>
      mandatoryFiles.has(file.file)
    );

  const optionalFileObjects =
    prioritizedFiles.filter((file) =>
      !mandatoryFiles.has(file.file)
    );

  // --------------------------------------------------
  // STEP 2: CALCULATE MANDATORY HEADER COST
  // --------------------------------------------------

  const mandatoryHeaderCost =
    mandatoryFileObjects.reduce(
      (total, file) =>
        total + getHeaderSize(file.file),
      0
    );

  const mandatoryContentBudget =
    Math.max(
      0,
      contextBudget - mandatoryHeaderCost
    );

  // --------------------------------------------------
  // STEP 3: FAIR BUDGET ALLOCATION
  // --------------------------------------------------

  const mandatoryAllocations =
    new Map<string, number>();

  let remainingMandatoryBudget =
    mandatoryContentBudget;

  let activeMandatoryFiles = [
    ...mandatoryFileObjects,
  ];

  // Water-filling allocation:
  // Smaller files release unused budget
  // for larger mandatory files.

  while (
    activeMandatoryFiles.length > 0 &&
    remainingMandatoryBudget > 0
  ) {
    const fairShare = Math.floor(
      remainingMandatoryBudget /
      activeMandatoryFiles.length
    );

    if (fairShare <= 0) {
      break;
    }

    const nextActiveFiles: ThemeFile[] = [];

    let allocatedThisRound = 0;

    for (const file of activeMandatoryFiles) {
      const maxContentLength = Math.min(
        file.content.length,
        MAX_FILE_SIZE
      );

      const alreadyAllocated =
        mandatoryAllocations.get(file.file) ?? 0;

      const remainingFileContent =
        maxContentLength - alreadyAllocated;

      const allocation = Math.min(
        fairShare,
        remainingFileContent
      );

      mandatoryAllocations.set(
        file.file,
        alreadyAllocated + allocation
      );

      allocatedThisRound += allocation;

      if (
        remainingFileContent > allocation
      ) {
        nextActiveFiles.push(file);
      }
    }

    remainingMandatoryBudget -=
      allocatedThisRound;

    activeMandatoryFiles = nextActiveFiles;

    if (allocatedThisRound === 0) {
      break;
    }
  }

  // --------------------------------------------------
  // STEP 4: INCLUDE ALL MANDATORY FILES FIRST
  // --------------------------------------------------

  console.log(
    "\n========== MANDATORY FAIR ALLOCATION =========="
  );

  console.table(
    mandatoryFileObjects.map((file) => ({
      file: file.file,

      originalCharacters:
        file.content.length,

      allocatedCharacters:
        mandatoryAllocations.get(file.file) ?? 0,

      included:
        true,
    }))
  );

  for (const file of mandatoryFileObjects) {
    const allocation =
      mandatoryAllocations.get(file.file) ?? 0;

    const included = appendFile(
      file,
      allocation
    );

    if (!included) {
      filesSkippedFromContext.push(file.file);
    }
  }

  // --------------------------------------------------
  // STEP 5: OPTIONAL FILES
  // --------------------------------------------------

  for (const file of optionalFileObjects) {
    const headerSize =
      getHeaderSize(file.file);

    const remainingBudget =
      contextBudget - context.length;

    const availableContentBudget =
      remainingBudget - headerSize;

    const requiredContentLength = Math.min(
      file.content.length,
      MAX_FILE_SIZE
    );

    // Optional files are included only when
    // their complete capped content fits.

    if (
      availableContentBudget < requiredContentLength
    ) {
      filesSkippedFromContext.push(file.file);
      continue;
    }

    appendFile(
      file,
      requiredContentLength
    );
  }

  // --------------------------------------------------
  // STEP 6: REPORT
  // --------------------------------------------------

  const mandatoryIncludedCount =
    mandatoryFileObjects.filter((file) =>
      filesIncludedInContext.includes(file.file)
    ).length;

  console.log(
    "\n========== CONTEXT REPORT =========="
  );

  console.log({
    totalAvailableFiles: files.length,

    mandatoryFiles: mandatoryFiles.size,

    mandatoryFilesIncluded:
      mandatoryIncludedCount,

    mandatoryFilesSkipped:
      mandatoryFiles.size - mandatoryIncludedCount,

    filesIncluded:
      filesIncludedInContext.length,

    filesSkipped:
      filesSkippedFromContext.length,

    truncatedFiles:
      truncatedFiles.length,

    contextBudget,

    contextUsed: context.length,

    contextRemaining:
      contextBudget - context.length,

    mandatoryHeaderCost,

    mandatoryContentBudget,
  });

  console.log(
    "\n========== FILES SENT TO GEMINI =========="
  );

  console.table(filesIncludedInContext);

  console.log(
    "\n========== FILES SKIPPED =========="
  );

  console.table(filesSkippedFromContext);

  console.log(
    "\n========== TRUNCATED FILES =========="
  );

  console.table(truncatedFiles);

  // --------------------------------------------------
  // RETURN
  // --------------------------------------------------
  return {
    context,
    filesIncludedInContext,
    filesSkippedFromContext,
    includedSourceContent,
    truncatedFiles,
    filePriorities: new Map(
      prioritizedFiles.map((file) => [
        file.file,
        getFilePriority(file),
      ])
    ),
  };
}

// --------------------------------------------------
// PARSE PLANNER RESPONSE — need_chunks (primary) / need_files (fallback)
// --------------------------------------------------

function parsePlannerResponse(
  text: string,
  availableFiles: Set<string>,
  fileLengths: Map<string, number>
): PlannerChunkRequest[] {
  if (!text || typeof text !== "string") {
    console.error("PLANNER RESPONSE IS EMPTY");
    return [];
  }

  try {
    const cleaned = text
      .trim()
      .replace(/^```(?:json)?\\s*/i, "")
      .replace(/\\s*```$/i, "")
      .trim();

    const parsed: unknown = JSON.parse(cleaned);

    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      throw new Error("Planner response must be a JSON object");
    }

    const obj = parsed as Record<string, unknown>;
    const requests: PlannerChunkRequest[] = [];
    const requestedFiles = new Set<string>();

    // ----------------------------------------------
    // PRIMARY FORMAT: need_chunks
    // ----------------------------------------------

    if (Array.isArray(obj.need_chunks)) {
      for (const item of obj.need_chunks) {
        if (typeof item !== "object" || item === null) continue;

        const r = item as Record<string, unknown>;

        if (typeof r.file !== "string") continue;

        const file = normalizePath(r.file.trim());

        if (!availableFiles.has(file)) {
          console.log("PLANNER REQUESTED UNAVAILABLE FILE:", file);
          continue;
        }

        if (requestedFiles.has(file)) {
          console.log("PLANNER DUPLICATE FILE REQUEST:", file);
          continue;
        }

        const totalLength = fileLengths.get(file);

        if (
          typeof totalLength !== "number" ||
          !Number.isFinite(totalLength) ||
          totalLength <= 0
        ) {
          console.log("PLANNER FILE LENGTH UNAVAILABLE:", {
            file,
            totalLength,
          });
          continue;
        }

        let start: number | null = null;

        if (r.start_offset !== undefined && r.start_offset !== null) {
          if (
            typeof r.start_offset !== "number" ||
            !Number.isFinite(r.start_offset) ||
            r.start_offset < 0
          ) {
            console.error("PLANNER PROVIDED INVALID OFFSET:", {
              file,
              startOffset: r.start_offset,
              totalLength,
            });
            continue;
          }

          start = Math.floor(r.start_offset);
        }

        const maxChars =
          typeof r.max_chars === "number" &&
            Number.isFinite(r.max_chars) &&
            r.max_chars > 0
            ? Math.max(
              1,
              Math.min(Math.floor(r.max_chars), MAX_CHUNK_SIZE)
            )
            : MAX_CHUNK_SIZE;

        if (start !== null && start >= totalLength) {
          console.error("PLANNER REQUESTED OFFSET OUT OF RANGE:", {
            file,
            requestedOffset: start,
            totalLength,
          });
          continue;
        }

        requests.push({
          file,
          start_offset: start,
          max_chars: maxChars,
        });

        requestedFiles.add(file);

        if (requests.length >= MAX_ON_DEMAND_FILES) break;
      }

      console.log("VALIDATED PLANNER CHUNK REQUESTS:", requests);
      return requests;
    }

    // ----------------------------------------------
    // LEGACY FALLBACK: need_files
    // ----------------------------------------------

    if (Array.isArray(obj.need_files)) {
      for (const item of obj.need_files) {
        if (typeof item !== "string") continue;

        const file = normalizePath(item.trim());

        if (!availableFiles.has(file)) continue;
        if (requestedFiles.has(file)) continue;

        const totalLength = fileLengths.get(file);

        if (
          typeof totalLength !== "number" ||
          !Number.isFinite(totalLength) ||
          totalLength <= 0
        ) {
          console.log("PLANNER FILE LENGTH UNAVAILABLE:", {
            file,
            totalLength,
          });
          continue;
        }

        requests.push({
          file,
          start_offset: null,
          max_chars: MAX_CHUNK_SIZE,
        });

        requestedFiles.add(file);

        if (requests.length >= MAX_ON_DEMAND_FILES) break;
      }
    }

    console.log("VALIDATED PLANNER CHUNK REQUESTS:", requests);
    return requests;
  } catch (error) {
    console.error("PLANNER JSON PARSE ERROR:", error);
    console.error("RAW PLANNER RESPONSE:", text);
    return [];
  }
}

// --------------------------------------------------
// REQUEST COVERAGE DOMAINS
// --------------------------------------------------

type PlannerManifestFile = {
  file: string;
  totalLength: number;
  priorityScore: number;
  includedInContext: boolean;
  retrievedRanges: RetrievedRange[];
  nextOffset: number;
  hasMore: boolean;
  isComplete: boolean;
};

type CoverageDomain = "sticky-header" | "header" | "pricing" | "settings";

function isStickyHeaderArchitectureIntent(prompt: string): boolean {
  return /\bsticky[\s-]?header\b|\bheader[\s-]?sticky\b|\benable_sticky_header\b/i.test(prompt);
}

function isSettingsSchemaIntent(prompt: string): boolean {
  return /\b(setting|settings|schema|configuration|config|default value|definition|defined)\b/i.test(prompt);
}

function detectCoverageDomains(prompt: string): CoverageDomain[] {
  const lower = prompt.toLowerCase();
  const domains: CoverageDomain[] = [];

  const stickyHeaderIntent = isStickyHeaderArchitectureIntent(prompt);

  if (stickyHeaderIntent) {
    // Sticky behavior is centered in the header section and its styles; do not
    // classify it as a broad menu/navigation task.
    domains.push("sticky-header");
  } else if (/\b(header|navigation|nav|menu|mega[\s-]?menu|drawer)\b/i.test(lower)) {
    domains.push("header");
  }

  if (/\b(pric(?:e|ing)|compare[\s-]?at|sale[\s-]?price|unit[\s-]?price|volume[\s-]?pricing|quantity[\s-]?(?:break|price)|currency)\b/i.test(lower)) {
    domains.push("pricing");
  }

  // Sticky header settings are section-level settings declared inside the
  // section's own {% schema %} block, not global settings_schema.json entries.
  if (isSettingsSchemaIntent(prompt) && !stickyHeaderIntent) {
    domains.push("settings");
  }

  return domains;
}

function getCoverageFilePriority(
  filePath: string,
  domain: CoverageDomain
): number {
  const file = normalizePath(filePath).toLowerCase();

  if (domain === "settings") {
    return file === "config/settings_schema.json" ? 2200 : 0;
  }

  if (domain === "sticky-header") {
    if (file === "sections/header.liquid") return 3000;
    if (file === "assets/base.css") return 2800;
    if (/^assets\/[^/]*(?:header|sticky)[^/]*\.(?:css|js)$/.test(file)) return 2600;
    if (/^assets\/(?:global|theme)\.js$/.test(file)) return 2200;
    return 0;
  }

  if (domain === "header") {
    const exact: Record<string, number> = {
      // The caller block owns the mobile/navigation_bar/default case branches.
      // Give it first access to the on-demand window needed to prove those
      // relationships instead of spending the first slot on a child snippet.
      "blocks/_header-menu.liquid": 1900,
      "sections/header.liquid": 1750,
      "snippets/header-drawer.liquid": 1650,
      "snippets/mega-menu-list.liquid": 1450,
      "snippets/overflow-list.liquid": 1350,
      "blocks/_header-logo.liquid": 1200,
      "snippets/header-row.liquid": 1150,
      "snippets/measure-header-heights.liquid": 1100,
      "snippets/menu-font-styles.liquid": 1050,
      "snippets/submenu-font-styles.liquid": 1025,
      "snippets/contrast-override.liquid": 1000,
    };

    if (exact[file] !== undefined) return exact[file];
    if (/header-menu|header-row|header-drawer/.test(file)) return 850;
    if (/menu|drawer|navigation|overflow-list/.test(file)) return 800;
    return 0;
  }

  const exact: Record<string, number> = {
    "snippets/price.liquid": 1600,
    "snippets/price-styles.liquid": 1550,
    "snippets/unit-price.liquid": 1500,
  };

  if (exact[file] !== undefined) return exact[file];
  if (/\/(?:price|unit-price|price-styles)\.liquid$/.test(file)) return 1200;
  return 0;
}

function getAvailableCoverageDomains(
  prompt: string,
  manifestFiles: PlannerManifestFile[]
): CoverageDomain[] {
  return detectCoverageDomains(prompt).filter((domain) =>
    manifestFiles.some((file) =>
      file.hasMore &&
      file.totalLength > 0 &&
      getCoverageFilePriority(file.file, domain) > 0
    )
  );
}

// --------------------------------------------------
// TARGETED GLOBAL SETTING DEFINITION RANGES
// --------------------------------------------------

type SourceObjectRange = {
  startOffset: number;
  endOffset: number;
};

function findJsonSettingObjectRange(
  source: string,
  settingId: string
): SourceObjectRange | null {
  const escapedId = settingId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const idPattern = new RegExp(`\"id\"\\s*:\\s*\"${escapedId}\"`, "g");
  let match: RegExpExecArray | null;

  while ((match = idPattern.exec(source)) !== null) {
    // Try the nearest opening brace first. If a brace inside a JSON string
    // happens to be closer, JSON parsing rejects it and the previous brace
    // is tested instead.
    let candidateStart = source.lastIndexOf("{", match.index);

    while (candidateStart >= 0) {
      let depth = 0;
      let inString = false;
      let escaped = false;
      let endOffset = -1;

      for (let i = candidateStart; i < source.length; i++) {
        const char = source[i];

        if (inString) {
          if (escaped) {
            escaped = false;
          } else if (char === "\\") {
            escaped = true;
          } else if (char === '"') {
            inString = false;
          }
          continue;
        }

        if (char === '"') {
          inString = true;
          continue;
        }

        if (char === "{") depth++;
        if (char === "}") {
          depth--;
          if (depth === 0) {
            endOffset = i + 1;
            break;
          }
        }
      }

      if (endOffset > candidateStart) {
        try {
          const parsed = JSON.parse(source.slice(candidateStart, endOffset));
          if (
            parsed &&
            typeof parsed === "object" &&
            !Array.isArray(parsed) &&
            (parsed as Record<string, unknown>).id === settingId
          ) {
            return { startOffset: candidateStart, endOffset };
          }
        } catch {
          // Continue to the previous opening brace if this was not an object.
        }
      }

      candidateStart = source.lastIndexOf("{", candidateStart - 1);
    }
  }

  return null;
}

function buildSettingDefinitionChunkRequests(
  prompt: string,
  dependencies: VerifiedDependency[],
  retrievalState: SharedRetrievalState
): PlannerChunkRequest[] {
  if (!isSettingsSchemaIntent(prompt) || isStickyHeaderArchitectureIntent(prompt)) return [];

  const schemaPath = "config/settings_schema.json";
  const schemaContent = retrievalState.fullContents.get(schemaPath);
  if (!schemaContent) {
    console.log("SETTING SCHEMA CONTENT UNAVAILABLE FOR TARGETED RETRIEVAL");
    return [];
  }

  const promptLower = prompt.toLowerCase();
  const explicitlyNamedSettingIds = new Set(
    [...promptLower.matchAll(/[a-z0-9_]+/g)].map((match) => match[0])
  );
  const allSettingIds = [...new Set(
    dependencies
      .filter((dependency) =>
        dependency.verified &&
        dependency.relationship === "uses global setting" &&
        dependency.to.startsWith("setting:global.")
      )
      .map((dependency) => dependency.to.replace("setting:global.", ""))
      .filter(Boolean)
  )];
  const explicitlyRequestedSettingIds = allSettingIds.filter((settingId) =>
    explicitlyNamedSettingIds.has(settingId.toLowerCase())
  );
  // When the prompt names concrete setting IDs, retrieve only those definitions.
  // Keep the previous dependency-driven fallback for broader settings questions.
  const settingIds = explicitlyRequestedSettingIds.length > 0
    ? explicitlyRequestedSettingIds
    : allSettingIds;

  const requests: PlannerChunkRequest[] = [];
  let totalRequestedCharacters = 0;

  for (const settingId of settingIds) {
    if (requests.length >= MAX_ON_DEMAND_FILES) break;

    const range = findJsonSettingObjectRange(schemaContent, settingId);
    if (!range) {
      console.log("SETTING DEFINITION RANGE NOT FOUND:", settingId);
      continue;
    }

    const objectLength = range.endOffset - range.startOffset;
    const existingRanges = mergeRanges(
      retrievalState.fileStates.get(schemaPath)?.retrievedRanges ?? []
    );
    const alreadyComplete = existingRanges.some((existing) =>
      existing.startOffset <= range.startOffset &&
      existing.endOffset >= range.endOffset
    );

    if (alreadyComplete) {
      console.log("SETTING DEFINITION ALREADY INCLUDED IN GEMINI CONTEXT:", settingId);
      continue;
    }

    const requestStart = findNextGap(
      existingRanges,
      range.startOffset,
      schemaContent.length
    );
    if (requestStart >= range.endOffset) continue;

    const remainingObjectLength = range.endOffset - requestStart;
    const requestedLength = Math.min(MAX_CHUNK_SIZE, remainingObjectLength);

    if (requestedLength <= 0) continue;
    if (totalRequestedCharacters + requestedLength > MAX_ON_DEMAND_FILE_SIZE) break;

    requests.push({
      file: schemaPath,
      start_offset: requestStart,
      max_chars: requestedLength,
      must_retrieve: true,
    });
    totalRequestedCharacters += requestedLength;

    console.log("TARGETED SETTING DEFINITION RANGE:", {
      settingId,
      file: schemaPath,
      startOffset: requestStart,
      endOffset: range.endOffset,
      requestedCharacters: requestedLength,
      completeObjectRequested: remainingObjectLength <= MAX_CHUNK_SIZE,
    });
  }

  return requests;
}

// Sticky/header section settings are declared inside the owning Liquid file's
// {% schema %} block. Retrieve the exact object from that file instead of
// looking for it in config/settings_schema.json (which contains global settings).
function buildLiquidSchemaSettingDefinitionRequests(
  prompt: string,
  dependencies: VerifiedDependency[],
  retrievalState: SharedRetrievalState
): PlannerChunkRequest[] {
  if (!isStickyHeaderArchitectureIntent(prompt)) return [];

  const candidateIds = [...new Set(
    dependencies
      .filter((dependency) =>
        dependency.verified &&
        dependency.relationship === "uses section setting" &&
        dependency.to.startsWith("setting:section.")
      )
      .map((dependency) => dependency.to.replace("setting:section.", ""))
      .filter((settingId) => /sticky/i.test(settingId))
  )];

  const requests: PlannerChunkRequest[] = [];
  let totalRequestedCharacters = 0;

  for (const settingId of candidateIds) {
    if (requests.length >= MAX_ON_DEMAND_FILES) break;

    let matched: { file: string; startOffset: number; endOffset: number } | null = null;

    for (const [filePath, fullContent] of retrievalState.fullContents.entries()) {
      const schemaMatch = /{%\s*schema\s*%}([\s\S]*?){%\s*endschema\s*%}/i.exec(fullContent);
      if (!schemaMatch) continue;

      const schemaBody = schemaMatch[1];
      const objectRange = findJsonSettingObjectRange(schemaBody, settingId);
      if (!objectRange) continue;

      const bodyStart = schemaMatch.index + schemaMatch[0].indexOf(schemaBody);
      matched = {
        file: filePath,
        startOffset: bodyStart + objectRange.startOffset,
        endOffset: bodyStart + objectRange.endOffset,
      };
      break;
    }

    if (!matched) {
      console.log("LIQUID SECTION SETTING DEFINITION RANGE NOT FOUND:", settingId);
      continue;
    }

    const existingRanges = mergeRanges(
      retrievalState.fileStates.get(matched.file)?.retrievedRanges ?? []
    );
    const alreadyComplete = existingRanges.some((existing) =>
      existing.startOffset <= matched!.startOffset &&
      existing.endOffset >= matched!.endOffset
    );
    if (alreadyComplete) {
      console.log("LIQUID SECTION SETTING DEFINITION ALREADY INCLUDED:", {
        settingId, file: matched.file,
      });
      continue;
    }

    const requestStart = findNextGap(
      existingRanges,
      matched.startOffset,
      retrievalState.fullContents.get(matched.file)?.length ?? matched.endOffset
    );
    if (requestStart >= matched.endOffset) continue;

    const requestedLength = Math.min(MAX_CHUNK_SIZE, matched.endOffset - requestStart);
    if (requestedLength <= 0 || totalRequestedCharacters + requestedLength > MAX_ON_DEMAND_FILE_SIZE) break;

    requests.push({
      file: matched.file,
      start_offset: requestStart,
      max_chars: requestedLength,
      must_retrieve: true,
    });
    totalRequestedCharacters += requestedLength;

    console.log("TARGETED LIQUID SECTION SETTING DEFINITION RANGE:", {
      settingId,
      file: matched.file,
      startOffset: requestStart,
      endOffset: matched.endOffset,
      requestedCharacters: requestedLength,
      completeObjectRequested: requestStart + requestedLength >= matched.endOffset,
    });
  }

  return requests;
}

// Return only theme assets explicitly loaded by the owning section. Search
// similarity alone must not make menu or sticky-add-to-cart JS look like
// sticky-header implementation code.
function getDirectlyReferencedThemeAssetFiles(
  sourceText: string,
  candidateFiles: Array<{ file: string }>
): Set<string> {
  const referenced = new Set<string>();
  if (!sourceText) return referenced;

  for (const candidate of candidateFiles) {
    const normalized = normalizePath(candidate.file);
    if (!/^assets\//.test(normalized) || !/\.(?:css|js)$/.test(normalized)) continue;

    const basename = normalized.split('/').pop() || normalized;
    const escaped = basename.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const patterns = [
      new RegExp(`['"]${escaped}['"]\\s*\\|\\s*asset_url\\b`, 'i'),
      new RegExp(`\\b(?:src|href)\\s*=\\s*['"][^'"]*${escaped}[^'"]*['"]`, 'i'),
      new RegExp(`\\b(?:import|from)\\s*['"][^'"]*${escaped}['"]`, 'i'),
      new RegExp(`\\bimport\\s*\\(\\s*['"][^'"]*${escaped}['"]`, 'i'),
    ];

    if (patterns.some((pattern) => pattern.test(sourceText))) referenced.add(normalized);
  }

  return referenced;
}

// Find additional base.css ranges that actually contain sticky-header CSS
// selectors/declarations. The full source is already loaded server-side, so
// use it to select evidence-bearing offsets rather than blindly reading the
// next sequential 6,000 characters. Existing source ranges are never repeated.
function buildStickyCssTargetRequests(
  manifestFiles: PlannerManifestFile[],
  fullContents: Map<string, string>,
  maxRequests = 2
): PlannerChunkRequest[] {
  const cssPath = "assets/base.css";
  const manifest = manifestFiles.find(
    (file) => normalizePath(file.file).toLowerCase() === cssPath
  );
  const source = fullContents.get(cssPath);
  if (!manifest || !source || !manifest.hasMore || maxRequests <= 0) return [];

  const knownRanges = mergeRanges(manifest.retrievedRanges ?? []);
  const selected: PlannerChunkRequest[] = [];
  const patterns = [
    /(?:^|\n)[^\n{}]*(?:sticky|not-sticky|scroll-up)[^\n{}]*\{/gim,
    /\bposition\s*:\s*sticky\b/gi,
  ];

  const matches: number[] = [];
  for (const pattern of patterns) {
    let match: RegExpExecArray | null;
    while ((match = pattern.exec(source)) !== null) {
      matches.push(match.index + (match[0].startsWith("\n") ? 1 : 0));
      if (match[0].length === 0) pattern.lastIndex++;
    }
  }

  matches.sort((a, b) => a - b);
  const uniqueMatches = [...new Set(matches)];

  const isCovered = (offset: number, ranges: RetrievedRange[]) =>
    ranges.some((range) => range.startOffset <= offset && offset < range.endOffset);

  for (const matchOffset of uniqueMatches) {
    if (selected.length >= maxRequests) break;
    if (isCovered(matchOffset, knownRanges)) continue;

    const desiredStart = Math.max(0, matchOffset - 1800);
    const requestStart = findNextGap(knownRanges, desiredStart, source.length);
    if (requestStart >= source.length || requestStart > matchOffset) continue;

    // Keep the target selector/declaration inside the requested window.
    const maxChars = Math.min(MAX_CHUNK_SIZE, source.length - requestStart);
    if (matchOffset >= requestStart + maxChars) continue;

    selected.push({
      file: cssPath,
      start_offset: requestStart,
      max_chars: maxChars,
    });
    knownRanges.push({
      startOffset: requestStart,
      endOffset: requestStart + maxChars,
    });
    const merged = mergeRanges(knownRanges);
    knownRanges.splice(0, knownRanges.length, ...merged);
  }

  console.log("STICKY CSS TARGET RANGE CANDIDATES:", selected.map((request) => ({
    file: request.file,
    startOffset: request.start_offset,
    maxChars: request.max_chars,
  })));

  return selected;
}

// Select the referenced header script range around actual scroll/observer
// logic when present. This avoids assuming the relevant code is at byte zero.
function buildStickyScriptTargetRequests(
  manifestFiles: PlannerManifestFile[],
  fullContents: Map<string, string>,
  referencedAssets: Set<string>,
  maxRequests = 2,
  reservedRequests: PlannerChunkRequest[] = []
): PlannerChunkRequest[] {
  const candidates = manifestFiles
    .filter((file) =>
      file.hasMore &&
      file.totalLength > 0 &&
      /^assets\/.*\.js$/i.test(normalizePath(file.file)) &&
      referencedAssets.has(normalizePath(file.file))
    )
    .sort((a, b) =>
      getCoverageFilePriority(b.file, "sticky-header") -
      getCoverageFilePriority(a.file, "sticky-header") ||
      a.file.localeCompare(b.file)
    );

  const selected: PlannerChunkRequest[] = [];
  const chunkSize = Math.min(3600, MAX_CHUNK_SIZE);
  // Prefer actual observer/listener constructs. Method names are secondary
  // navigation anchors only; they are not evidence that an observer/listener
  // is actually installed. The final claim verifier still requires exact code.
  const behaviorPatterns = [
    /new\s+IntersectionObserver\b/gi,
    /addEventListener\s*\(\s*['\"]scroll['\"]/gi,
    /\b(?:window|document)\.onscroll\b/gi,
    /#observeStickyPosition\b/gi,
    /#handleWindowScroll\b/gi,
  ];

  for (const manifest of candidates) {
    const source = fullContents.get(manifest.file);
    if (!source) continue;

    const plannerReservedRanges = reservedRequests
      .filter((request) => normalizePath(request.file).toLowerCase() === normalizePath(manifest.file).toLowerCase())
      .map((request) => {
        // A null start_offset means "continue from the file's next uncovered
        // offset" in retrieveChunk(); reserve that same range here so the
        // targeted sticky-script window does not overlap it.
        const requestedStart = request.start_offset ?? manifest.nextOffset;
        const startOffset = Math.max(
          0,
          Math.min(source.length, requestedStart)
        );
        const endOffset = Math.max(
          startOffset,
          Math.min(source.length, requestedStart + request.max_chars)
        );
        return { startOffset, endOffset };
      })
      .filter((range) => range.endOffset > range.startOffset);
    const knownRanges = mergeRanges([
      ...(manifest.retrievedRanges ?? []),
      ...plannerReservedRanges,
    ]);
    const offsets: Array<{ offset: number; priority: number }> = [];
    for (let priority = 0; priority < behaviorPatterns.length; priority++) {
      const pattern = behaviorPatterns[priority];
      let match: RegExpExecArray | null;
      while ((match = pattern.exec(source)) !== null) {
        offsets.push({ offset: match.index, priority });
        if (match[0].length === 0) pattern.lastIndex++;
      }
    }

    // Sort explicit constructs ahead of method-name anchors, then by source
    // position. This lets two small windows cover separated observer/scroll
    // implementations instead of spending the whole budget at byte zero.
    offsets.sort((a, b) => a.priority - b.priority || a.offset - b.offset);
    const seen = new Set<number>();

    for (const candidate of offsets) {
      if (selected.length >= maxRequests) break;
      if (seen.has(candidate.offset)) continue;
      seen.add(candidate.offset);
      if (knownRanges.some((range) =>
        range.startOffset <= candidate.offset && candidate.offset < range.endOffset
      )) continue;

      const desiredStart = Math.max(0, candidate.offset - 900);
      const requestStart = findNextGap(knownRanges, desiredStart, source.length);
      if (requestStart >= source.length || requestStart > candidate.offset) continue;

      const maxChars = Math.min(chunkSize, source.length - requestStart);
      if (candidate.offset >= requestStart + maxChars) continue;

      selected.push({
        file: manifest.file,
        start_offset: requestStart,
        max_chars: maxChars,
      });
      knownRanges.push({
        startOffset: requestStart,
        endOffset: requestStart + maxChars,
      });
      const merged = mergeRanges(knownRanges);
      knownRanges.splice(0, knownRanges.length, ...merged);
    }

    if (selected.length > 0) break;
  }

  console.log("STICKY SCRIPT TARGET RANGE CANDIDATES:", selected.map((request) => ({
    file: request.file,
    startOffset: request.start_offset,
    maxChars: request.max_chars,
  })));

  return selected;
}

// --------------------------------------------------
// DETERMINISTIC, DOMAIN-AWARE BACKEND COVERAGE CHECK
// --------------------------------------------------

function buildBackendCoverageRequests(
  manifestFiles: PlannerManifestFile[],
  plannerRequests: PlannerChunkRequest[],
  maxRequests = MAX_ON_DEMAND_FILES,
  prompt = "",
  fullContents: Map<string, string> = new Map()
): PlannerChunkRequest[] {
  const requestedFiles = new Set(
    plannerRequests.map((request) => request.file)
  );

  const requestedDomains = detectCoverageDomains(prompt);
  const stickyHeaderSource = fullContents.get("sections/header.liquid") ?? "";
  const directlyReferencedStickyAssets = getDirectlyReferencedThemeAssetFiles(
    stickyHeaderSource,
    manifestFiles
  );
  const domains = requestedDomains.filter((domain) =>
    manifestFiles.some((file) =>
      file.hasMore &&
      file.totalLength > 0 &&
      getCoverageFilePriority(file.file, domain) > 0
    )
  );
  const selected: PlannerChunkRequest[] = [];
  const selectedFiles = new Set(requestedFiles);
  const requestLimit = domains.length === 1 && domains[0] === "header"
    ? Math.min(maxRequests, 3)
    : domains.length === 1 && domains[0] === "sticky-header"
      ? Math.min(maxRequests, 4)
      : maxRequests;

  // A narrow request must never fall through to arbitrary high-priority
  // files when every file in its requested domain is already complete.
  // Planner-selected cross-file ranges are retained by the caller; this
  // backend fallback only supplements them with domain-relevant sources.
  if (requestedDomains.length > 0 && domains.length === 0) {
    console.log("NO INCOMPLETE FILES IN REQUESTED COVERAGE DOMAINS — SKIPPING GENERIC FALLBACK:", requestedDomains);
    return [];
  }

  const makeDomainCandidates = (domain: CoverageDomain) =>
    manifestFiles
      .filter((file) => {
        if (
          !file.hasMore ||
          file.totalLength <= 0 ||
          selectedFiles.has(file.file) ||
          getCoverageFilePriority(file.file, domain) <= 0
        ) return false;

        // base.css is the explicit stylesheet root. Any other CSS/JS asset
        // must be directly referenced by sections/header.liquid.
        const normalizedFile = normalizePath(file.file);
        if (
          domain === "sticky-header" &&
          /^assets\/.*\.(?:css|js)$/i.test(normalizedFile) &&
          normalizedFile !== "assets/base.css"
        ) {
          return directlyReferencedStickyAssets.has(normalizedFile);
        }
        return true;
      })
      .sort((a, b) => {
        const aDomain = getCoverageFilePriority(a.file, domain);
        const bDomain = getCoverageFilePriority(b.file, domain);
        const aUnseen = a.retrievedRanges.length === 0 ? 35 : 0;
        const bUnseen = b.retrievedRanges.length === 0 ? 35 : 0;
        return (bDomain + b.priorityScore * 0.2 + bUnseen) -
          (aDomain + a.priorityScore * 0.2 + aUnseen) ||
          a.file.localeCompare(b.file);
      });

  // Sticky-header needs broader CSS coverage than a single sequential chunk.
  // Select up to three non-overlapping base.css windows around actual sticky CSS
  // selectors/declarations, plus one explicitly referenced JS asset. This is
  // still limited to the exact section-owned implementation files.
  if (domains.length === 1 && domains[0] === "sticky-header") {
    // Sticky CSS often has several separated state selectors. Retrieve up to
    // three non-overlapping evidence windows before spending the remaining
    // explicit-only budget on generic sequential gaps. This keeps the CSS
    // state rules together in the model context instead of using a large
    // portion of the budget on unrelated offsets.
    const stickyCssRequests = buildStickyCssTargetRequests(
      manifestFiles,
      fullContents,
      Math.min(3, requestLimit)
    );
    selected.push(...stickyCssRequests);

    // STICKY HEADER RANGE FIX: include planner-reserved JS ranges and seek the next uncovered source gap.
    const referencedScripts = buildStickyScriptTargetRequests(
      manifestFiles,
      fullContents,
      directlyReferencedStickyAssets,
      Math.min(2, Math.max(0, requestLimit - selected.length)),
      plannerRequests
    );

    selected.push(...referencedScripts);

    return selected;
  }

  const addCandidate = (file: PlannerManifestFile) => {
    const startOffset = file.retrievedRanges.length > 0
      ? file.nextOffset
      : 0;

    if (
      !Number.isFinite(startOffset) ||
      startOffset < 0 ||
      startOffset >= file.totalLength
    ) {
      return false;
    }

    selected.push({
      file: file.file,
      start_offset: startOffset,
      max_chars: Math.min(MAX_CHUNK_SIZE, file.totalLength - startOffset),
    });
    selectedFiles.add(file.file);
    return true;
  };

  // First pass: guarantee a relevant additional source file for every
  // detected architecture area, even when the planner focuses on only one.
  for (const domain of domains) {
    if (selected.length >= requestLimit) break;
    const candidate = makeDomainCandidates(domain)[0];
    if (candidate) addCandidate(candidate);
  }

  // Second pass: fill remaining slots by alternating domains. This avoids
  // spending every slot on Header when the request also asks about Pricing.
  if (domains.length > 0) {
    let madeProgress = true;
    while (selected.length < requestLimit && madeProgress) {
      madeProgress = false;
      for (const domain of domains) {
        if (selected.length >= requestLimit) break;
        const candidate = makeDomainCandidates(domain)[0];
        if (candidate && addCandidate(candidate)) madeProgress = true;
      }
    }
  } else {
    const fallbackCandidates = manifestFiles
      .filter((file) =>
        file.hasMore &&
        file.totalLength > 0 &&
        file.priorityScore > 0 &&
        !selectedFiles.has(file.file)
      )
      .sort((a, b) => {
        const aPriority = a.priorityScore + (a.retrievedRanges.length === 0 ? 20 : 0);
        const bPriority = b.priorityScore + (b.retrievedRanges.length === 0 ? 20 : 0);
        return bPriority - aPriority || a.file.localeCompare(b.file);
      });

    for (const candidate of fallbackCandidates) {
      if (selected.length >= requestLimit) break;
      addCandidate(candidate);
    }
  }

  return selected;
}

// --------------------------------------------------
// BUDGET-AWARE REQUEST ALLOCATION
// --------------------------------------------------

function allocateChunkRequestBudgets(
  requests: PlannerChunkRequest[],
  manifestFiles: PlannerManifestFile[],
  availableBudget: number
): PlannerChunkRequest[] {
  if (requests.length === 0 || availableBudget <= 0) return [];

  const manifestByFile = new Map(
    manifestFiles.map((file) => [file.file, file])
  );

  const prepared = requests.flatMap((request, index) => {
    const file = manifestByFile.get(request.file);
    if (!file) return [];

    const startOffset = request.start_offset ?? file.nextOffset;
    if (
      !Number.isFinite(startOffset) ||
      startOffset < 0 ||
      startOffset >= file.totalLength
    ) {
      return [];
    }

    const sourceRemaining = file.totalLength - startOffset;
    const capacity = Math.max(
      0,
      Math.min(
        Math.floor(request.max_chars),
        MAX_CHUNK_SIZE,
        sourceRemaining,
        MAX_ON_DEMAND_FILE_SIZE
      )
    );

    if (capacity <= 0) return [];

    return [{
      index,
      request,
      startOffset,
      sourceRemaining,
      capacity,
      mustRetrieve: request.must_retrieve === true,
      allocated: 0,
    }];
  });

  let remaining = Math.floor(availableBudget);

  // Exact setting-definition objects are small by design. Allocate these
  // complete source ranges before applying the generic 1,200-character
  // minimum used to avoid wasting budget on arbitrary planner fragments.
  for (const item of prepared) {
    if (!item.mustRetrieve) continue;
    if (item.capacity > remaining) {
      console.log("REQUIRED TARGETED SOURCE RANGE DOES NOT FIT REMAINING BUDGET:", {
        file: item.request.file,
        startOffset: item.startOffset,
        requestedCharacters: item.capacity,
        remainingBudget: remaining,
      });
      continue;
    }
    item.allocated = item.capacity;
    remaining -= item.allocated;
  }

  // Complete short files first (for example price-styles.liquid) rather
  // than leaving a tiny unusable tail allocation on the final large file.
  for (const item of prepared) {
    if (item.mustRetrieve || item.allocated > 0) continue;
    const isCompleteSmallFile =
      item.sourceRemaining <= MAX_CHUNK_SIZE &&
      item.capacity === item.sourceRemaining;

    if (!isCompleteSmallFile) continue;
    if (item.capacity > remaining) continue;

    item.allocated = item.capacity;
    remaining -= item.allocated;
  }

  const largeItems = prepared.filter(
    (item) => item.allocated === 0 && !item.mustRetrieve
  );

  if (largeItems.length > 0 && remaining > 0) {
    const fairShare = Math.floor(remaining / largeItems.length);

    for (const item of largeItems) {
      const allocation = Math.min(item.capacity, fairShare);

      // Do not send a tiny fragment merely to consume the last budget
      // characters. A later request can use the next turn's fresh context.
      if (allocation < 1200) continue;

      item.allocated = allocation;
      remaining -= allocation;
    }

    // Redistribute leftover budget to requests that still have capacity.
    for (const item of largeItems) {
      if (remaining <= 0) break;
      if (item.allocated <= 0) continue;

      const extraCapacity = item.capacity - item.allocated;
      if (extraCapacity <= 0) continue;

      const extra = Math.min(extraCapacity, remaining);
      item.allocated += extra;
      remaining -= extra;
    }
  }

  return prepared
    .filter((item) => item.allocated > 0)
    .sort((a, b) => a.index - b.index)
    .map((item) => ({
      file: item.request.file,
      start_offset: item.startOffset,
      max_chars: item.allocated,
    }));
}

// --------------------------------------------------
// SOURCE RANGE COMPLETENESS
// --------------------------------------------------

function hasCompleteSourceCoverage(
  totalLength: number,
  ranges: RetrievedRange[]
): boolean {
  if (!Number.isFinite(totalLength) || totalLength <= 0) return false;
  if (ranges.length === 0) return false;

  const sortedRanges = [...ranges].sort(
    (a, b) => a.startOffset - b.startOffset
  );
  let coveredUntil = 0;

  for (const range of sortedRanges) {
    if (range.startOffset > coveredUntil) return false;
    coveredUntil = Math.max(coveredUntil, range.endOffset);
    if (coveredUntil >= totalLength) return true;
  }

  return false;
}

// --------------------------------------------------
// COVERAGE REPORT
// --------------------------------------------------

function buildRetrievalCoverageReport(
  prompt: string,
  manifestFiles: PlannerManifestFile[],
  state: SharedRetrievalState
) {
  const domains = detectCoverageDomains(prompt);

  return domains.map((domain) => {
    const relevantFiles = manifestFiles
      .filter((file) => getCoverageFilePriority(file.file, domain) > 0)
      .sort((a, b) =>
        getCoverageFilePriority(b.file, domain) -
        getCoverageFilePriority(a.file, domain)
      );

    const files = relevantFiles.map((file) => {
      const stateForFile = state.fileStates.get(file.file);
      const ranges = stateForFile?.retrievedRanges ?? [];
      const includedCharacters = ranges.reduce(
        (total, range) => total + (range.endOffset - range.startOffset),
        0
      );

      const complete = hasCompleteSourceCoverage(
        file.totalLength,
        ranges
      );

      return {
        file: file.file,
        totalLength: file.totalLength,
        includedCharacters,
        coveragePercent: file.totalLength > 0
          ? Number(((includedCharacters / file.totalLength) * 100).toFixed(1))
          : 100,
        retrievedRanges: ranges,
        includedInGeminiContext: includedCharacters > 0,
        complete,
      };
    });

    return {
      domain,
      candidateFiles: files.length,
      filesIncluded: files.filter((file) => file.includedInGeminiContext).length,
      filesComplete: files.filter((file) => file.complete).length,
      files,
    };
  });
}

// --------------------------------------------------
// SAFE GEMINI JSON PARSER
// --------------------------------------------------

function parseGeminiJson<T = any>(rawText: string): T {
  if (!rawText || typeof rawText !== "string") {
    throw new Error("Empty Gemini response");
  }

  // Remove markdown code fences
  let cleaned = rawText
    .trim()
    .replace(/^```(?:json)?\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();

  // Find JSON object boundaries
  const firstBrace = cleaned.indexOf("{");
  const lastBrace = cleaned.lastIndexOf("}");

  if (firstBrace === -1 || lastBrace === -1) {
    throw new Error("No JSON object found in Gemini response");
  }

  cleaned = cleaned.slice(firstBrace, lastBrace + 1);

  // --------------------------------------------------
  // FIRST ATTEMPT
  // --------------------------------------------------

  try {
    return JSON.parse(cleaned) as T;
  } catch (firstError) {
    console.warn(
      "Initial Gemini JSON parsing failed. Trying repair..."
    );
  }

  // --------------------------------------------------
  // REPAIR INVALID BACKSLASH ESCAPES
  // --------------------------------------------------

  const repaired = cleaned.replace(
    /\\(?!["\\/bfnrtu])/g,
    "\\\\"
  );

  // --------------------------------------------------
  // SECOND ATTEMPT
  // --------------------------------------------------

  try {
    return JSON.parse(repaired) as T;
  } catch (secondError) {
    console.error(
      "GEMINI JSON PARSE ERROR:",
      secondError
    );

    console.error(
      "RAW GEMINI JSON:",
      cleaned
    );

    throw new Error(
      "Gemini returned invalid JSON even after repair"
    );
  }
}

// --------------------------------------------------
// EVIDENCE MATCHING
// --------------------------------------------------
// Evidence excerpts must be contiguous in both the supplied context and the
// actual theme file. Only whitespace and Liquid trim-marker differences are
// normalized; omitted-code markers are deliberately rejected.

function normalizeEvidenceForSourceMatch(text: string): string {
  return normalizeEvidence(text)
    // Liquid whitespace-control hyphens do not change the tag's operation.
    // Normalize them so `{%- render ... -%}` and `{% render ... %}` can be
    // compared without weakening the requirement that the excerpt's code
    // tokens and ordering exist in the original source.
    .replace(/\{%-/g, "{%")
    .replace(/-\%}/g, "%}")
    .replace(/\{\{-/g, "{{")
    .replace(/-\}\}/g, "}}");
}

function evidenceMatchesSource(
  evidence: string,
  source: string
): boolean {
  const normalizedEvidence = normalizeEvidenceForSourceMatch(evidence);
  const normalizedSource = normalizeEvidenceForSourceMatch(source);

  if (!normalizedEvidence) return false;

  // Evidence must be one exact contiguous excerpt after only the narrowly
  // permitted whitespace and Liquid trim-marker normalization above. Do not
  // split ellipsis-separated fragments and search for them independently:
  // that would allow distant source locations to be stitched into one proof.
  // A literal `...` remains valid when it is genuinely present in the source.
  return normalizedSource.includes(normalizedEvidence);
}

/**
 * Explain exactly which quoted evidence items match the text sent to Gemini
 * and the current on-disk source. This is diagnostic only: it never relaxes
 * the exact contiguous-source verification rule.
 */
function getEvidenceMatchDiagnostics(
  evidenceItems: string[],
  sourceSegments: string[],
  actualSource: string
): Array<{
  evidenceIndex: number;
  normalizedLength: number;
  sentContextMatch: boolean;
  matchedSegmentIndexes: number[];
  actualSourceMatch: boolean;
  actualOccurrenceCount: number;
  preview: string;
}> {
  const normalizedActualSource = normalizeEvidenceForSourceMatch(actualSource);

  return evidenceItems.map((evidence, index) => {
    const normalizedEvidence = normalizeEvidenceForSourceMatch(evidence);
    const matchedSegmentIndexes = sourceSegments
      .map((segment, segmentIndex) =>
        evidenceMatchesSource(evidence, segment) ? segmentIndex : -1
      )
      .filter((segmentIndex) => segmentIndex >= 0);

    let actualOccurrenceCount = 0;
    if (normalizedEvidence) {
      let searchFrom = 0;
      while (searchFrom <= normalizedActualSource.length) {
        const foundAt = normalizedActualSource.indexOf(normalizedEvidence, searchFrom);
        if (foundAt < 0) break;
        actualOccurrenceCount++;
        searchFrom = foundAt + Math.max(normalizedEvidence.length, 1);
      }
    }

    return {
      evidenceIndex: index + 1,
      normalizedLength: normalizedEvidence.length,
      sentContextMatch: matchedSegmentIndexes.length > 0,
      matchedSegmentIndexes,
      actualSourceMatch: actualOccurrenceCount > 0,
      actualOccurrenceCount,
      preview: normalizedEvidence.slice(0, 220),
    };
  });
}

/** Return short, line-numbered source windows around currency-related code. */
function getCurrencySourceDiagnosticWindows(
  actualSource: string,
  maxWindows = 8
): Array<{ line: number; excerpt: string }> {
  const lines = actualSource.split(/\r?\n/);
  const hits: number[] = [];

  for (let index = 0; index < lines.length; index++) {
    if (
      /\buse_currency\b/i.test(lines[index]) ||
      /\bassign\s+price\s*=\s*price\s*\|\s*money\b(?!_with_currency)/i.test(lines[index])
    ) {
      hits.push(index);
    }
  }

  const windows: Array<{ line: number; excerpt: string }> = [];
  for (const hit of hits) {
    const start = Math.max(0, hit - 2);
    const end = Math.min(lines.length, hit + 3);
    windows.push({
      line: hit + 1,
      excerpt: lines
        .slice(start, end)
        .map((line, offset) => `${start + offset + 1}: ${line}`)
        .join("\n"),
    });
    if (windows.length >= maxWindows) break;
  }

  return windows;
}

// --------------------------------------------------
// EXTRACT EXPLICIT THEME FILE REFERENCES FROM THE ANSWER
// --------------------------------------------------

function extractThemeFileReferences(
  answer: string,
  knownThemePaths: Iterable<string> = []
): string[] {
  const references = new Set<string>();
  // Resolve model-authored path casing back to the canonical path discovered
  // in the theme manifest. A capitalized `Snippets/price.liquid` must not
  // bypass path auditing, while genuine case-sensitive paths remain intact.
  const canonicalPaths = new Map<string, string>();
  for (const knownPath of knownThemePaths) {
    const normalizedKnownPath = normalizePath(knownPath);
    if (normalizedKnownPath) {
      canonicalPaths.set(normalizedKnownPath.toLowerCase(), normalizedKnownPath);
    }
  }

  const pattern = /\b(?:assets|blocks|config|layout|sections|snippets|templates)\/[A-Za-z0-9_./-]+\.(?:liquid|json|js|css)\b/gi;

  for (const match of answer.matchAll(pattern)) {
    const normalized = normalizePath(match[0]);
    const canonical = canonicalPaths.get(normalized.toLowerCase()) ?? normalized;

    if (canonical && isSafeThemePath(canonical)) {
      references.add(canonical);
    }
  }

  return [...references].sort();
}

function claimSourceSubjectMatchesFile(
  claim: string,
  sourceFile: string,
  knownThemePaths: Iterable<string> = []
): boolean {
  const references = extractThemeFileReferences(claim, knownThemePaths);
  if (references.length === 0) return true;

  const sourceAction =
    '(?:renders?|loads?|imports?|includes?|calls?|references?|assigns?|sets?|captures?|uses?|checks?|evaluates?|determines?|formats?|computes?|calculates?|detects?|retrieves?)';

  // When a file path is grammatically used as the subject of a source action,
  // it must match claim.file. This catches claims that cite one file's source
  // while attributing its behavior to a different file.
  const subjectReferences = references.filter((reference) => {
    const escaped = reference.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp(`${escaped}\\s+${sourceAction}\\b`, "i").test(claim);
  });

  return subjectReferences.every((reference) =>
    normalizePath(reference).toLowerCase() === normalizePath(sourceFile).toLowerCase()
  );
}

// --------------------------------------------------
// CLAIM / EVIDENCE RELEVANCE CHECK
// --------------------------------------------------

const CLAIM_EVIDENCE_STOPWORDS = new Set([
  // Grammar / connective words.
  "this", "that", "with", "from", "into", "when", "then", "than",
  "they", "them", "their", "there", "these", "those", "what", "which",
  "where", "while", "also", "only", "each", "more", "most", "some",
  "such", "like", "via", "including", "example", "according", "will", "would", "could", "should", "does", "doing", "done",
  "have", "has", "had", "been", "being", "were", "was", "are", "for",
  "and", "but", "all", "any", "one", "two", "the", "a", "an", "of",
  "in", "on", "to", "by", "as", "or", "than", "based", "using",
  "is", "be", "being", "it", "its", "if", "up", "notwithstanding",
  // The verifier separately checks Liquid control flow; the word "condition"
  // itself need not be repeated verbatim in the user-facing paraphrase.
  "condition",

  // Generic action words. Specific operations such as "calculate" are
  // deliberately NOT removed: the evidence must support that action too.
  // Source-description verbs vary naturally between claim metadata and the
  // user-facing sentence (for example, "implements" vs "defines"). They do
  // not represent distinct implementation facts; identifiers and behavior
  // tokens remain mandatory for traceability.
  "define", "defines", "defined", "defining", "implement", "implements",
  "implemented", "implementing", "implementation",
  "use", "used", "uses", "get", "gets", "got", "retrieve", "retrieves",
  "retrieved", "retrieving", "check", "checks", "checked", "checking",
  "set", "sets", "setup", "setting", "determine", "determines",
  "determined", "determining", "capture", "captures", "captured",
  "capturing", "load", "loads", "loaded", "loading", "evaluate",
  "evaluates", "evaluated", "evaluating", "handle", "handles", "handled",
  "accept", "accepts", "accepted", "accepting", "flag", "flags", "flagged",
  "assign", "assigns", "assigned", "assigning", "render", "renders",
  "rendered", "rendering", "include", "includes", "included", "including",
  "import", "imports", "imported", "importing", "directly", "conditionally",
  "reference", "references", "referenced", "referencing",

  // Generic source / theme terminology and non-distinctive nouns.
  // These words frequently appear in natural-language descriptions but do
  // not identify a concrete implementation detail in the source excerpt.
  "configure", "configures", "configured", "configuring",
  // Descriptive prose modifiers that do not represent separate Liquid
  // identifiers or operations. Their presence must not make an otherwise
  // exact assignment/condition fail token coverage.
  "configuration", "configurations", "regular",
  "distinguish", "distinguishes", "distinguished", "distinguishing",
  "verify", "verifies", "verified", "verifying",
  "detect", "detects", "detected", "detecting",
  "whether", "under", "inside", "within", "through", "across",
  "container", "containers", "both", "headers",
  "values", "value", "usage", "uses", "used", "using",
  "snippet", "snippets", "block", "blocks", "section", "sections",
  "header", "file", "files", "source", "code", "theme", "shopify",
  "global", "settings", "based", "value", "true", "false", "assign",
  "liquid", "json", "javascript", "module", "modules", "asset", "assets",
  // Literal setting/variant/else identifiers carry branch identity; this
  // generic noun varies naturally in equivalent sentence paraphrases.
  "branch", "branches",
  "css", "js", "script", "scripts", "content", "architecture", "logic",
  "functionality", "parameter", "parameters", "option", "options", "rule",
  "rules", "type", "types", "mode",
]);

function normalizeClaimEvidenceToken(token: string): string {
  const aliases: Record<string, string> = {
    selected: "select",
    selecting: "select",
    selection: "select",
    retrieves: "retrieve",
    retrieved: "retrieve",
    retrieving: "retrieve",
    formatted: "money",
    formatting: "money",
    formats: "money",
    exceeds: "greater",
    exceeding: "greater",
    homepage: "home",
    pricing: "price",
    prices: "price",
    compared: "compare",
    comparing: "compare",
    comparison: "compare",
    checked: "check",
    checking: "check",
    conditions: "condition",
    names: "name",
    logos: "logo",
    menus: "menu",
    variants: "variant",
    contents: "content",
    collections: "collection",
    cards: "card",
    breaks: "break",
    submenus: "submenu",
    different: "conditional",
    type: "variant",
    enabled: "active",
    enables: "active",
    enabling: "active",
    pages: "template",
    page: "template",
    templates: "template",
    template: "template",
    settings: "setting",
    configured: "configure",
    configures: "configure",
    configuring: "configure",
    distinguishes: "distinguish",
    distinguished: "distinguish",
    distinguishing: "distinguish",
    verifying: "verify",
    verifies: "verify",
    verified: "verify",
    detects: "detect",
    detected: "detect",
    detecting: "detect",
    // "handling" is descriptive prose, not a distinct code concept.
    handling: "handle",
    sub_menu: "submenu",
    // Natural-language descriptions often say "case/matches" while the
    // corresponding Liquid branch is represented by `{% when ... %}`.
    conditional: "condition",
    match: "condition",
    matches: "condition",
    matching: "condition",
    equals: "equal",
    equal: "equal",
    truthy: "present",
    falsy: "absent",
    initialize: "assign",
    initializes: "assign",
    initialized: "assign",
    initializing: "assign",
  };

  return aliases[token] ?? token;
}

function getClaimEvidenceTokens(value: string): string[] {
  // File paths identify the subject file, but must not count as evidence
  // for the behavior asserted about that file.
  const withoutThemePaths = value.replace(
    /\b(?:assets|blocks|config|layout|sections|snippets|templates)\/[A-Za-z0-9_./-]+\.(?:liquid|json|js|css)\b/gi,
    " "
  );

  // Normalize common Liquid / code operators into concepts that can be
  // compared with natural-language claims.
  const normalizedText = withoutThemePaths
    // Preserve this concrete asset identifier as one distinct concept; a
    // generic "menu" token alone must not verify a JavaScript-module claim.
    .replace(/\bheader-menu\.js\b/gi, "headermenujs")
    .replace(/!==|!=/g, " not equal ")
    .replace(/===|==/g, " equal ")
    .replace(/([A-Za-z0-9_.]+)\s*>=\s*([A-Za-z0-9_.]+)/g, "$1 greater equal $2")
    .replace(/([A-Za-z0-9_.]+)\s*<=\s*([A-Za-z0-9_.]+)/g, "$1 less equal $2")
    .replace(/([A-Za-z0-9_.]+)\s*>\s*([A-Za-z0-9_.]+)/g, "$1 greater $2")
    .replace(/([A-Za-z0-9_.]+)\s*<\s*([A-Za-z0-9_.]+)/g, "$1 less $2")
    .replace(/\bcase\b/gi, " conditional ")
    .replace(/\bwhen\b/gi, " condition ")
    .replace(/\b(?:if|elsif|unless)\b/gi, " condition ")
    .replace(/\bcontent_for\b/gi, " content ")
    // Normalize common identifier separators and camelCase without losing
    // the individual semantic components (e.g. overflowMenu -> overflow menu).
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ");

  const tokens = (normalizedText.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .map((token) => normalizeClaimEvidenceToken(token.trim()))
    .filter((token) =>
      token.length >= 4 && !CLAIM_EVIDENCE_STOPWORDS.has(token)
    );

  return [...new Set(tokens)];
}

function hasMeaningfulTokenOverlap(left: string, right: string): boolean {
  const leftTokens = getClaimEvidenceTokens(left);
  const rightTokens = getClaimEvidenceTokens(right);

  if (leftTokens.length === 0 || rightTokens.length === 0) {
    return false;
  }

  // Every meaningful concept in the claim must be represented in the
  // evidence/statement. A single overlapping keyword cannot verify a
  // compound assertion (e.g. "calculates volume pricing" from only a
  // boolean quantity-break configuration check).
  return leftTokens.every((leftToken) =>
    rightTokens.some((rightToken) => {
      if (leftToken === rightToken) return true;

      const prefixLength = Math.min(
        5,
        leftToken.length,
        rightToken.length
      );

      return prefixLength >= 4 &&
        leftToken.slice(0, prefixLength) ===
        rightToken.slice(0, prefixLength);
    })
  );
}

function isStickyHeaderSettingStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  if (!/\benable_sticky_header\b/.test(claimText) ||
      !/\bsetting\b/.test(statementText) ||
      !/\bsticky\s+(?:variable|mode|behavior|behaviour)\b/.test(statementText) ||
      !/\bassign(?:ed|s)?\b/.test(statementText)) return false;

  const values = ["always", "scroll-up"].filter((value) =>
    new RegExp(`\\b${value.replace("-", "[- ]")}\\b`, "i").test(claimText)
  );
  if (values.length === 0) return false;

  // Accept a general overview only when it explicitly describes the setting
  // being mapped to an assigned sticky mode. Atomic sentences still need the
  // specific mode value to be stated.
  const hasConcreteMode = values.every((value) =>
    new RegExp(`\\b${value.replace("-", "[- ]")}\\b`, "i").test(statementText)
  );
  const isExplicitSummary = /\b(?:accordingly|based\s+on|depending\s+on)\b/.test(statementText);
  return hasConcreteMode || isExplicitSummary;
}

function isStickyHeaderAttributeOutputStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  const claimDescribesOutput = /header[- ]component/.test(claimText) &&
    /sticky\s+attribute/.test(claimText) && /\boutputs?\b/.test(claimText);
  const statementDescribesOutput = /header[- ]component/.test(statementText) &&
    /sticky\s+attribute/.test(statementText) &&
    /\b(?:receives?|outputs?|passes?|sets?|populates?|renders?)\b/.test(statementText);
  const preservesPresenceCondition =
    !/\b(?:when|if|while)\b/.test(claimText) ||
    (/\bsticky\s+(?:mode|variable|value)\b/.test(statementText) &&
      /\b(?:present|enabled|populated|truthy)\b/.test(statementText));
  const exactQuotedGuard = /{%\s*if\s+sticky\s*%}/i.test(statement) &&
    /sticky\s*=\s*["']{{\s*sticky\s*}}["']/i.test(statement);
  return claimDescribesOutput && statementDescribesOutput &&
    (exactQuotedGuard || preservesPresenceCondition);
}

function isStickyDesktopCssStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  const describesDesktopStickyContent = (value: string) =>
    /\bdesktop\b/.test(value) &&
    /sticky[-\s]content/.test(value) &&
    (/\bposition\b.{0,32}\bsticky\b|\bsticky\s+position(?:ing)?\b/.test(value)) &&
    /\btop\b|\boffset\b/.test(value);
  return describesDesktopStickyContent(claimText) &&
    describesDesktopStickyContent(statementText);
}

function isStickyCssDescriptionStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  const claimDescribesStickyContent = /\bsticky[-\s]+content\s+(?:css\s+)?(?:class|classes|style|styles|rule|rules)\b/.test(claimText);
  const statementDescribesStickyContent = /\bsticky[-\s]+content\s+(?:css\s+)?(?:class|classes|style|styles|rule|rules)\b/.test(statementText);
  const hasStickyPositionAndTop = (value: string) =>
    /\bposition\b.{0,24}\bsticky\b|\bsticky\s+position(?:ing)?\b/.test(value) &&
    /\b(?:top|offset)\b/.test(value);
  return claimDescribesStickyContent && statementDescribesStickyContent &&
    hasStickyPositionAndTop(claimText) && hasStickyPositionAndTop(statementText);
}

function isStickyObserverConstructionStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  const sameObserverOperation =
    /#observestickyposition/i.test(claim) &&
    /\bintersectionobserver\b/i.test(claim) &&
    /\b(?:instantiat\w*|creat\w*|construct\w*)\b/.test(claimText) &&
    /#observestickyposition/i.test(statement) &&
    /\bintersectionobserver\b/i.test(statement) &&
    /\b(?:instantiat\w*|creat\w*|construct\w*|uses?)\b/.test(statementText);
  const thresholdPreserved = !/\bthreshold\b/i.test(claim) ||
    /\bthreshold\b/i.test(statement);
  return sameObserverOperation && thresholdPreserved;
}

function isStickyIntersectionStateStatementAligned(claim: string, statement: string): boolean {
  const hasExactStateSemantics = (value: string) => {
    const text = value.toLowerCase();
    return /dataset\.stickystate/.test(text) &&
      /isintersecting/.test(text) &&
      /alwayssticky/.test(text) &&
      /(?:intersection\s*observer callback|#observestickyposition)/i.test(value) &&
      /inactive/.test(text) &&
      /active/.test(text);
  };
  return hasExactStateSemantics(claim) && hasExactStateSemantics(statement);
}

function isStickyConnectedCallbackInitStatementAligned(claim: string, statement: string): boolean {
  const claimDescribesInitFlow = /connectedcallback/i.test(claim) &&
    /#observeStickyPosition/i.test(claim) && /sticky/i.test(claim);
  const statementPreservesInitFlow = /connectedcallback/i.test(statement) &&
    /#observeStickyPosition/i.test(statement) && /sticky/i.test(statement) &&
    /always/i.test(statement);
  return claimDescribesInitFlow && statementPreservesInitFlow;
}

// Keep scroll-listener claim wording aligned with the exact mode guard. This
// accepts equivalent prose ("registers" / "adds") but never broadens the
// source condition to every truthy sticky mode.
function isStickyScrollListenerStatementAligned(claim: string, statement: string): boolean {
  const describesExactScrollRegistration = (value: string): boolean => {
    const normalized = value.toLowerCase();
    const preservesBothModes = /\bstickyMode\b/i.test(value) &&
      /\bscroll[- ]up\b/i.test(normalized) &&
      /\balways\b/i.test(normalized) &&
      /\b(?:or)\b|\|\|/i.test(normalized);
    const describesRegistration = /\bscroll\s+event\s+listener\b/i.test(normalized) &&
      /\b(?:adds?|registers?|attaches?|installs?)\b/i.test(normalized);
    return preservesBothModes && describesRegistration;
  };

  return describesExactScrollRegistration(claim) &&
    describesExactScrollRegistration(statement);
}

function isStickyObserverThresholdStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  const sameThreshold = /#observestickyposition/.test(claimText) &&
    /#observestickyposition/.test(statementText) &&
    /intersectionobserver/.test(claimText) && /intersectionobserver/.test(statementText) &&
    /threshold/.test(claimText) && /threshold/.test(statementText) &&
    /alwayssticky/.test(claimText) && /alwayssticky/.test(statementText);
  const rootPreserved = !/getintersectionroot|\broot\b/.test(claimText) ||
    /getintersectionroot|\broot\b/.test(statementText);
  return sameThreshold && rootPreserved;
}

function isStickyOffscreenAssignmentStatementAligned(claim: string, statement: string): boolean {
  const claimText = claim.toLowerCase();
  const statementText = statement.toLowerCase();
  return /#offscreen/.test(claimText) && /#offscreen/.test(statementText) &&
    /alwayssticky/.test(claimText) && /alwayssticky/.test(statementText) &&
    /false/.test(claimText) && /false/.test(statementText) &&
    /isintersecting/.test(claimText) && /isintersecting/.test(statementText) &&
    /dataset\.stickystate/.test(claimText) && /dataset\.stickystate/.test(statementText);
}

function isStickyScrollRafStatementAligned(claim: string, statement: string): boolean {
  const claimHasRafScrollHandler =
    /(?:#handleWindowScroll|scroll\s+handling|scroll\s+handler)/i.test(claim) &&
    /requestAnimationFrame/i.test(claim) &&
    /#?updateScrollState/i.test(claim);
  const statementDescribesSameHandler =
    /(?:#handleWindowScroll|scroll\s+handler)/i.test(statement) &&
    /\bscroll\b/i.test(statement) &&
    /requestAnimationFrame/i.test(statement) &&
    /#?updateScrollState/i.test(statement) &&
    /\b(?:update|updates|call|calls|trigger|triggers|throttle|throttles)\b/i.test(statement);
  return claimHasRafScrollHandler && statementDescribesSameHandler;
}

// The model is instructed to emit one independently verifiable fact per
// claim. Enforce that contract server-side as well, so a broad claim cannot
// pass merely by supplying a collection of unrelated evidence excerpts.
function isAtomicAnswerClaim(claim: string): boolean {
  const normalized = claim.toLowerCase();

  // One-to-many dependency statements must be split so every edge can be
  // independently audited (e.g. renders A, B, and C). Do not reject ordinary
  // conditional descriptions such as "renders navigation_bar when X and Y".
  // A source path plus two or more target paths is a reliable one-to-many
  // signal; a simple conjunction by itself is not.
  const isDependencyVerb = /\b(?:renders?|includes?|imports?|calls?|references?)\b/i.test(normalized);
  const explicitFileReferences = extractThemeFileReferences(claim);
  if (isDependencyVerb && explicitFileReferences.length > 2) return false;

  const actionMatches = [...normalized.matchAll(
    // Require a verb to be followed by whitespace. This prevents nouns such
    // as "compare-at price" from being misread as a second action.
    /\b(?:retrieves?|checks?|compares?|formats?|renders?|captures?|loads?|imports?|includes?|sets?|assigns?|calculates?|computes?|evaluates?|determines?|flags?|uses?|detects?|distinguishes?|verifies?)(?=\s)/gi
  )].filter((match) => {
    // Natural language such as "the setting is set to always" does not
    // describe a second source action after the assignment verb.
    if (!/^set$/i.test(match[0])) return true;
    const prefix = normalized.slice(0, match.index ?? 0);
    return !/\bis\s+$/.test(prefix);
  });

  if (actionMatches.length <= 1) return true;

  // "Checks whether X to determine Y" is one logical classification rule,
  // not two unrelated source actions.
  const isCheckToDetermine =
    actionMatches.length === 2 &&
    /\bchecks?\b[\s\S]{0,160}\bto\s+determine\b/i.test(normalized);

  if (isCheckToDetermine) return true;

  // A compare/check condition followed by a flag assignment is one logical
  // source behavior when the evidence also contains a comparison operator.
  const isConditionToAssignment =
    actionMatches.length === 2 &&
    /\b(?:checks?|compares?|verifies?)\b[\s\S]{0,140}\b(?:assigns?|sets?|flags?)\b/i.test(normalized) &&
    /\b(?:greater|less|equal|not)\b|!==?|===?|[<>]=?/.test(normalized);

  return isConditionToAssignment;
}

// --------------------------------------------------
// DETECT BACKEND-ONLY SOURCE COVERAGE CLAIMS
// --------------------------------------------------

function isBackendCoverageMetadataClaim(claim: AnswerClaim): boolean {
  const statement = (claim.claim || "").toLowerCase();
  const evidence = (Array.isArray(claim.evidence)
    ? claim.evidence.join(" ")
    : claim.evidence || "").toLowerCase();

  const statementDescribesCoverage =
    /\b(?:source|theme source|source file|file)\b.{0,65}\b(?:coverage|complete|partial|incomplete|retrieved|supplied|included|available|provided)\b/.test(statement) ||
    /\b(?:coverage|completeness)\b.{0,45}\b(?:source|file|percent|percentage|range|retrieved|supplied)\b/.test(statement) ||
    /\b(?:partial|incomplete|complete|full)\s+(?:source|file)\s+(?:coverage|available|provided|supplied)\b/.test(statement) ||
    /\b(?:incomplete|partial|complete|full)\s+(?:source\s+)?files?\b/.test(statement) ||
    // Gemini often phrases backend manifest facts as "<path> coverage is
    // incomplete" or simply "<path> is complete". These are not source-code
    // claims and must be handled by the backend coverage manifest instead.
    ( /\bcoverage\b/.test(statement) &&
      /\b(?:complete|incomplete|partial|full)\b/.test(statement) ) ||
    ( Boolean(claim.file) &&
      /\b(?:is|are)\s+(?:not\s+)?(?:complete|incomplete|partial(?:ly)?|full)\b/.test(statement) &&
      extractThemeFileReferences(statement).length > 0 );

  const evidenceIsManifestMetadata =
    /\b(?:hasmore|iscomplete|coveragepercent|retrievedranges|includedcharacters|totallength|filescomplete)\b/.test(evidence) ||
    /["'](?:complete|hasMore|isComplete|coveragePercent|retrievedRanges|includedCharacters)["']\s*:\s*(?:true|false|\d+)/i.test(evidence) ||
    /\b(?:partial|partially|incomplete|complete)\s*\(\s*\d+(?:\.\d+)?%\s+(?:supplied|covered|retrieved)\b/i.test(evidence);

  return statementDescribesCoverage || evidenceIsManifestMetadata;
}

// --------------------------------------------------
// DEPENDENCY RELATIONSHIP VALIDATION
// --------------------------------------------------

type DependencyRelationshipKind =
  | "render"
  | "include"
  | "import"
  | "extend"
  | "reference"
  | "call"
  | "use"
  | "unknown";

function getDependencyRelationshipKind(
  relationship: string
): DependencyRelationshipKind {
  const value = relationship
    .toLowerCase()
    .trim()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ");

  // Match the dependency labels emitted by extractDependencies().
  // In particular, "renders snippet" is plural and must still map to render.
  if (/^renders?\s+(?:snippet|theme block|section)$/.test(value)) {
    return "render";
  }

  if (/^includes?\b/.test(value)) return "include";
  if (/^imports?\b/.test(value)) return "import";
  if (/^(?:extends?|inherits?)\b/.test(value)) return "extend";
  if (/^references?\b/.test(value)) return "reference";
  if (/^calls?\b/.test(value)) return "call";
  if (/^uses?\b/.test(value)) return "use";

  return "unknown";
}

function relationshipMatchesStatement(
  relationship: string,
  statement: string
): boolean {
  const kind = getDependencyRelationshipKind(relationship);

  const patterns: Record<
    Exclude<DependencyRelationshipKind, "unknown">,
    RegExp
  > = {
    render: /\brender(?:s|ed|ing)?\b/i,
    include: /\binclud(?:e|es|ed|ing)\b/i,
    import: /\bimport(?:s|ed|ing)?\b/i,
    extend: /\bextend(?:s|ed|ing)?\b|\binherit(?:s|ed|ing)?\b/i,
    reference: /\breferenc(?:e|es|ed|ing)\b/i,
    call: /\bcall(?:s|ed|ing)?\b/i,
    use: /\buse(?:s|d|ing)?\b/i,
  };

  if (kind === "unknown") return false;

  // A positive dependency edge cannot support a negated statement.
  if (/\b(?:not|never|without|no|cannot|can['’]t|doesn['’]t|does not|isn['’]t|is not|do not)\b/i.test(statement)) {
    return false;
  }

  // A graph edge proves one relationship type only. Do not let a statement
  // containing a second, different relationship piggyback on the matching one.
  const statementWithoutPaths = statement.replace(
    /\b(?:assets|blocks|config|layout|sections|snippets|templates)\/[A-Za-z0-9_./-]+\.(?:liquid|json|js|css)\b/gi,
    " "
  );
  const matchedKinds = (Object.entries(patterns) as Array<
    [Exclude<DependencyRelationshipKind, "unknown">, RegExp]
  >)
    .filter(([, pattern]) => pattern.test(statementWithoutPaths))
    .map(([matchedKind]) => matchedKind);

  return matchedKinds.length === 1 && matchedKinds[0] === kind;
}

function dependencyTargetMentioned(
  statement: string,
  targetFile: string
): boolean {
  const normalizedTarget = normalizePath(targetFile);

  if (!normalizedTarget) return false;

  const normalizedStatement = statement
    .replace(/\\/g, "/")
    .toLowerCase();

  if (normalizedStatement.includes(normalizedTarget.toLowerCase())) {
    return true;
  }

  const basename = normalizedTarget.split("/").pop() || "";
  const nameWithoutExtension = basename.replace(/\.[^.]+$/, "");

  if (!nameWithoutExtension) return false;

  const escapedName = nameWithoutExtension.replace(
    /[.*+?^${}()|[\]\\]/g,
    "\\$&"
  );

  return new RegExp(
    `\\b${escapedName}\\b`,
    "i"
  ).test(statement);
}

function isPureDependencyRelationshipStatement(
  statement: string,
  sourceFile: string,
  targetFile: string,
  relationship: string
): boolean {
  if (
    !dependencyTargetMentioned(statement, targetFile) ||
    !relationshipMatchesStatement(relationship, statement)
  ) {
    return false;
  }

  let residual = statement;
  const namesToRemove = [
    normalizePath(sourceFile),
    normalizePath(targetFile),
    path.basename(normalizePath(sourceFile)),
    path.basename(normalizePath(targetFile)),
    path.basename(normalizePath(sourceFile)).replace(/\.[^.]+$/, ""),
    path.basename(normalizePath(targetFile)).replace(/\.[^.]+$/, ""),
  ].filter(Boolean);

  for (const name of [...new Set(namesToRemove)].sort((a, b) => b.length - a.length)) {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    residual = residual.replace(new RegExp(escaped, "gi"), " ");
  }

  residual = residual
    .replace(/\b(?:renders?|rendered|rendering|includes?|included|including|imports?|imported|importing|extends?|extended|extending|inherits?|inherited|calls?|called|calling|references?|referenced|referencing|uses?|used|using)\b/gi, " ")
    .replace(/\b(?:snippet|snippets|theme|block|blocks|section|sections|file|files|module|modules|asset|assets|directly|the|a|an|this|that|from|to|and|of|via|on|in)\b/gi, " ");

  // Graph-only verification is allowed only for the relationship itself.
  // Extra behavior such as position, ordering, visibility, or responsive
  // qualifiers needs direct source evidence and cannot ride on a dependency edge.
  if (/\b(?:top|bottom|first|last|before|after|only|mobile|desktop|tablet|default|always|never|conditionally|conditionally)\b/i.test(residual)) {
    return false;
  }

  return getClaimEvidenceTokens(residual).length === 0;
}

function findVerifiedDependencyRelationship(
  sourceFile: string,
  targetFile: string,
  statement: string,
  verifiedDependencies: VerifiedDependency[]
): VerifiedDependency | undefined {
  const normalizedSource = normalizePath(sourceFile);
  const normalizedTarget = normalizePath(targetFile);

  if (!normalizedSource || !normalizedTarget) {
    return undefined;
  }

  return verifiedDependencies.find((dependency) => {
    if (!dependency.verified) return false;

    if (
      normalizePath(dependency.from) !== normalizedSource ||
      normalizePath(dependency.to) !== normalizedTarget
    ) {
      return false;
    }

    if (!dependency.verificationEvidence?.trim()) {
      return false;
    }

    return (
      dependencyTargetMentioned(statement, normalizedTarget) &&
      relationshipMatchesStatement(
        dependency.relationship,
        statement
      )
    );
  });
}

function canAuthorizeDependencyTargetPath(
  statement: string,
  statementFilePaths: string[],
  claim: VerifiedAnswerClaim,
  dependency: VerifiedDependency
): boolean {
  if (isPureDependencyRelationshipStatement(
    statement, claim.file, dependency.to, dependency.relationship
  )) return true;

  // A richer sentence may name only the target snippet path when the cited
  // direct-source claim itself explicitly proves that this caller renders that
  // exact target. The verified dependency edge and sentence/claim overlap are
  // still mandatory; this avoids rejecting valid wording such as
  // "unit-price.liquid is rendered" solely because the caller path is omitted
  // from the prose (the claim ID already anchors the caller source file).
  const callerPathNamedInStatement =
    statementFilePaths.includes(normalizePath(claim.file));
  const callerToTargetEdgeExplicitInClaim =
    dependencyTargetMentioned(claim.claim, dependency.to) &&
    relationshipMatchesStatement(dependency.relationship, claim.claim);

  return claim.verificationMode === "direct_source" &&
    (callerPathNamedInStatement || callerToTargetEdgeExplicitInClaim) &&
    hasMeaningfulTokenOverlap(claim.claim, statement);
}

// --------------------------------------------------
// SOURCE-CONTEXT WINDOWS FOR CONDITIONAL EVIDENCE
// --------------------------------------------------

type EvidenceContextWindow = {
  text: string;
  evidenceStart: number;
};

type ActiveLiquidBranch = {
  kind: "if" | "case";
  expression: string;
  branch: "if" | "elsif" | "unless" | "else" | "when";
  branchValue: string;
};

/**
 * Return a bounded window around each exact excerpt, but only within the same
 * contiguous source segment sent to Gemini. This lets the verifier inspect an
 * enclosing Liquid condition without crossing an unretrieved-range gap.
 */
function getEvidenceContextWindows(
  evidenceItems: string[],
  sourceSegments: string[],
  radius = 1800
): EvidenceContextWindow[] {
  const windows: EvidenceContextWindow[] = [];

  for (const evidenceItem of evidenceItems) {
    const needle = normalizeEvidenceForSourceMatch(evidenceItem);
    if (!needle) continue;

    for (const segment of sourceSegments) {
      const haystack = normalizeEvidenceForSourceMatch(segment);
      let searchFrom = 0;

      while (searchFrom <= haystack.length) {
        const foundAt = haystack.indexOf(needle, searchFrom);
        if (foundAt < 0) break;

        const start = Math.max(0, foundAt - radius);
        const end = Math.min(haystack.length, foundAt + needle.length + radius);
        windows.push({
          text: haystack.slice(start, end),
          evidenceStart: foundAt - start,
        });
        searchFrom = foundAt + Math.max(needle.length, 1);
      }
    }
  }

  return windows;
}

/** Parse the Liquid control-flow stack immediately before a source excerpt. */
// Shopify's `{% liquid %}` tag contains control-flow statements as plain
// lines (`if`, `elsif`, `else`, `endif`) rather than nested `{% if %}` tags.
// Expand only those control-flow lines so branch verification can inspect the
// same enclosing condition without treating arbitrary Liquid statements as tags.
function expandLiquidControlFlowStatements(source: string): string {
  return source.replace(
    /\{%-?\s*liquid\b([\s\S]*?)(?:%-?\}|$)/gi,
    (_match, body: string) => {
      // Evidence matching normalizes whitespace, so a `{% liquid %}` block
      // may arrive as a single line. Split it at Liquid control keywords
      // rather than relying on original line breaks.
      const controlKeywords = new Set([
        "if", "elsif", "unless", "else", "endif", "case", "when", "endcase",
      ]);
      const controls: Array<{ keyword: string; index: number; length: number }> = [];
      let quote: "'" | '"' | null = null;

      // Do not interpret words inside Liquid string literals as control-flow
      // statements (for example, an assigned label containing the word "else").
      for (let index = 0; index < body.length;) {
        const character = body[index];

        if (quote) {
          if (character === "\\") {
            index += 2;
            continue;
          }
          if (character === quote) quote = null;
          index++;
          continue;
        }

        if (character === "'" || character === '"') {
          quote = character;
          index++;
          continue;
        }

        // In `{% liquid %}` blocks, a line whose first non-whitespace
        // character is `#` is a comment. Apostrophes inside that comment
        // (for example `variant's pricing`) must not open a string literal
        // and hide later nested `if`/`endif` statements from the parser.
        if (character === "#") {
          const lineStart = body.lastIndexOf("\n", index - 1) + 1;
          if (/^[\t ]*$/.test(body.slice(lineStart, index))) {
            const lineEnd = body.indexOf("\n", index);
            index = lineEnd < 0 ? body.length : lineEnd + 1;
            continue;
          }
        }

        if (/[A-Za-z]/.test(character) &&
          (index === 0 || !/[A-Za-z0-9_]/.test(body[index - 1]))) {
          let wordEnd = index + 1;
          while (wordEnd < body.length && /[A-Za-z]/.test(body[wordEnd])) wordEnd++;
          const keyword = body.slice(index, wordEnd).toLowerCase();
          const hasWordBoundary = wordEnd === body.length ||
            !/[A-Za-z0-9_]/.test(body[wordEnd]);

          if (hasWordBoundary && controlKeywords.has(keyword)) {
            controls.push({ keyword, index, length: wordEnd - index });
          }
          index = wordEnd;
          continue;
        }

        index++;
      }

      return controls.map((control, controlIndex) => {
        const { keyword, index, length } = control;
        const next = controls[controlIndex + 1];
        const end = next?.index ?? body.length;
        const expression = body.slice(index + length, end).trim();

        if (["if", "elsif", "unless", "case", "when"].includes(keyword)) {
          return `{% ${keyword}${expression ? ` ${expression}` : ""} %}`;
        }

        return `{% ${keyword} %}`;
      }).join("\n");
    }
  );
}

function getActiveLiquidBranches(
  context: EvidenceContextWindow
): ActiveLiquidBranch[] {
  // Parse the prefix before the cited excerpt. An excerpt can start in the
  // middle of a `{% liquid %}` block, so also expand an intentionally
  // incomplete block at the end of this prefix.
  const rawPrefix = context.text.slice(0, context.evidenceStart);
  let prefix = expandLiquidControlFlowStatements(rawPrefix);

  const evidenceTail = context.text.slice(context.evidenceStart);
  // If the quoted excerpt begins with a normal Liquid branch tag, include it
  // even when the excerpt has leading whitespace.
  const leadingBranchTag = evidenceTail.match(
    /^\s*(\{%\s*(?:if|elsif|unless|else|case|when)\b[\s\S]*?%\})/i
  );
  if (leadingBranchTag) {
    prefix += leadingBranchTag[1];
  } else {
    // Inside a `{% liquid %}` block, branch transitions are plain statements.
    // Include a leading `else`/`elsif` line so the enclosing condition is
    // evaluated against the branch containing the quoted evidence.
    const lastLiquidOpen = rawPrefix.toLowerCase().lastIndexOf("{% liquid");
    const lastLiquidClose = rawPrefix.lastIndexOf("%}");
    const isInsideLiquidBlock = lastLiquidOpen > lastLiquidClose;
    const leadingLiquidBranch = isInsideLiquidBlock
      ? evidenceTail.match(/^\s*(else|elsif\b(?:\s+.{1,180})?|when\b(?:\s+.{1,120})?)/i)
      : null;

    if (leadingLiquidBranch) {
      prefix += `{% ${leadingLiquidBranch[1].trim()} %}`;
    }
  }

  const stack: ActiveLiquidBranch[] = [];
  const tagPattern = /\{%\s*(if|elsif|unless|else|endif|case|when|endcase)\b([\s\S]*?)%\}/gi;

  for (const match of prefix.matchAll(tagPattern)) {
    const tag = match[1].toLowerCase();
    const expression = (match[2] || "").trim();

    if (tag === "if" || tag === "unless") {
      stack.push({
        kind: "if",
        expression,
        branch: tag,
        branchValue: expression,
      });
      continue;
    }

    if (tag === "case") {
      stack.push({
        kind: "case",
        expression,
        branch: "when",
        branchValue: "",
      });
      continue;
    }

    if (tag === "elsif") {
      const frameIndex = stack.map((frame) => frame.kind).lastIndexOf("if");
      if (frameIndex >= 0) {
        stack[frameIndex] = {
          ...stack[frameIndex],
          branch: "elsif",
          branchValue: expression,
        };
      }
      continue;
    }

    if (tag === "else") {
      // Liquid `else` belongs to the innermost still-open control structure.
      // For `{% if %}` it is the if-else branch; for `{% case %}` it is the
      // default case branch. Looking only for an `if` loses case-else context.
      const frameIndex = stack.length - 1;
      if (frameIndex >= 0) {
        stack[frameIndex] = {
          ...stack[frameIndex],
          branch: "else",
          branchValue: "",
        };
      }
      continue;
    }

    if (tag === "when") {
      const frameIndex = stack.map((frame) => frame.kind).lastIndexOf("case");
      if (frameIndex >= 0) {
        stack[frameIndex] = {
          ...stack[frameIndex],
          branch: "when",
          branchValue: expression,
        };
      }
      continue;
    }

    if (tag === "endif" || tag === "endcase") {
      const expectedKind = tag === "endif" ? "if" : "case";
      const frameIndex = stack.map((frame) => frame.kind).lastIndexOf(expectedKind);
      if (frameIndex >= 0) stack.splice(frameIndex, 1);
    }
  }

  return stack;
}

function contextProvesUnitPriceRenderBranch(
  contexts: EvidenceContextWindow[]
): boolean {
  return contexts.some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "if" &&
      frame.branch !== "else" &&
      /\bshow_unit_price\b/i.test(frame.branchValue) &&
      /(?:selected_variant\.)?unit_price\b/i.test(frame.branchValue)
    )
  );
}

/**
 * Build branch-parser contexts anchored to a specific operation in the
 * contiguous source segments that were actually sent to Gemini. This avoids
 * depending on Gemini to repeat a distant enclosing `if` line in its quote.
 */
function getOperationEvidenceContexts(
  sourceSegments: string[],
  operationPattern: RegExp
): EvidenceContextWindow[] {
  const contexts: EvidenceContextWindow[] = [];

  for (const segment of sourceSegments) {
    const pattern = new RegExp(operationPattern.source, operationPattern.flags.includes("g")
      ? operationPattern.flags
      : `${operationPattern.flags}g`);

    for (const match of segment.matchAll(pattern)) {
      if (match.index === undefined) continue;

      // Keep the entire contiguous segment so its opening Liquid tag and all
      // intervening endif/else transitions remain available to the parser.
      // This does not cross an UNRETRIEVED SOURCE RANGE GAP.
      contexts.push({
        text: segment,
        evidenceStart: match.index,
      });
    }
  }

  return contexts;
}

function contextProvesCurrencyBranch(
  contexts: EvidenceContextWindow[],
  sourceSegments: string[],
  expectedBranch: "if" | "else"
): boolean {
  const operationContexts = expectedBranch === "else"
    ? getOperationEvidenceContexts(
        sourceSegments,
        /\bassign\s+price\s*=\s*price\s*\|\s*money\b(?!_with_currency)/i
      )
    : getOperationEvidenceContexts(
        sourceSegments,
        /\bassign\s+price\s*=\s*price\s*\|\s*money_with_currency\b/i
      );

  return [...contexts, ...operationContexts].some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "if" &&
      frame.branch === expectedBranch &&
      /\buse_currency\b/i.test(frame.expression)
    )
  );
}

function contextProvesTrueCurrencyBranch(
  contexts: EvidenceContextWindow[],
  sourceSegments: string[]
): boolean {
  return contextProvesCurrencyBranch(contexts, sourceSegments, "if");
}

function contextProvesFalseCurrencyBranch(
  contexts: EvidenceContextWindow[],
  sourceSegments: string[] = []
): boolean {
  if (contextProvesCurrencyBranch(contexts, sourceSegments, "else")) return true;

  return contexts.some((context) => {
    const isFalseCurrencyBranch = (candidate: EvidenceContextWindow) =>
      getActiveLiquidBranches(candidate).some((frame) =>
        frame.kind === "if" &&
        frame.branch === "else" &&
        /\buse_currency\b/i.test(frame.expression)
      );

    if (isFalseCurrencyBranch(context)) return true;

    // A bounded evidence window may begin in the middle of a long `{% liquid %}`
    // block. In that case its opening tag is outside the window, so the normal
    // expander cannot recognize the plain `if use_currency` statement. Rebuild
    // only the local source prefix from that exact condition through the cited
    // else branch, then run the same Liquid stack parser. No source outside this
    // contiguous evidence window is consulted, and an intervening endif still
    // closes the condition normally.
    const evidenceTail = context.text.slice(context.evidenceStart);
    if (!/^\s*else(?:\s|$)/i.test(evidenceTail)) return false;

    const rawPrefix = context.text.slice(0, context.evidenceStart);
    const branchStart = rawPrefix.toLowerCase().lastIndexOf("if use_currency");
    if (branchStart < 0) return false;

    const localContext: EvidenceContextWindow = {
      text: `{% liquid ${rawPrefix.slice(branchStart)} ${evidenceTail}`,
      evidenceStart: rawPrefix.length - branchStart,
    };

    return isFalseCurrencyBranch(localContext);
  });
}

// --------------------------------------------------
// TARGETED SEMANTIC EVIDENCE CHECKS
// --------------------------------------------------

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Accept "extracts price from selected_variant.price" only when the exact
// Liquid assignment exists in the evidence: assign price = selected_variant.price.
function supportsAssignmentExtractionClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const match = claim.match(
    /\b(?:extracts?|retrieves?|gets?|reads?|takes?)\s+([A-Za-z_][A-Za-z0-9_]*)\s+from\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)/i
  );
  if (!match) return false;

  const assignedName = match[1];
  const sourceExpression = match[2];
  const assignmentPattern = new RegExp(
    `\\bassign\\s+${escapeRegExp(assignedName)}\\s*=\\s*${escapeRegExp(sourceExpression)}\\b`,
    "i"
  );
  return evidenceItems.some((evidence) => assignmentPattern.test(evidence));
}

// A script asset is only verified as an ES module when the cited source
// contains one actual <script> opening tag with both the exact asset URL and
// type="module". Generic mentions of JavaScript/module terminology are not proof.
function supportsModuleScriptLoadingClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const assetMatch = claim.match(/\b([A-Za-z0-9_-]+\.js)\b/i);
  if (
    !assetMatch ||
    !/\bmodule\b/i.test(claim) ||
    !/\bscript\b/i.test(claim)
  ) {
    return false;
  }

  const escapedAsset = assetMatch[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const sourcePattern = new RegExp(
    `\\bsrc\\s*=\\s*(["'])[^\\r\\n]*?${escapedAsset}[^\\r\\n]*?\\1`,
    "i"
  );

  return evidenceItems.some((evidence) => {
    const openingTags = evidence.match(/<script\b[^>]*>/gi) ?? [];
    return openingTags.some((tag) =>
      sourcePattern.test(tag) &&
      /\btype\s*=\s*["']module["']/i.test(tag)
    );
  });
}

// Shopify Liquid fallback-price wording is accepted only when the same quoted
// contiguous excerpt shows the blank-product condition and the exact constant
// assignment. The word "fallback" itself need not appear in source code.
function supportsBlankProductFallbackPriceClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\bfallback\b/.test(text) ||
    !/\bprice\b/.test(text) ||
    !/\b1999\b/.test(text) ||
    !/\bproduct_resource\b/.test(text) ||
    !/\bblank\b/.test(text)
  ) {
    return false;
  }

  // Gemini may quote the condition and assignment as two separate exact
  // evidence items even though they are adjacent lines in one Liquid block.
  // Require both pieces to be explicitly cited, then confirm their ordered
  // relationship inside one contiguous source segment sent to Gemini. This
  // does not stitch across retrieval gaps or unrelated source locations.
  const combinedEvidence = evidenceItems.join(" ");
  const citesBlankCondition =
    /\bif\s+product_resource\s*==\s*blank\b/i.test(combinedEvidence);
  const citesExactAssignment =
    /\bassign\s+price\s*=\s*1999\b/i.test(combinedEvidence);

  if (!citesBlankCondition || !citesExactAssignment) return false;

  const contiguousFallbackPattern =
    /\bif\s+product_resource\s*==\s*blank[\s\S]{0,220}\bassign\s+price\s*=\s*1999\b/i;

  return sourceSegments.some((segment) =>
    contiguousFallbackPattern.test(normalizeEvidenceForSourceMatch(segment))
  );
}

// Both currency branches are supported only by an actual if/else block.
function supportsCurrencyFormattingBranchesClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  const describesBothBranches =
    /\buse_currency\b/.test(text) &&
    /\bmoney_with_currency\b/.test(text) &&
    /\bmoney\b/.test(text) &&
    /\b(?:otherwise|else)\b/.test(text);
  if (!describesBothBranches) return false;

  const evidence = evidenceItems.join("\n");
  return /\bif\s+use_currency\b/i.test(evidence) &&
    /\|\s*money_with_currency\b/i.test(evidence) &&
    /\belse\b/i.test(evidence) &&
    /\|\s*money\b(?!_with_currency)/i.test(evidence);
}

// A conditional unit-price render is supported only when one contiguous
// excerpt contains both the `unit_price` + `show_unit_price` guard and the
// matching `unit-price` render before the closing `endif`.
function supportsUnitPriceRenderConditionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\bshow_unit_price\b/.test(text) ||
    !/\bunit_price\b/.test(text) ||
    !/\brender(?:s|ed|ing)?\b/.test(text)
  ) {
    return false;
  }

  const guardPattern = /\{%-?\s*if\b(?=[^%]*\bshow_unit_price\b)(?=[^%]*\b(?:selected_variant\.)?unit_price\b)[^%]*%\}/i;
  const renderPattern = /\{%-?\s*render\s+['"]unit-price['"][\s\S]{0,1600}/i;
  const citesExactGuard = evidenceItems.some((evidence) => guardPattern.test(evidence));
  const citesExactRender = evidenceItems.some((evidence) => renderPattern.test(evidence));
  if (!citesExactGuard || !citesExactRender) return false;

  const conditionalRenderPattern =
    /\{%-?\s*if\b(?=[^%]*\bshow_unit_price\b)(?=[^%]*\b(?:selected_variant\.)?unit_price\b)[^%]*%\}[\s\S]{0,6000}\{%-?\s*render\s+['"]unit-price['"][\s\S]{0,1600}\{%-?\s*endif\b[^%]*%-?\}/i;

  return sourceSegments.some((segment) =>
    conditionalRenderPattern.test(segment)
  );
}

// Verify a literal Liquid case expression without relying on generic token
// overlap. The source must contain the exact case variable named in the claim.
function supportsLiquidCaseStatementClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\bcase\b/.test(text) || !/\bvariant\b/.test(text)) return false;

  return evidenceItems.some((evidence) =>
    /\{%-?\s*case\s+variant\s*-?%\}/i.test(evidence)
  );
}

// Verify a literal `when` label (not the behavior of the branch) from the
// exact Liquid tag. This handles identifiers such as navigation_bar, whose
// underscore-separated words otherwise lose their branch semantics.
function supportsLiquidCaseBranchLabelClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  const labels = ["mobile", "navigation_bar"].filter((label) =>
    text.includes(label)
  );
  if (labels.length !== 1 || !/\b(?:case|branch|label|when)\b/i.test(text)) {
    return false;
  }

  const label = labels[0].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(
    `\\{%-?\\s*when\\s+(['"])${label}\\1\\s*-?%\\}`,
    "i"
  );
  return evidenceItems.some((evidence) => pattern.test(evidence));
}

// A default-case qualifier is valid only when the same evidence item that
// describes the operation is anchored inside an active Liquid `case` else.
// This distinguishes case-default from an `if` else or an unrelated else.
function supportsDefaultLiquidCaseBranchClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const behaviorWithoutDefaultQualifier = claim
    .replace(/\b(?:inside|within|under|in)\s+(?:the\s+)?default\s+(?:liquid\s+)?(?:case|branch|block(?:\s+case)?|path)\b/gi, " ")
    .replace(/\bdefault\s+(?:liquid\s+)?(?:case|branch|block(?:\s+case)?|path)\b/gi, " ")
    .replace(/\s+/g, " ")
    .trim();

  if (getClaimEvidenceTokens(behaviorWithoutDefaultQualifier).length === 0) {
    return false;
  }

  return evidenceItems.some((evidence) => {
    // The cited operation/subject must match this exact quote before its
    // source position can establish the active Liquid case-default branch.
    if (!hasMeaningfulTokenOverlap(behaviorWithoutDefaultQualifier, evidence)) {
      return false;
    }

    const contexts = getEvidenceContextWindows([evidence], sourceSegments);
    return contexts.some((context) =>
      getActiveLiquidBranches(context).some((frame) =>
        frame.kind === "case" && frame.branch === "else"
      )
    );
  });
}

// The transparent-header template rule is a coherent conditional behavior.
// Require the exact template guard and transparent assignment; require the
// paired text-color assignment only when the claim asserts that behavior.
function supportsTransparentHeaderTemplateRuleClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\btransparent\b/.test(text)) return false;

  const templates = [
    { name: "home", templateName: "index", setting: "enable_transparent_header_home", color: "text_color_transparent_home" },
    { name: "product", templateName: "product", setting: "enable_transparent_header_product", color: "text_color_transparent_product" },
    { name: "collection", templateName: "collection", setting: "enable_transparent_header_collection", color: "text_color_transparent_collection" },
  ];
  // Template names embedded in Shopify identifiers (for example
  // `enable_transparent_header_home`) do not have a word boundary before the
  // suffix because `_` is a word character. Match the full setting/color
  // identifiers as well as natural-language template names.
  const identifiedTemplates = templates.filter((item) =>
    new RegExp(`\\b${item.name}\\b`, "i").test(text) ||
    new RegExp(`\\b${item.setting}\\b`, "i").test(text) ||
    new RegExp(`\\b${item.color}\\b`, "i").test(text) ||
    new RegExp(`\\btemplate\\.name\\s*(?:is|==)\\s*(['\\\"])${item.templateName}\\1`, "i").test(text)
  );
  // One specialized claim may describe only one template variant. A claim
  // spanning home/product/collection must be split and verified per variant.
  if (identifiedTemplates.length !== 1) return false;
  const selected = identifiedTemplates[0];

  const guard = new RegExp(
    `\\b(?:if|elsif)\\s+section\\.settings\\.${selected.setting}\\s+and\\s+template\\.name\\s*==\\s*(['\"])${selected.templateName}\\1`,
    "i"
  );
  const transparentAssignment = /\bassign\s+transparent\s*=\s*(['"])always\1/i;
  const colorAssignment = new RegExp(
    `\\bassign\\s+transparent_text_color_setting\\s*=\\s*section\\.settings\\.${selected.color}\\b`,
    "i"
  );

  // A basic transparent-header claim needs the exact template guard and the
  // transparent assignment. The color assignment is required only when the
  // claim also asserts the corresponding text-color behavior.
  const claimAssertsColor =
    /text[\s_-]*color|transparent_text_color_setting|color\s+setting/.test(text);

  return evidenceItems.some((evidence) =>
    guard.test(evidence) &&
    transparentAssignment.test(evidence) &&
    (!claimAssertsColor || colorAssignment.test(evidence))
  );
}

// A mobile drawer_menu capture is a content_for block invocation, not a
// `when 'mobile'` Liquid case render. Verify the capture/type/variant tuple
// together so it cannot inherit support from an unrelated mobile branch.
function supportsHeaderDrawerMenuCaptureClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\bdrawer_menu\b/.test(text) ||
    !/\bmobile\b/.test(text) ||
    !/\bheader.?menu\b/.test(text)
  ) return false;

  const capturePattern =
    /\bcapture\s+drawer_menu\b[\s\S]{0,1400}\bcontent_for\s+(['"])block\1[\s\S]{0,400}\btype\s*:\s*(['"])_header-menu\2[\s\S]{0,700}\bvariant\s*:\s*(['"])mobile\3[\s\S]{0,400}\bendcapture\b/i;

  return evidenceItems.some((evidence) => capturePattern.test(evidence));
}

// Verify the exact transparent/sticky transition. The literal `not-sticky`
// value and both guard identifiers must appear together in the quoted excerpt.
function supportsTransparentNotStickyTransitionClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\btransparent\b/.test(text) ||
    !/\bsticky\b/.test(text) ||
    !/not[- ]sticky/.test(text)
  ) return false;

  const pattern = /\bif\s+transparent\s+and\s+sticky\b[\s\S]{0,500}\bassign\s+transparent\s*=\s*(['"])not-sticky\1/i;
  return evidenceItems.some((evidence) => pattern.test(evidence));
}

// A navigation_bar setting guard belongs to that case only when the literal
// case label and guard are quoted together and the source parser sees the
// guard under that active case branch.
function supportsNavigationBarBranchGuardClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\bnavigation_bar\b/.test(text) ||
    !/\bblock_settings\.navigation_bar\b/.test(text) ||
    !/\b(?:case|branch)\b/.test(text)
  ) return false;

  const labelAndGuard = evidenceItems.some((evidence) =>
    /\bwhen\s+(['"])navigation_bar\1/i.test(evidence) &&
    /\bif\s+block_settings\.navigation_bar\s*==\s*true\b/i.test(evidence)
  );
  if (!labelAndGuard) return false;

  const guardContexts = getOperationEvidenceContexts(
    sourceSegments,
    /\bif\s+block_settings\.navigation_bar\s*==\s*true\b/i
  );
  return guardContexts.some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "case" &&
      /\b(?:variant|menu_variant|navigation_variant)\b/i.test(frame.expression) &&
      /['\"]navigation_bar['\"]/i.test(frame.branchValue)
    )
  );
}

// The compare-at flag is one conditional behavior: when compare_at_price is
// greater than price, show_compare_price becomes true. Require both operations
// in one exact excerpt and in one contiguous source segment.
function supportsComparePriceFlagClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\bshow_compare_price\b/.test(text) ||
    !/\bcompare_at_price\b/.test(text) ||
    !/\bprice\b/.test(text)
  ) return false;

  const conditionPattern = /\bif\s+compare_at_price\s*>\s*price\b/i;
  const assignmentPattern = /\bassign\s+show_compare_price\s*=\s*true\b/i;
  const citedTogether = evidenceItems.some((evidence) =>
    conditionPattern.test(evidence) && assignmentPattern.test(evidence)
  );
  if (!citedTogether) return false;

  const contiguousPattern =
    /\bif\s+compare_at_price\s*>\s*price\b[\s\S]{0,320}\bassign\s+show_compare_price\s*=\s*true\b/i;
  return sourceSegments.some((segment) => contiguousPattern.test(segment));
}

// JSON theme-setting definitions are structured records. Generic prose-token
// matching incorrectly rejects valid claims because words such as "defined"
// do not appear in the JSON object. Verify the exact parsed object instead:
// its id and every scalar property/value described by the claim must match.
// This is intentionally narrow and does not relax generic evidence matching.
function supportsJsonSettingDefinitionClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const normalizedClaim = claim.toLowerCase();

  return evidenceItems.some((evidence) => {
    let parsed: unknown;
    const trimmedEvidence = evidence.trim();
    try {
      parsed = JSON.parse(trimmedEvidence);
    } catch {
      // Gemini sometimes quotes every property of a complete Liquid schema
      // setting object but omits only the surrounding `{` and `}`. Re-wrap
      // that fragment and parse it as JSON; do not repair malformed values,
      // missing properties, or incomplete nested structures.
      try {
        const propertyFragment = trimmedEvidence.replace(/,\s*$/, "");
        parsed = JSON.parse(`{${propertyFragment}}`);
      } catch {
        return false;
      }
    }

    if (
      !parsed ||
      typeof parsed !== "object" ||
      Array.isArray(parsed)
    ) {
      return false;
    }

    const definition = parsed as Record<string, unknown>;
    if (typeof definition.id !== "string" || !definition.id.trim()) {
      return false;
    }

    // The claim must identify this exact setting, not merely another object
    // with similar type/label values.
    if (!normalizedClaim.includes(definition.id.toLowerCase())) {
      return false;
    }

    // The exact setting identifier in the claim identifies the record's id;
    // requiring the literal word "id" would reject natural claims such as
    // "the enable_sticky_header setting" despite an exact source object quote.

    // If the claim names a known Shopify setting property, that property must
    // exist in the cited object. This prevents an excerpt missing `default`
    // (or another explicitly claimed field) from verifying that field.
    const knownSettingProperties = [
      "type", "label", "name", "default", "info", "options",
      "visible_if", "placeholder", "unit", "min", "max", "step",
      "accept", "content", "limit", "header", "group", "id",
    ];
    for (const property of knownSettingProperties) {
      // `header` frequently appears in a source path such as
      // sections/header.liquid; that alone does not claim a JSON `header` field.
      const propertyPattern = property === "header"
        ? /\bheader\b(?:\s+(?:property|field)|\s*[:=])/i
        : new RegExp(`\\b${property.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      if (propertyPattern.test(claim) && !Object.prototype.hasOwnProperty.call(definition, property)) {
        return false;
      }
    }

    for (const [key, value] of Object.entries(definition)) {
      if (key === "id") continue;

      const keyPattern = new RegExp(`\\b${key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i");
      const naturalTypeWording =
        key === "type" &&
        typeof value === "string" &&
        new RegExp(`\\b${String(value).replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}\\s+setting\\b`, "i").test(normalizedClaim);
      if (!keyPattern.test(claim) && !naturalTypeWording) return false;

      if (key === "options") {
        if (
          !Array.isArray(value) ||
          value.length === 0 ||
          !value.every((option) =>
            option !== null &&
            typeof option === "object" &&
            !Array.isArray(option) &&
            typeof (option as Record<string, unknown>).value === "string"
          )
        ) return false;

        const optionValues = value.map((option) =>
          String((option as Record<string, unknown>).value).toLowerCase()
        );
        if (new Set(optionValues).size !== optionValues.length) return false;
        if (!optionValues.every((optionValue) => normalizedClaim.includes(optionValue))) return false;

        const countMatch = normalizedClaim.match(
          /\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+options?\b/
        );
        if (countMatch) {
          const wordToNumber: Record<string, number> = {
            one: 1, two: 2, three: 3, four: 4, five: 5,
            six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
          };
          const claimedCount = /^\d+$/.test(countMatch[1])
            ? Number(countMatch[1])
            : wordToNumber[countMatch[1]];
          if (claimedCount !== value.length) return false;
        }
        continue;
      }

      // Keep non-options nested records unsupported until separately validated.
      if (value !== null && typeof value === "object") return false;

      const renderedValue = value === null ? "null" : String(value);
      const normalizedValue = renderedValue.toLowerCase();
      const exactValueMentioned = normalizedClaim.includes(normalizedValue);
      // Shopify Liquid schemas often store translated labels as keys such as
      // `t:settings.sticky_header`. A claim that names the exact final key
      // segment (while also identifying the label property) is sufficiently
      // specific without inventing the translated UI text.
      const translationKeySegmentMentioned =
        key === "label" &&
        /^t:[a-z0-9_-]+(?:\.[a-z0-9_-]+)+$/i.test(renderedValue) &&
        normalizedClaim.includes(renderedValue.split(".").pop()!.toLowerCase());

      // A scalar value must be associated with its own property. Merely
      // mentioning `always` somewhere in the claim is not enough to prove it
      // is the default, because the same word may also be one of the options.
      const escapedValue = normalizedValue.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const labelAssociationValue = translationKeySegmentMentioned
        ? renderedValue.split(".").pop()!.toLowerCase()
        : normalizedValue;
      const escapedLabelAssociationValue = labelAssociationValue.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const scalarAssociationPatterns: Record<string, RegExp> = {
        type: new RegExp(`\\b(?:type\\s+${escapedValue}|${escapedValue}\\s+(?:type|setting))\\b`, "i"),
        default: new RegExp(`\\b(?:default\\s+(?:value\\s+)?${escapedValue}|${escapedValue}\\s+default)\\b`, "i"),
        label: new RegExp(`(?:\\blabel\\b.{0,48}${escapedLabelAssociationValue}|${escapedLabelAssociationValue}.{0,24}\\blabel\\b)`, "i"),
      };
      const scalarAssociation = scalarAssociationPatterns[key];
      const associatedValueMentioned = scalarAssociation
        ? scalarAssociation.test(normalizedClaim)
        : new RegExp(`(?:\\b${key}\\b.{0,48}${escapedValue}|${escapedValue}.{0,24}\\b${key}\\b)`, "i").test(normalizedClaim);

      if (key === "label") {
        if (!exactValueMentioned && !translationKeySegmentMentioned) return false;
        if (!associatedValueMentioned) return false;
      } else if (!associatedValueMentioned) {
        return false;
      }
    }

    return true;
  });
}

// Section-level Shopify settings are JSON objects embedded inside a Liquid
// `{% schema %}` block. Verify the exact parsed object by its setting id and
// only the properties/values that the claim actually asserts. These settings
// must be attributed to their owning Liquid section file.
function supportsLiquidSectionSchemaSettingDefinitionClaim(
  claim: string,
  evidenceItems: string[],
  file: string
): boolean {
  if (
    !/^sections\/[a-z0-9_./-]+\.liquid$/i.test(normalizePath(file)) ||
    !/enable_sticky_header/i.test(claim)
  ) return false;

  const normalizedClaim = claim.toLowerCase();

  return evidenceItems.some((evidence) => {
    let parsed: unknown;
    const trimmedEvidence = evidence.trim();
    try {
      parsed = JSON.parse(trimmedEvidence);
    } catch {
      // Accept a complete property fragment only when wrapping it in braces
      // produces valid JSON. Do not repair malformed values or nested data.
      try {
        parsed = JSON.parse(`{${trimmedEvidence.replace(/,\s*$/, "")}}`);
      } catch {
        return false;
      }
    }

    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
    const definition = parsed as Record<string, unknown>;
    if (definition.id !== "enable_sticky_header") return false;

    if (/\btype\b/i.test(claim)) {
      if (typeof definition.type !== "string" ||
          !normalizedClaim.includes(definition.type.toLowerCase())) return false;
    }
    if (/\blabel\b/i.test(claim)) {
      if (typeof definition.label !== "string" ||
          !normalizedClaim.includes(definition.label.toLowerCase())) return false;
    }
    if (/\bdefault\b/i.test(claim)) {
      if (typeof definition.default !== "string" ||
          !normalizedClaim.includes(definition.default.toLowerCase())) return false;
    }
    if (/\boptions?\b/i.test(claim)) {
      if (!Array.isArray(definition.options) || definition.options.length === 0) return false;
      const optionValues = definition.options.map((option) => {
        if (!option || typeof option !== "object" || Array.isArray(option)) return null;
        const value = (option as Record<string, unknown>).value;
        return typeof value === "string" ? value.toLowerCase() : null;
      });
      if (optionValues.some((value) => value === null) ||
          !optionValues.every((value) => normalizedClaim.includes(value!))) return false;
    }
    return true;
  });
}

// Sticky CSS claims require the named selector and the claimed declarations
// to occur together in one quoted rule. Generic mentions of `sticky` or `top`
// cannot establish which selector owns those declarations.
function supportsStickyCssRuleClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const claimWithoutThemePaths = claim.replace(
    /\b(?:assets|blocks|config|layout|sections|snippets|templates)\/[A-Za-z0-9_./-]+\.(?:liquid|json|js|css)\b/gi,
    " "
  );
  const selectorMatch = claimWithoutThemePaths.match(/\.(?:[A-Za-z_-][A-Za-z0-9_-]*)/);
  if (!selectorMatch || !/\b(?:position|sticky|top|offset|z-index)\b/i.test(claim)) return false;

  const selector = selectorMatch[0];
  const requiresStickyPosition = /\bposition\b.{0,32}\bsticky\b|\bposition\s*:\s*sticky\b|\bsticky positioning\b/i.test(claim);
  const requiresTop = /\btop\b|\boffset\b/i.test(claim);
  const requiresZIndex = /\bz-index\b/i.test(claim);
  const requiresDesktopMedia = /\bdesktop media query\b|\bdesktop\s+query\b/i.test(claim);
  const normalizeCssEvidence = (value: string) =>
    value.replace(/\s+/g, " ").trim().toLowerCase();

  // CSS selectors may be grouped, so the claimed selector can be followed by
  // a comma (or another selector suffix) before the rule's opening brace.
  // Locate the next rule block after the selector instead of requiring
  // `selector {` with no intervening selector-list syntax.
  return sourceSegments.some((segment) => {
    const normalizedSegment = normalizeCssEvidence(segment);
    const evidenceForRule = evidenceItems.find((evidence) => {
      const normalizedEvidence = normalizeCssEvidence(evidence);
      return normalizedEvidence.length > 0 &&
        normalizedSegment.includes(normalizedEvidence) &&
        normalizedEvidence.includes(selector.toLowerCase()) &&
        (!requiresStickyPosition || /position\s*:\s*sticky\b/i.test(evidence)) &&
        (!requiresTop || /\btop\s*:/i.test(evidence)) &&
        (!requiresZIndex || /\bz-index\s*:/i.test(evidence)) &&
        (!requiresDesktopMedia || /@media\s+(?:screen\s+and\s+)?\(\s*min-width\s*:\s*750px\s*\)/i.test(evidence));
    });
    if (!evidenceForRule) return false;

    const selectorIndex = segment.toLowerCase().indexOf(selector.toLowerCase());
    if (selectorIndex < 0) return false;
    const openingBrace = segment.indexOf("{", selectorIndex + selector.length);
    if (openingBrace < 0) return false;
    const closingBraceBeforeOpen = segment.indexOf("}", selectorIndex + selector.length);
    if (closingBraceBeforeOpen >= 0 && closingBraceBeforeOpen < openingBrace) return false;
    const closingBrace = segment.indexOf("}", openingBrace + 1);
    if (closingBrace < 0) return false;

    const declarations = segment.slice(openingBrace + 1, closingBrace);
    const selectorList = segment.slice(selectorIndex, openingBrace);
    if (!selectorList.toLowerCase().includes(selector.toLowerCase())) return false;

    if (requiresStickyPosition && !/(?:^|;)\s*position\s*:\s*sticky\b/i.test(declarations)) return false;
    if (requiresTop && !/(?:^|;)\s*top\s*:\s*[^;]+/i.test(declarations)) return false;
    if (requiresZIndex && !/(?:^|;)\s*z-index\s*:\s*[^;]+/i.test(declarations)) return false;

    if (requiresDesktopMedia) {
      const mediaStart = segment.lastIndexOf("@media", selectorIndex);
      if (mediaStart < 0) return false;
      const mediaPrelude = segment.slice(mediaStart, selectorIndex);
      if (!/@media\s+(?:screen\s+and\s+)?\(\s*min-width\s*:\s*750px\s*\)/i.test(mediaPrelude)) return false;
    }

    return true;
  });
}

// Accept a natural-language description of the sticky-content class only
// when the same quoted rule and contiguous source segment bind that exact
// selector to both position: sticky and a top declaration.
function supportsStickyCssContentClassDescriptionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\bsticky[-\s]+content\s+(?:css\s+)?(?:class|classes|style|styles)\b/.test(text) ||
      !/\bposition\b/.test(text) || !/\bsticky\b/.test(text) ||
      !/\b(?:top|offset)\b/.test(text)) return false;

  const exactRule = (value: string) =>
    /\.sticky-content\b[^{}]*\{[^}]*position\s*:\s*sticky\b[^}]*\btop\s*:[^}]*\}/i.test(value);

  return evidenceItems.some((evidence) =>
    exactRule(evidence) && sourceSegments.some((segment) =>
      evidenceMatchesSource(evidence, segment) && exactRule(segment)
    )
  );
}

// v45: recognize exact sticky-header code excerpts when the model's claim
// uses equivalent but shorter wording. Every check still requires both the
// quoted evidence and the matching contiguous source segment.
function supportsStickyObserverBooleanCallDescriptionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/observeStickyPosition/i.test(claim) ||
      !/stickyMode/i.test(claim) ||
      !/\balways\b/.test(text) ||
      !/\bboolean\b/.test(text)) return false;

  const exactCall = /#observeStickyPosition\s*\(\s*stickyMode\s*===\s*["']always["']\s*\)/i;
  const quotedCallPresent = evidenceItems.some((evidence) => exactCall.test(evidence));
  const sourceShowsAttributeToBooleanFlow = sourceSegments.some((segment) =>
    /const\s+stickyMode\s*=\s*this\.getAttribute\(\s*["']sticky["']\s*\)/i.test(segment) &&
    exactCall.test(segment)
  );
  return quotedCallPresent && sourceShowsAttributeToBooleanFlow;
}

function supportsStickyScrollListenerDescriptionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/stickyMode/i.test(claim) ||
      !/scroll\s+event\s+listener/.test(text) ||
      !/scroll-up/.test(text) || !/\balways\b/.test(text)) return false;

  const exactRegistration = /if\s*\(\s*stickyMode\s*===\s*["']scroll-up["']\s*\|\|\s*stickyMode\s*===\s*["']always["']\s*\)\s*\{[\s\S]*?getScrollEventTarget\s*\(\s*\)[\s\S]*?addEventListener\s*\(\s*["']scroll["']\s*,\s*this\.#handleWindowScroll\s*\)/i;
  return evidenceItems.some((evidence) => exactRegistration.test(evidence)) &&
    sourceSegments.some((segment) => exactRegistration.test(segment));
}

function extractPrivateArrowMethodBody(source: string, methodName: string): string | null {
  const pattern = new RegExp("#" + methodName + "\\s*=\\s*\\([^)]*\\)\\s*=>\\s*\\{", "m");
  const match = pattern.exec(source);
  if (!match) return null;

  const bodyStart = match.index + match[0].length;
  const rest = source.slice(bodyStart);
  const nextMethod = rest.search(/\n\s*#[A-Za-z_$][A-Za-z0-9_$]*\s*=/);
  return nextMethod >= 0 ? rest.slice(0, nextMethod) : rest.slice(0, 16000);
}

function supportsStickyRafDescriptionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[],
  actualSource: string
): boolean {
  if (!/#handleWindowScroll/i.test(claim) ||
      !/requestAnimationFrame/i.test(claim) ||
      !/#updateScrollState/i.test(claim)) return false;

  const methodBody = extractPrivateArrowMethodBody(actualSource, "handleWindowScroll");
  if (!methodBody) return false;

  const exactOperation =
    /requestAnimationFrame\s*\(\s*\(\)\s*=>\s*\{[\s\S]*?#updateScrollState\s*\(\s*\)[\s\S]*?\}\s*\)/i;

  return evidenceItems.some((evidence) =>
    /requestAnimationFrame\s*\(/i.test(evidence) &&
    /#updateScrollState\s*\(/i.test(evidence) &&
    evidenceMatchesSource(evidence, methodBody) &&
    sourceSegments.some((segment) => evidenceMatchesSource(evidence, segment))
  ) && exactOperation.test(methodBody);
}

function supportsStickyDesktopCssDescriptionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  const describesDesktopStickyRule =
    /\bdesktop\b/.test(text) &&
    /sticky[-\s]content(?:--desktop)?/.test(text) &&
    (/\bposition\b.{0,32}\bsticky\b/.test(text) || /\bsticky\s+position(?:ing)?\b/.test(text)) &&
    /\btop\b|\boffset\b/.test(text);
  if (!describesDesktopStickyRule) return false;

  // The claim may call this a desktop sticky-content rule without saying
  // "media query". The quoted source must still prove the exact grouped
  // selector, 750px media guard, position declaration, and top offset.
  const desktopStickyRule = /@media\s+(?:screen\s+and\s*)?\(\s*min-width\s*:\s*750px\s*\)\s*\{[\s\S]*?\.sticky-content--desktop\s*,\s*\.sticky-content--desktop\.full-height--desktop\s*>\s*\.group-block\s*\{(?=[^}]*\bposition\s*:\s*sticky\s*;)(?=[^}]*\btop\s*:\s*var\(\s*--sticky-header-offset\s*,\s*0\s*\)\s*;)[^}]*\}\s*\}/i;
  return evidenceItems.some((evidence) => desktopStickyRule.test(evidence)) &&
    sourceSegments.some((segment) =>
      evidenceItems.some((evidence) =>
        evidenceMatchesSource(evidence, segment) && desktopStickyRule.test(segment)
      )
    );
}

// Verify connectedCallback's sticky-attribute-to-observer initialization flow.
// The exact quote and its placement inside the lifecycle callback are required.
function supportsStickyConnectedCallbackStickyObserverInitClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/connectedCallback/i.test(claim) || !/sticky/i.test(claim) ||
      !/#observeStickyPosition/i.test(claim)) return false;

  const exactFlow = (value: string) =>
    /const\s+stickyMode\s*=\s*this\.getAttribute\(\s*["']sticky["']\s*\)/i.test(value) &&
    /if\s*\(\s*stickyMode\s*\)\s*\{[\s\S]*?#observeStickyPosition\s*\(\s*stickyMode\s*===\s*["']always["']\s*\)/i.test(value);

  if (!evidenceItems.some(exactFlow)) return false;
  return sourceSegments.some((segment) => {
    const callback = /\bconnectedCallback\b\s*(?:=\s*)?(?:async\s*)?\([^)]*\)\s*(?:=>\s*)?\{/i.exec(segment);
    if (!callback) return false;
    const bodyStart = callback.index + callback[0].length;
    return exactFlow(segment.slice(bodyStart, bodyStart + 12000));
  });
}

// Verify the threshold ternary only when it belongs to the actual observer
// initialization inside #observeStickyPosition.
function supportsStickyObserverThresholdClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/#observeStickyPosition/i.test(claim) ||
      !/IntersectionObserver/i.test(claim) ||
      !/threshold/i.test(claim) ||
      !/alwaysSticky/i.test(claim)) return false;

  const exactThreshold = /threshold\s*:\s*alwaysSticky\s*\?\s*1\s*:\s*0/i;
  if (!evidenceItems.some((evidence) => exactThreshold.test(evidence))) return false;

  return sourceSegments.some((segment) => {
    const method = /#observeStickyPosition\s*(?:=\s*\([^)]*\)\s*=>|\([^)]*\))\s*\{/i.exec(segment);
    if (!method) return false;
    const bodyStart = method.index + method[0].length;
    const afterMethod = segment.slice(bodyStart);
    const nextPrivateMethod = afterMethod.search(/\n\s*#[A-Za-z_$][A-Za-z0-9_$]*\s*(?:=|\()/);
    const body = nextPrivateMethod >= 0 ? afterMethod.slice(0, nextPrivateMethod) : afterMethod.slice(0, 12000);
    return exactThreshold.test(body) &&
      /new\s+IntersectionObserver\s*\(/i.test(body) &&
      /\.observe\s*\(\s*this\s*\)/i.test(body);
  });
}

// v44: align sticky Liquid-setting, sticky-content CSS, and observer-property
// claims with their exact supplied source excerpts without weakening generic matching.
// Narrow explicit-evidence validators for sticky-header claims. These validate
// the exact quoted Liquid/JS constructs; they do not grant generic keyword-based
// verification. The quoted evidence is separately required to match both the
// sent context and the actual theme source by verifyAnswerClaim.
// Verify the broader statement that the Liquid section evaluates the sticky
// setting. The quote and contiguous source must show the actual setting
// comparisons; merely mentioning the setting name is not enough.
function supportsStickyHeaderSettingEvaluationClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\benable_sticky_header\b/.test(text) ||
      !/\b(?:evaluat\w*|read\w*|check\w*|control\w*|determin\w*)\b/.test(text)) return false;

  const hasBothBranches = (value: string) =>
    /if\s+section\.settings\.enable_sticky_header\s*==\s*['"]always['"]/i.test(value) &&
    /elsif\s+section\.settings\.enable_sticky_header\s*==\s*['"]scroll-up['"]/i.test(value);

  return evidenceItems.some(hasBothBranches) && sourceSegments.some(hasBothBranches);
}

// The connectedCallback excerpt must read the sticky attribute within that
// lifecycle method, not merely somewhere else in HeaderComponent.
function supportsStickyConnectedCallbackAttributeReadClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/connectedcallback/.test(text) || !/sticky attribute/.test(text) ||
      !/\b(?:read|reads|reading|getattribute)\b/.test(text)) return false;

  const hasQuotedRead = evidenceItems.some((evidence) =>
    /getAttribute\s*\(\s*['"]sticky['"]\s*\)/i.test(evidence)
  );
  if (!hasQuotedRead) return false;

  return sourceSegments.some((segment) => {
    const callback = /connectedCallback\s*=\s*\(\)\s*=>\s*\{|connectedCallback\s*\(\s*\)\s*\{/i.exec(segment);
    if (!callback) return false;
    const bodyStart = callback.index + callback[0].length;
    const body = segment.slice(bodyStart, bodyStart + 8000);
    return /getAttribute\s*\(\s*['"]sticky['"]\s*\)/i.test(body);
  });
}

// Verify that the boolean passed to #observeStickyPosition is derived from
// the actual sticky attribute value, rather than accepting a generic method
// mention or an unrelated boolean argument.
function supportsStickyAttributeBooleanObserverCallClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/#observeStickyPosition/i.test(claim) || !/\bboolean\b/.test(text) ||
      !/sticky attribute/.test(text)) return false;

  const hasExactCall = (value: string) =>
    /getAttribute\s*\(\s*['"]sticky['"]\s*\)/i.test(value) &&
    /#observeStickyPosition\s*\(\s*stickyMode\s*===\s*['"]always['"]\s*\)/i.test(value);
  return evidenceItems.some(hasExactCall) && sourceSegments.some(hasExactCall);
}

// A scroll-listener claim is accepted only when the excerpt proves the mode
// guard, resolves the scroll target, and registers the exact handler.
function supportsStickyScrollContainerListenerClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/scroll container/.test(text) || !/scroll event listener/.test(text) ||
      !/stickyMode/i.test(claim) || !/scroll-up/.test(text) || !/always/.test(text)) return false;

  const hasExactRegistration = (value: string) =>
    /if\s*\(\s*stickyMode\s*===\s*['"]scroll-up['"]\s*\|\|\s*stickyMode\s*===\s*['"]always['"]\s*\)/i.test(value) &&
    /getScrollEventTarget\s*\(\s*\)/i.test(value) &&
    /addEventListener\s*\(\s*['"]scroll['"]\s*,\s*this\.#handleWindowScroll\s*\)/i.test(value);
  return evidenceItems.some(hasExactRegistration) && sourceSegments.some(hasExactRegistration);
}

// The dataset claim must be backed by an assignment inside the observed
// sticky-position callback and be present verbatim in both supplied evidence
// and the contiguous source segment.
function supportsStickyIntersectionObserverDatasetUpdateClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/intersectionobserver/.test(text) || !/dataset\.stickystate/.test(text) ||
      !/\b(?:update|updates|assign|sets?)\b/.test(text)) return false;

  const hasStateAssignment = (value: string) =>
    /if\s*\(\s*alwaysSticky\s*\)/i.test(value) &&
    /dataset\.stickyState\s*=\s*isIntersecting\s*\?\s*['"]inactive['"]\s*:\s*['"]active['"]/i.test(value);

  return evidenceItems.some(hasStateAssignment) && sourceSegments.some((segment) =>
    /#observeStickyPosition\s*=\s*\([^)]*\)\s*=>\s*\{/i.test(segment) &&
    /new\s+IntersectionObserver\s*\(/i.test(segment) &&
    hasStateAssignment(segment)
  );
}

function recoverExactStickyLiquidBranchEvidence(
  claim: string,
  sourceSegments: string[]
): string | null {
  const text = claim.toLowerCase();
  if (!/enable_sticky_header/.test(text) || !/\bsticky\b/.test(text)) return null;
  const mode = /\bscroll-up\b/.test(text)
    ? "scroll-up"
    : /\balways\b/.test(text)
      ? "always"
      : null;
  if (!mode) return null;

  const branch = mode === "always"
    ? /^\s*if\s+section\.settings\.enable_sticky_header\s*==\s*['"]always['"]\s*$/i
    : /^\s*elsif\s+section\.settings\.enable_sticky_header\s*==\s*['"]scroll-up['"]\s*$/i;
  const assignment = mode === "always"
    ? /^\s*assign\s+sticky\s*=\s*['"]always['"]\s*$/i
    : /^\s*assign\s+sticky\s*=\s*['"]scroll-up['"]\s*$/i;

  for (const segment of sourceSegments) {
    const lines = segment.split(/\r?\n/);
    for (let start = 0; start < lines.length; start++) {
      if (!branch.test(lines[start])) continue;
      for (let end = start + 1; end < Math.min(lines.length, start + 8); end++) {
        if (/^\s*(?:elsif|else|endif)\b/i.test(lines[end])) break;
        if (assignment.test(lines[end])) {
          // Return the exact contiguous source lines; never synthesize Liquid tags.
          return lines.slice(start, end + 1).join("\n");
        }
      }
    }
  }
  return null;
}

function supportsStickyHeaderLiquidSettingAssignmentClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/enable_sticky_header/.test(text) || !/\bsticky\b/.test(text)) return false;

  // Verify one atomic setting branch at a time. Shopify's {% liquid %} tag
  // contains plain "if"/"elsif" statements; do not require invented wrappers.
  const mode = /\bscroll-up\b/.test(text)
    ? "scroll-up"
    : /\balways\b/.test(text)
      ? "always"
      : null;
  if (!mode) return false;

  const branchPattern = mode === "always"
    ? /(?:\{%-?\s*)?if\s+section\.settings\.enable_sticky_header\s*==\s*['"]always['"]/i
    : /(?:\{%-?\s*)?elsif\s+section\.settings\.enable_sticky_header\s*==\s*['"]scroll-up['"]/i;
  const assignmentPattern = mode === "always"
    ? /assign\s+sticky\s*=\s*['"]always['"]/i
    : /assign\s+sticky\s*=\s*['"]scroll-up['"]/i;
  const provesBranch = (value: string) =>
    branchPattern.test(value) && assignmentPattern.test(value);

  return evidenceItems.some(provesBranch) &&
    sourceSegments.some(provesBranch);
}

function supportsStickyHeaderAttributeOutputClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\bsticky\b/.test(text) || !/header component|header-component/.test(text) ||
      !/attribute/.test(text)) return false;
  return evidenceItems.some((evidence) =>
    /{%\s*if\s+sticky\s*%}/i.test(evidence) &&
    /sticky\s*=\s*["']{{\s*sticky\s*}}["']/i.test(evidence) &&
    /{%\s*endif\s*%}/i.test(evidence)
  );
}

function supportsStickyIntersectionObserverFieldClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  const identifiesObserverProperty = /intersectionobserver/.test(text) &&
    /\b(?:property|field|instance\s+property)\b/.test(text);
  if (!identifiesObserverProperty) return false;

  const hasTypedFieldEvidence = (evidence: string) =>
    /intersection observer for monitoring sticky header position/i.test(evidence) &&
    /@type\s*\{\s*IntersectionObserver\s*\|\s*null\s*\}/i.test(evidence) &&
    /#intersectionObserver\s*=\s*null\s*;/i.test(evidence);

  return evidenceItems.some(hasTypedFieldEvidence) && sourceSegments.some((segment) =>
    /class\s+HeaderComponent\b/.test(segment) &&
    /#intersectionObserver\s*=\s*null\s*;/i.test(segment)
  );
}

function supportsStickyIntersectionObserverBehaviorClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/intersectionobserver/.test(text) || !/sticky position/.test(text) ||
      !/alwaysSticky/.test(claim)) return false;
  return evidenceItems.some((evidence) =>
    /new\s+IntersectionObserver\s*\(/i.test(evidence) &&
    /if\s*\(\s*alwaysSticky\s*\)/i.test(evidence) &&
    /dataset\.stickyState\s*=/.test(evidence) &&
    /#offscreen\s*=/.test(evidence)
  );
}

function supportsStickyScrollRafHandlerClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/#handleWindowScroll/i.test(claim) || !/requestanimationframe/.test(text) ||
      !/scroll event|scroll (?:state )?updates?/.test(text)) return false;
  return evidenceItems.some((evidence) =>
    /#handleWindowScroll\s*=\s*\(\)\s*=>\s*\{/i.test(evidence) &&
    /requestAnimationFrame\s*\(/i.test(evidence) &&
    /#updateScrollState\s*\(/i.test(evidence)
  );
}

function supportsStickyConnectedCallbackBreakpointListenerClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/connectedCallback/i.test(claim) ||
      !/change event listener/i.test(claim) ||
      !/scrollContainerMediaQuery/i.test(claim) ||
      !/breakpoint/i.test(claim)) return false;

  const listener =
    /scrollContainerMediaQuery\.addEventListener\s*\(\s*['"]change['"]\s*,\s*this\.#handleBreakpointChange\s*\)/i;

  return evidenceItems.some((evidence) => listener.test(evidence)) &&
    sourceSegments.some((segment) =>
      /connectedCallback\s*\(\s*\)\s*\{/.test(segment) &&
      listener.test(segment)
    );
}

function supportsStickyConnectedCallbackSetupClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/connectedCallback/i.test(claim) || !/scroll event listener/.test(text) && !/scroll listeners/.test(text) ||
      !/intersection observer/.test(text) || !/sticky mode/.test(text)) return false;
  return evidenceItems.some((evidence) =>
    /getAttribute\s*\(\s*['"]sticky['"]\s*\)/i.test(evidence) &&
    /#observeStickyPosition\s*\(/i.test(evidence) &&
    /addEventListener\s*\(\s*['"]scroll['"]/i.test(evidence) &&
    /scrollContainerMediaQuery\.addEventListener/i.test(evidence)
  );
}

// Exact validators for the concrete HeaderComponent excerpts commonly returned
// by the planner. Each validator requires the quoted source operations; these
// are not generic keyword exemptions.
function supportsStickyBreakpointAttributeReadClaim(claim: string, evidenceItems: string[]): boolean {
  if (!/#handleBreakpointChange/i.test(claim) || !/sticky attribute/i.test(claim)) return false;
  return evidenceItems.some((evidence) =>
    /#handleBreakpointChange\s*=\s*\(\)\s*=>\s*\{/i.test(evidence) &&
    /getAttribute\s*\(\s*['"]sticky['"]\s*\)/i.test(evidence)
  );
}

function supportsStickyBreakpointObserverResetClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[],
  actualSource: string
): boolean {
  if (!/#handleBreakpointChange/i.test(claim) ||
      !/intersection observer/i.test(claim) ||
      !/disconnect/i.test(claim) ||
      !/recreat/i.test(claim)) return false;

  const methodBody = extractPrivateArrowMethodBody(actualSource, "handleBreakpointChange");
  if (!methodBody) return false;

  const provesReset = (value: string) =>
    /#intersectionObserver\s*\?\.\s*disconnect\s*\(\s*\)/i.test(value) &&
    /#intersectionObserver\s*=\s*null/i.test(value) &&
    /#observeStickyPosition\s*\(\s*stickyMode\s*===\s*['"]always['"]\s*\)/i.test(value);

  return evidenceItems.some((evidence) =>
    provesReset(evidence) &&
    evidenceMatchesSource(evidence, methodBody) &&
    sourceSegments.some((segment) => evidenceMatchesSource(evidence, segment))
  ) && provesReset(methodBody);
}

function supportsStickyObserverStateOrOffscreenBranchClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!(/observer callback/.test(text) || /#observeStickyPosition/i.test(claim)) ||
      !/alwayssticky/.test(text) ||
      !/sticky state/.test(text) ||
      !/offscreen state/.test(text)) return false;

  const pairedAssignments =
    /if\s*\(\s*alwaysSticky\s*\)\s*\{[\s\S]{0,1800}?this\.dataset\.stickyState\s*=\s*isIntersecting\s*\?\s*['"]inactive['"]\s*:\s*['"]active['"][\s\S]{0,500}?\}\s*else\s*\{\s*this\.#offscreen\s*=\s*!isIntersecting\s*\|\|\s*this\.dataset\.stickyState\s*===\s*['"]active['"]/i;

  return evidenceItems.some((evidence) => pairedAssignments.test(evidence)) &&
    sourceSegments.some((segment) => pairedAssignments.test(segment));
}

function supportsStickyObserverThemeColorClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/#observeStickyPosition/i.test(claim) ||
      !/alwaysSticky/i.test(claim) ||
      !/theme.?color/i.test(claim) ||
      !/meta theme color/i.test(claim)) return false;

  const themeColorOperation =
    /if\s*\(\s*this\.dataset\.themeColor\s*\)\s*changeMetaThemeColor\s*\(\s*this\.dataset\.themeColor\s*\)/i;

  const hasMethodContext = (segment: string) => {
    const method = /#observeStickyPosition\s*=\s*\([^)]*\)\s*=>\s*\{/.exec(segment);
    if (!method) return false;
    const bodyStart = method.index + method[0].length;
    const afterMethod = segment.slice(bodyStart);
    const nextPrivateMethod = afterMethod.search(/\n\s*#[A-Za-z_$][A-Za-z0-9_$]*\s*=/);
    const body = nextPrivateMethod >= 0
      ? afterMethod.slice(0, nextPrivateMethod)
      : afterMethod.slice(0, 12000);

    return /if\s*\(\s*alwaysSticky\s*\)/i.test(body) &&
      themeColorOperation.test(body);
  };

  return evidenceItems.some((evidence) => themeColorOperation.test(evidence)) &&
    sourceSegments.some(hasMethodContext);
}

function supportsStickyIntersectionStateAssignmentClaim(claim: string, evidenceItems: string[]): boolean {
  const text = claim.toLowerCase();
  if (!(/intersection observer callback/.test(text) || /#observestickyposition/i.test(claim)) ||
      !/dataset\.stickystate/.test(text) || !/(?:isintersecting|intersection)/.test(text) ||
      !/alwayssticky/.test(text)) return false;
  return evidenceItems.some((evidence) =>
    /if\s*\(\s*alwaysSticky\s*\)/i.test(evidence) &&
    /dataset\.stickyState\s*=\s*isIntersecting\s*\?\s*['"]inactive['"]\s*:\s*['"]active['"]/i.test(evidence)
  );
}

function supportsStickyOffscreenAssignmentClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[],
  actualSource: string
): boolean {
  const text = claim.toLowerCase();
  if (!(/intersection\s*observer callback/.test(text) ||
        /intersection.?observer/.test(text) ||
        /#observeStickyPosition/i.test(claim)) ||
      !/#offscreen/.test(claim) ||
      !/alwayssticky/.test(text) ||
      !/false/.test(text)) return false;

  const methodBody = extractPrivateArrowMethodBody(actualSource, "observeStickyPosition");
  if (!methodBody) return false;

  const pairedBranch =
    /if\s*\(\s*alwaysSticky\s*\)\s*\{[\s\S]{0,1800}?\}\s*else\s*\{\s*this\.#offscreen\s*=\s*!isIntersecting\s*\|\|\s*this\.dataset\.stickyState\s*===\s*['"]active['"]/i;
  const quotedElseAssignment =
    /else\s*\{\s*this\.#offscreen\s*=\s*!isIntersecting\s*\|\|\s*this\.dataset\.stickyState\s*===\s*['"]active['"]/i;

  return evidenceItems.some((evidence) =>
    quotedElseAssignment.test(evidence) &&
    evidenceMatchesSource(evidence, methodBody) &&
    sourceSegments.some((segment) => evidenceMatchesSource(evidence, segment))
  ) && pairedBranch.test(methodBody);
}

function supportsStickyScrollDirectionAssignmentClaim(claim: string, evidenceItems: string[]): boolean {
  const text = claim.toLowerCase();
  if (!/#updateScrollState/i.test(claim) || !/dataset\.scrolldirection/.test(text) ||
      !/stickyMode/i.test(claim) || !/always/.test(text)) return false;
  return evidenceItems.some((evidence) =>
    /if\s*\(\s*stickyMode\s*===\s*['"]always['"]\s*\)/i.test(evidence) &&
    /dataset\.scrollDirection\s*=\s*['"]none['"]/i.test(evidence) &&
    /dataset\.scrollDirection\s*=\s*['"]up['"]/i.test(evidence) &&
    /dataset\.scrollDirection\s*=\s*['"]down['"]/i.test(evidence)
  );
}

function supportsStickyBreakpointRebindClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/#handleBreakpointChange/i.test(claim) ||
      !/rebind|re-register|register.*again/i.test(claim) ||
      !/scroll listener/i.test(claim) ||
      !/intersectionobserver/i.test(claim)) return false;

  const provesRebind = (value: string) =>
    /#scrollContainer\.removeEventListener\s*\(\s*['"]scroll['"]/i.test(value) &&
    /#scrollContainer\s*=\s*getScrollEventTarget\s*\(\s*\)/i.test(value) &&
    /#scrollContainer\.addEventListener\s*\(\s*['"]scroll['"]/i.test(value) &&
    /#intersectionObserver\?\.disconnect\s*\(\s*\)/i.test(value) &&
    /#intersectionObserver\s*=\s*null/i.test(value) &&
    /#observeStickyPosition\s*\(\s*stickyMode\s*===\s*['"]always['"]\s*\)/i.test(value);

  return evidenceItems.some(provesRebind) &&
    sourceSegments.some((segment) =>
      /#handleBreakpointChange\s*=\s*\(\)\s*=>\s*\{/.test(segment) &&
      provesRebind(segment)
    );
}

function supportsStickyDisconnectedCleanupClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/disconnectedCallback/i.test(claim) ||
      !/observer/i.test(claim) ||
      !/event listener/i.test(claim) ||
      !/header.height|header-height/i.test(claim)) return false;

  const provesCleanup = (value: string) =>
    /disconnectedCallback\s*\(\s*\)\s*\{/.test(value) &&
    /#resizeObserver\.disconnect\s*\(\s*\)/i.test(value) &&
    /#intersectionObserver\?\.disconnect\s*\(\s*\)/i.test(value) &&
    /removeEventListener\s*\(/i.test(value) &&
    /--header-height['"]\s*,\s*['"]0px['"]/i.test(value);

  return evidenceItems.some(provesCleanup) &&
    sourceSegments.some(provesCleanup);
}

// A claim that IntersectionObserver is installed by #observeStickyPosition
// needs both the constructor in cited evidence and the constructor inside
// that same method in the contiguous source segment sent to the model.
function supportsStickyHeaderObserverClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/#observeStickyPosition/i.test(claim) || !/\bIntersectionObserver\b/i.test(claim)) return false;
  if (!evidenceItems.some((evidence) => /new\s+IntersectionObserver\s*\(/i.test(evidence))) return false;

  const methodPattern = /#observeStickyPosition\s*=\s*\([^)]*\)\s*=>\s*\{/;
  return sourceSegments.some((segment) => {
    const method = methodPattern.exec(segment);
    if (!method) return false;
    const bodyStart = method.index + method[0].length;
    const afterMethodStart = segment.slice(bodyStart);
    const nextPrivateMethod = afterMethodStart.search(/\n\s*#[A-Za-z_$][A-Za-z0-9_$]*\s*=/);
    const methodBody = nextPrivateMethod >= 0
      ? afterMethodStart.slice(0, nextPrivateMethod)
      : afterMethodStart.slice(0, 12000);
    return /new\s+IntersectionObserver\s*\(/i.test(methodBody);
  });
}

// Verify sticky scroll-state claims only when the quoted evidence and one
// contiguous supplied source segment contain the exact private method plus
// the state/direction operations described by the claim. This is deliberately
// structural: generic words such as "sticky" or "scroll" are not sufficient.
function supportsStickyHeaderScrollStateClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/#updateScrollState/i.test(claim)) return false;
  const claimText = claim.toLowerCase();
  const mentionsState = /\b(?:state|sticky|active|dataset)\b/.test(claimText);
  const mentionsDirection = /\b(?:direction|scrolling|scroll)\b/.test(claimText);
  if (!mentionsState || !mentionsDirection) return false;

  const methodPattern = /#updateScrollState\s*=\s*\(\)\s*=>\s*\{/;
  return sourceSegments.some((segment) => {
    const method = methodPattern.exec(segment);
    if (!method) return false;
    const bodyStart = method.index + method[0].length;
    const afterMethodStart = segment.slice(bodyStart);
    const nextPrivateMethod = afterMethodStart.search(/\n\s*#[A-Za-z_$][A-Za-z0-9_$]*\s*=/);
    const methodBody = nextPrivateMethod >= 0
      ? afterMethodStart.slice(0, nextPrivateMethod)
      : afterMethodStart.slice(0, 12000);

    const hasStickyMode = /getAttribute\s*\(\s*["']sticky["']\s*\)/i.test(methodBody);
    const hasScrollPosition = /getScrollTop\s*\(/i.test(methodBody);
    const hasDirectionState = /dataset\.scrollDirection\s*=/.test(methodBody);
    const hasStickyState = /dataset\.stickyState\s*=/.test(methodBody);
    const hasEvidenceQuote = evidenceItems.some((evidence) =>
      /#updateScrollState/i.test(evidence) &&
      /dataset\.scrollDirection\s*=/.test(evidence) &&
      /dataset\.stickyState\s*=/.test(evidence)
    );

    return hasStickyMode && hasScrollPosition && hasDirectionState &&
      hasStickyState && hasEvidenceQuote;
  });
}
// Product-card currency settings are simple truthy Liquid guards rather than
// equality comparisons. Verify the exact setting-to-flag assignment pair.
function supportsUseCurrencySettingAssignmentClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\buse_currency\b/.test(text)) return false;

  const productPages = /currency_code_enabled_product_pages/.test(text);
  const productCards = /currency_code_enabled_product_cards/.test(text);
  if (productPages === productCards) return false;

  const setting = productPages
    ? "currency_code_enabled_product_pages"
    : "currency_code_enabled_product_cards";
  const settingPattern = new RegExp(`\\bsettings\\.${setting}\\b`, "i");
  const assignmentPattern = /\bassign\s+use_currency\s*=\s*true\b/i;

  return evidenceItems.some((evidence) =>
    settingPattern.test(evidence) && assignmentPattern.test(evidence)
  );
}

// Strict evidence helper retained for claims that explicitly describe the
// complete `if is_product_card` / `else` block. A narrower product-page
// quantity-break claim can instead quote its exact nested check + assignment;
// the active `else` branch is then independently checked against the same
// contiguous source segment sent to Gemini (see supportsVolumePricingDetectionClaim).
function evidenceProvesCompleteProductPageQuantityBreakBlock(
  evidenceItems: string[]
): boolean {
  const completeBlockPattern =
    /\bif\s+is_product_card\b[\s\S]{0,6000}\belse\b[\s\S]{0,3000}\bif\s+selected_variant\.quantity_price_breaks\.size\s*>\s*0\b[\s\S]{0,1200}\bassign\s+has_volume_pricing\s*=\s*true\b[\s\S]{0,800}\bendif\b[\s\S]{0,800}\bendif\b/i;

  return evidenceItems.some((evidence) =>
    completeBlockPattern.test(evidence)
  );
}

// Verify the product-page quantity-break condition even when the model calls
// it an `else`-branch check rather than explicitly saying "volume pricing".
// The evidence quote itself must contain the complete outer `if/else` block.
function supportsProductPageQuantityBreakBranchClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/selected_variant\.quantity_price_breaks\.size/.test(text) ||
    !/\belse\b/.test(text)
  ) return false;

  const checkPattern = /\bif\s+selected_variant\.quantity_price_breaks\.size\s*>\s*0\b/i;
  const assignmentPattern = /\bassign\s+has_volume_pricing\s*=\s*true\b/i;
  const citesBoth = evidenceItems.some((evidence) =>
    checkPattern.test(evidence) && assignmentPattern.test(evidence)
  );
  if (!citesBoth || !evidenceProvesCompleteProductPageQuantityBreakBlock(evidenceItems)) {
    return false;
  }

  const assignmentContexts = getOperationEvidenceContexts(
    sourceSegments,
    assignmentPattern
  );
  return assignmentContexts.some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "if" &&
      frame.branch === "else" &&
      /\bis_product_card\b/i.test(frame.expression)
    )
  );
}

// Product cards and product pages use two different, independently
// verifiable quantity-break checks. Do not require both excerpts for either
// claim: each claim must be proved by its own branch and exact source context.
function supportsVolumePricingDetectionClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (!/\bvolume\s+pricing\b|\bhas_volume_pricing\b/i.test(text)) return false;
  if (/quantity_price_breaks_configured\?\s*\.\s*[a-z]{2,}/i.test(text)) return false;

  const productCardClaim = /\bproduct\s+cards?\b|\bis_product_card\b/i.test(text);
  const productPageClaim = /\bproduct\s+pages?\b/i.test(text);

  const cardAssignmentPattern =
    /\bassign\s+has_volume_pricing\s*=\s*product_resource\.quantity_price_breaks_configured\?(?:\s*\|\s*default\s*:\s*false)?/i;
  const variantCheckPattern =
    /\bif\s+selected_variant\.quantity_price_breaks\.size\s*>\s*0\b/i;
  const variantAssignmentPattern =
    /\bassign\s+has_volume_pricing\s*=\s*true\b/i;

  const citesCardAssignment = evidenceItems.some((evidence) =>
    cardAssignmentPattern.test(evidence)
  );
  const citesCardGuard = evidenceItems.some((evidence) =>
    /\bif\s+is_product_card\b/i.test(evidence)
  );
  const cardAssignmentContexts = getOperationEvidenceContexts(
    sourceSegments,
    cardAssignmentPattern
  );
  const cardAssignmentIsInsideProductCardBranch = cardAssignmentContexts.some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "if" &&
      frame.branch !== "else" &&
      /\bis_product_card\b/i.test(frame.expression)
    )
  );

  const citesVariantCheckAndAssignment = evidenceItems.some((evidence) =>
    variantCheckPattern.test(evidence) && variantAssignmentPattern.test(evidence)
  );
  // Evidence must quote the exact nested selected-variant condition and true
  // assignment together. The enclosing product-card `else` is verified from
  // the same contiguous source segment, rather than requiring Gemini to repeat
  // the entire outer block in its evidence quote.
  const productCardElseScopeInSentContext = sourceSegments.some((segment) =>
    /\bif\s+is_product_card\b[\s\S]{0,6000}\belse\b/i.test(segment)
  );
  const variantAssignmentContexts = getOperationEvidenceContexts(
    sourceSegments,
    variantAssignmentPattern
  );
  const variantAssignmentIsInsideProductPageBranch = variantAssignmentContexts.some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "if" &&
      frame.branch === "else" &&
      /\bis_product_card\b/i.test(frame.expression)
    )
  );

  if (productCardClaim) {
    return citesCardAssignment && citesCardGuard &&
      cardAssignmentIsInsideProductCardBranch;
  }
  if (productPageClaim) {
    return citesVariantCheckAndAssignment &&
      productCardElseScopeInSentContext && variantAssignmentIsInsideProductPageBranch;
  }

  return (citesCardAssignment && citesCardGuard &&
      cardAssignmentIsInsideProductCardBranch) ||
    (citesVariantCheckAndAssignment &&
      productCardElseScopeInSentContext && variantAssignmentIsInsideProductPageBranch);
}

function supportsMoneyElseBranchClaim(
  claim: string,
  evidenceItems: string[],
  contexts: EvidenceContextWindow[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\buse_currency\b/.test(text) ||
    !/\belse\b/.test(text) ||
    !/\bmoney\b/.test(text) ||
    /\bmoney_with_currency\b/.test(text)
  ) return false;

  const evidenceContainsPairedBranch = evidenceItems.some((item) => {
    const condition = item.match(/\bif\s+use_currency\b/i);
    const elseBranch = item.match(/\belse\b/i);
    const moneyFilter = item.match(/\|\s*money\b(?!_with_currency)/i);
    if (
      !condition || !elseBranch || !moneyFilter ||
      condition.index! >= elseBranch.index! ||
      elseBranch.index! >= moneyFilter.index!
    ) return false;

    const betweenConditionAndElse = item.slice(
      condition.index! + condition[0].length,
      elseBranch.index!
    );
    const betweenElseAndMoney = item.slice(
      elseBranch.index! + elseBranch[0].length,
      moneyFilter.index!
    );
    const nestedControlFlow = /\b(?:if|elsif|unless|case|when|endif|endcase)\b/i;
    return !nestedControlFlow.test(betweenConditionAndElse) &&
      !nestedControlFlow.test(betweenElseAndMoney);
  });

  return evidenceContainsPairedBranch &&
    contextProvesFalseCurrencyBranch(contexts, sourceSegments);
}

function supportsTrueCurrencyBranchClaim(
  claim: string,
  evidenceItems: string[],
  contexts: EvidenceContextWindow[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  const evidence = evidenceItems.join("\n");
  const hasCurrencyFilter = /\|\s*money_with_currency\b/i.test(evidence);

  return /\buse_currency\b/.test(text) &&
    /\btrue\b/.test(text) &&
    /\bmoney_with_currency\b/.test(text) &&
    hasCurrencyFilter &&
    contextProvesTrueCurrencyBranch(contexts, sourceSegments);
}

function supportsFalseCurrencyBranchClaim(
  claim: string,
  evidenceItems: string[],
  contexts: EvidenceContextWindow[],
  sourceSegments: string[] = []
): boolean {
  const text = claim.toLowerCase();
  const evidence = evidenceItems.join("\n");
  const hasMoneyFilter = /\|\s*money\b(?!_with_currency)/i.test(evidence);
  const citesElseBranch = /\belse\b/i.test(evidence);

  // Preferred proof: the exact evidence excerpt itself includes the paired
  // `if use_currency` condition, the corresponding else branch, and the money
  // filter operation. Reject nested control-flow between those tokens so an
  // unrelated inner else cannot be mistaken for the currency branch.
  const evidenceContainsPairedBranch = evidenceItems.some((item) => {
    const condition = item.match(/\bif\s+use_currency\b/i);
    const elseBranch = item.match(/\belse\b/i);
    const moneyFilter = item.match(/\|\s*money\b(?!_with_currency)/i);
    if (
      !condition ||
      !elseBranch ||
      !moneyFilter ||
      condition.index! >= elseBranch.index! ||
      elseBranch.index! >= moneyFilter.index!
    ) {
      return false;
    }

    const conditionToElse = item.slice(
      condition.index! + condition[0].length,
      elseBranch.index!
    );
    const elseToMoney = item.slice(
      elseBranch.index! + elseBranch[0].length,
      moneyFilter.index!
    );
    const nestedControlFlow = /\b(?:if|elsif|unless|case|when|endif|endcase)\b/i;

    return !nestedControlFlow.test(conditionToElse) &&
      !nestedControlFlow.test(elseToMoney);
  });

  return /\buse_currency\b/.test(text) &&
    /\bfalse\b/.test(text) &&
    hasMoneyFilter &&
    citesElseBranch &&
    (evidenceContainsPairedBranch ||
      contextProvesFalseCurrencyBranch(contexts, sourceSegments));
}

// A narrow statement that only reports the literal Liquid case label is
// verifiable from the label itself. It must not imply drawer rendering or
// other mobile-navigation behavior unless the separate render verifier passes.
function supportsMobileCaseLabelClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const text = claim.toLowerCase();
  const describesLiteralLabel =
    /\b(?:case|branch|label|condition|variant)\b/.test(text) &&
    /\bmobile\b/.test(text) &&
    /\bwhen\b/.test(text);
  const addsBehavior =
    /\b(?:renders?|handles?|displays?|shows?|opens?|loads?|outputs?)\b/.test(text) ||
    /\bnavigation\b/.test(text);

  return describesLiteralLabel && !addsBehavior &&
    evidenceItems.some((evidence) => /\bwhen\s+(['"])mobile\1/i.test(evidence));
}

// A data attribute containing "mobile" does not prove a mobile-only branch.
// Require the quoted evidence to show both the mobile `when` tag and the
// header-drawer render, then inspect the active case branch at that exact
// render location in the same contiguous source segment sent to Gemini.
function supportsMobileVariantRenderClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  const text = claim.toLowerCase();
  if (
    !/\bmobile\b/.test(text) ||
    !/\b(?:variant|case|branch)\b/.test(text) ||
    !/\brender(?:s|ed|ing)?\b/.test(text) ||
    !/\bheader-drawer(?:\.liquid)?\b/.test(text)
  ) {
    return false;
  }

  const citesMobileBranchAndDrawerRender = evidenceItems.some((evidence) =>
    /\bwhen\s+(['"])mobile\1/i.test(evidence) &&
    /\{%-?\s*render\s+['"]header-drawer['"](?=\s|,|%|})/i.test(evidence)
  );
  if (!citesMobileBranchAndDrawerRender) return false;

  const drawerRenderContexts = getOperationEvidenceContexts(
    sourceSegments,
    /\{%-?\s*render\s+['"]header-drawer['"](?=\s|,|%|})/i
  );

  return drawerRenderContexts.some((context) =>
    getActiveLiquidBranches(context).some((frame) =>
      frame.kind === "case" &&
      /\b(?:variant|menu_variant|navigation_variant)\b/i.test(frame.expression) &&
      /['"]mobile['"]/i.test(frame.branchValue)
    )
  );
}

// Conditional claims must be backed by the active Liquid branch that contains
// the cited source excerpt, not merely by an unrelated if/else elsewhere.
function claimAddsUnsupportedControlFlow(
  claim: string,
  evidenceItems: string[],
  contexts: EvidenceContextWindow[] = [],
  sourceSegments: string[] = []
): boolean {
  const text = claim.toLowerCase();
  const evidence = evidenceItems.join("\n").toLowerCase();

  const positionQualifiers = [
    /\b(?:at\s+the\s+)?top\s+of\s+(?:the\s+)?(?:file|template|section|block|snippet)\b/i,
    /\b(?:at\s+the\s+)?bottom\s+of\s+(?:the\s+)?(?:file|template|section|block|snippet)\b/i,
    /\b(?:first|last)\s+(?:line|statement|script|block|section|item|branch)\b/i,
    /\b(?:before|after)\s+(?:the\s+)?[a-z0-9_-]+/i,
  ];
  const hasPositionQualifier = positionQualifiers.some((pattern) => pattern.test(text));
  if (hasPositionQualifier) {
    const positionIsExplicitlyQuoted = positionQualifiers.some((pattern) => pattern.test(evidence));
    if (!positionIsExplicitlyQuoted) return true;
  }

  const claimsNavigationBarGuardBranch =
    /\bnavigation_bar\b/.test(text) &&
    /\bblock_settings\.navigation_bar\b/.test(text) &&
    /\b(?:case|branch)\b/.test(text);
  if (claimsNavigationBarGuardBranch) {
    return !supportsNavigationBarBranchGuardClaim(
      claim, evidenceItems, sourceSegments
    );
  }

  const claimsDefaultLiquidCase =
    /\bdefault\b.{0,50}\b(?:case|branch|block|path|variant)\b/.test(text) ||
    /\b(?:case|branch|block|path|variant)\b.{0,50}\bdefault\b/.test(text);
  if (claimsDefaultLiquidCase) {
    return !supportsDefaultLiquidCaseBranchClaim(
      claim, evidenceItems, sourceSegments
    );
  }

  const claimsMobileDrawerMenuCapture =
    /\bdrawer_menu\b/.test(text) && /\bmobile\b/.test(text);
  if (claimsMobileDrawerMenuCapture) {
    return !supportsHeaderDrawerMenuCaptureClaim(claim, evidenceItems);
  }

  const claimsMobileVariant =
    /\bmobile\b/.test(text) &&
    /\b(?:branch|variant|mode|path)\b/.test(text);
  if (claimsMobileVariant) {
    if (supportsMobileCaseLabelClaim(claim, evidenceItems)) return false;
    return !supportsMobileVariantRenderClaim(claim, evidenceItems, sourceSegments);
  }

  const claimsUnitPriceCondition =
    /\bshow_unit_price\b/.test(text) &&
    /\bunit_price\b/.test(text) &&
    /\b(?:when|if|present|truthy)\b/.test(text);
  if (claimsUnitPriceCondition) return !contextProvesUnitPriceRenderBranch(contexts);

  const claimsFalseCurrencyFormattingBranch =
    /\buse_currency\b/.test(text) &&
    /\bfalse\b/.test(text) &&
    /\b(?:money|format(?:s|ting)?|else|branch)\b/.test(text);

  if (claimsFalseCurrencyFormattingBranch) {
    return !supportsFalseCurrencyBranchClaim(
      claim, evidenceItems, contexts, sourceSegments
    );
  }

  if (
    /\buse_currency\b/.test(text) &&
    /\btrue\b/.test(text) &&
    /\bmoney_with_currency\b/.test(text)
  ) {
    return !supportsTrueCurrencyBranchClaim(
      claim, evidenceItems, contexts, sourceSegments
    );
  }

  const claimsConditionalBehavior =
    /\b(?:when|if|unless|otherwise|truthy|falsy)\b/.test(text) ||
    /\b(?:default|desktop|mobile|tablet)\b.{0,35}\b(?:branch|variant|mode|path)\b/.test(text) ||
    /\b(?:branch|fallback)\b/.test(text) ||
    /\bis\s+(?:true|false)\b/.test(text);
  if (!claimsConditionalBehavior) return false;

  const contextText = contexts.map((context) => context.text).join("\n").toLowerCase();
  const hasControlFlow =
    /\{%\s*(?:if|elsif|unless|case|when|else)\b/.test(evidence) ||
    /\b(?:if|elsif|unless|else)\b/.test(evidence) ||
    /\{%\s*(?:if|elsif|unless|case|when|else)\b/.test(contextText);
  if (!hasControlFlow) return true;
  if (/\b(?:otherwise|else)\b/.test(text) && !/\belse\b/.test(contextText + evidence)) return true;
  if (/\b(?:default|fallback)\s+(?:branch|path|case)\b/.test(text) && !/\belse\b/.test(contextText + evidence)) return true;
  return false;
}

// --------------------------------------------------
// VERIFY ANSWER CLAIMS
// --------------------------------------------------

async function verifyAnswerClaim(
  claim: AnswerClaim,
  availableThemeFiles: Set<string>,
  filesIncludedInContext: Set<string>,
  includedSourceContent: Map<string, string>,
  verifiedDependencies: VerifiedDependency[] = [],
  explicitFileAllowlist?: Set<string> | null
): Promise<VerifiedAnswerClaim> {
  const file = normalizePath(claim.file || "");
  const knownThemePaths = new Set<string>([
    ...availableThemeFiles,
    ...verifiedDependencies.flatMap((dependency) => [dependency.from, dependency.to]),
  ]);
  const claimSubjectMatchesSourceFile = claimSourceSubjectMatchesFile(
    claim.claim || "",
    file,
    knownThemePaths
  );
  const evidenceItems = (Array.isArray(claim.evidence)
    ? claim.evidence
    : [claim.evidence || ""])
    .map((item) => item.trim())
    .filter(Boolean);
  const combinedEvidence = evidenceItems.join("\n");

  let verified = false;
  let verificationReason = "";
  let verificationMode: VerifiedAnswerClaim["verificationMode"] = "unverified";

  if (
    !claim.claim?.trim() ||
    !file ||
    evidenceItems.length === 0
  ) {
    verificationReason =
      "Missing claim, file path, or evidence.";
  } else if (explicitFileAllowlist && !isPathAllowedByScope(file, explicitFileAllowlist)) {
    verificationReason =
      "Claim source file is outside the user's explicit file-only scope.";
  } else if (!availableThemeFiles.has(file)) {
    verificationReason =
      "File does not exist in the Horizon theme.";
  } else if (!filesIncludedInContext.has(file)) {
    verificationReason =
      "File was not included in Gemini context.";
  } else {
    try {
      const actualSource = await readThemeFile(file);
      const sentSource = includedSourceContent.get(file);

      if (!sentSource) {
        verificationReason =
          "Exact source content sent to Gemini is unavailable.";
      } else {
        // ------------------------------------------
        // SPLIT NON-CONTIGUOUS SOURCE RANGES
        // ------------------------------------------

        const sourceSegments = sentSource.split(
          "\n\n[UNRETRIEVED SOURCE RANGE GAP]\n\n"
        );

        // Gemini occasionally rewrites plain statements inside a Shopify
        // {% liquid %} block as invented {% if %}/{% elsif %} tags. For the
        // two known sticky-setting assignment claims only, recover the exact
        // contiguous branch from source that was actually sent to Gemini.
        // This never reads unseen source into evidence and never relaxes the
        // exact-source matcher.
        const recoveredStickyEvidence = recoverExactStickyLiquidBranchEvidence(
          claim.claim,
          sourceSegments
        );
        if (recoveredStickyEvidence) {
          const currentEvidenceMatches = evidenceItems.some((item) =>
            evidenceMatchesSource(item, sourceSegments.join("\n"))
          );
          if (!currentEvidenceMatches) {
            evidenceItems.splice(0, evidenceItems.length, recoveredStickyEvidence);
            console.log("RECOVERED EXACT STICKY LIQUID EVIDENCE:", {
              claimId: claim.id,
              file,
              evidence: recoveredStickyEvidence,
            });
          }
        }

        // ------------------------------------------
        // DIRECT SOURCE EVIDENCE
        // ------------------------------------------

        // Every supplied excerpt must independently exist in one contiguous
        // range sent to Gemini and in the actual file. This permits a claim to
        // cite several locations without allowing invented/stitched evidence.
        const existsInSentContext = evidenceItems.every((evidenceItem) =>
          sourceSegments.some((segment) =>
            evidenceMatchesSource(evidenceItem, segment)
          )
        );

        const existsInActualSource = evidenceItems.every((evidenceItem) =>
          evidenceMatchesSource(evidenceItem, actualSource)
        );

        if (!existsInSentContext || !existsInActualSource) {
          console.warn("SOURCE EVIDENCE MATCH DIAGNOSTIC:", {
            claimId: claim.id ?? null,
            file,
            evidenceMatches: getEvidenceMatchDiagnostics(
              evidenceItems,
              sourceSegments,
              actualSource
            ),
          });
        }

        const isFalseCurrencyBranchClaim =
          /\buse_currency\b/i.test(claim.claim) && /\bfalse\b/i.test(claim.claim);

        // The previous audit showed a currency claim whose combined quote did
        // not match the actual file. Print per-quote match states and short
        // source windows so the next diagnosis is based on the theme file,
        // not on an assumed Liquid structure. This is diagnostic only; failed
        // quotes remain unverified.
        if (isFalseCurrencyBranchClaim && (!existsInSentContext || !existsInActualSource)) {
          console.warn("CURRENCY FALSE-BRANCH EVIDENCE DIAGNOSTIC:", {
            claimId: claim.id ?? null,
            file,
            existsInSentContext,
            existsInActualSource,
            evidenceMatches: getEvidenceMatchDiagnostics(
              evidenceItems,
              sourceSegments,
              actualSource
            ),
            actualSourceWindows: getCurrencySourceDiagnosticWindows(actualSource),
          });
        }

        // Check the claim against the union of its verified excerpts. Each
        // meaningful concept must still be represented; a single keyword is
        // not enough to verify a compound behavior claim.
        const evidenceContexts = getEvidenceContextWindows(
          evidenceItems,
          sourceSegments
        );
        const verifiedAtomicPattern =
          supportsTransparentHeaderTemplateRuleClaim(claim.claim, evidenceItems) ||
          supportsHeaderDrawerMenuCaptureClaim(claim.claim, evidenceItems) ||
          supportsTransparentNotStickyTransitionClaim(claim.claim, evidenceItems) ||
          supportsNavigationBarBranchGuardClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsVolumePricingDetectionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsProductPageQuantityBreakBranchClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsJsonSettingDefinitionClaim(claim.claim, evidenceItems) ||
          supportsLiquidSectionSchemaSettingDefinitionClaim(claim.claim, evidenceItems, file) ||
          supportsStickyCssRuleClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyCssContentClassDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderObserverClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderScrollStateClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderSettingEvaluationClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyConnectedCallbackAttributeReadClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyConnectedCallbackStickyObserverInitClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyObserverThresholdClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyAttributeBooleanObserverCallClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyObserverBooleanCallDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyScrollContainerListenerClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyScrollListenerDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyRafDescriptionClaim(claim.claim, evidenceItems, sourceSegments, actualSource) ||
          supportsStickyDesktopCssDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyIntersectionObserverDatasetUpdateClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderLiquidSettingAssignmentClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderAttributeOutputClaim(claim.claim, evidenceItems) ||
          supportsStickyIntersectionObserverFieldClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyIntersectionObserverBehaviorClaim(claim.claim, evidenceItems) ||
          supportsStickyScrollRafHandlerClaim(claim.claim, evidenceItems) ||
          supportsStickyConnectedCallbackSetupClaim(claim.claim, evidenceItems) ||
          supportsStickyConnectedCallbackBreakpointListenerClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyBreakpointAttributeReadClaim(claim.claim, evidenceItems) ||
          supportsStickyBreakpointObserverResetClaim(claim.claim, evidenceItems, sourceSegments, actualSource) ||
          supportsStickyObserverStateOrOffscreenBranchClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyObserverThemeColorClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyIntersectionStateAssignmentClaim(claim.claim, evidenceItems) ||
          supportsStickyOffscreenAssignmentClaim(claim.claim, evidenceItems, sourceSegments, actualSource) ||
          supportsStickyBreakpointRebindClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyDisconnectedCleanupClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyScrollDirectionAssignmentClaim(claim.claim, evidenceItems) ||
          supportsUseCurrencySettingAssignmentClaim(claim.claim, evidenceItems) ||
          supportsComparePriceFlagClaim(claim.claim, evidenceItems, sourceSegments);
        const claimIsAtomic = isAtomicAnswerClaim(claim.claim) || verifiedAtomicPattern;
        const hasUnsupportedControlFlow = claimAddsUnsupportedControlFlow(
          claim.claim,
          evidenceItems,
          evidenceContexts,
          sourceSegments
        );
        const claimTextLower = claim.claim.toLowerCase();
        const requiresSpecializedEvidence =
          (/\bdrawer_menu\b/.test(claimTextLower) && /\bmobile\b/.test(claimTextLower)) ||
          /\bvolume\s+pricing\b/.test(claimTextLower) ||
          /\bhas_volume_pricing\b/.test(claimTextLower) ||
          (/\bnavigation_bar\b/.test(claimTextLower) && /\b(?:case|branch)\b/.test(claimTextLower)) ||
          (/\btransparent\b/.test(claimTextLower) && /enable_transparent_header_(?:home|product|collection)/.test(claimTextLower)) ||
          (/\btransparent\b/.test(claimTextLower) && /text[\s_-]*color|transparent_text_color_setting/.test(claimTextLower)) ||
          (/\btransparent\b/.test(claimTextLower) && /not[- ]sticky/.test(claimTextLower)) ||
          (/\bshow_unit_price\b/.test(claimTextLower) && /\bunit_price\b/.test(claimTextLower)) ||
          (/\b[a-z0-9_-]+\.js\b/.test(claimTextLower) && /\bmodule\b/.test(claimTextLower) && /\bscript\b/.test(claimTextLower)) ||
          (/enable_sticky_header/i.test(claimTextLower) && /\b(?:schema|setting|select|options?)\b/.test(claimTextLower)) ||
          (/\.sticky-content\b/i.test(claimTextLower) && /\b(?:position|sticky|top|offset|z-index)\b/.test(claimTextLower)) ||
          (/#observestickyposition/i.test(claimTextLower) && /\bintersectionobserver\b/.test(claimTextLower)) ||
          (/#updatescrollstate/i.test(claimTextLower) && /\b(?:sticky|state|scroll|direction|dataset)\b/.test(claimTextLower)) ||
          (/\buse_currency\b/.test(claimTextLower) &&
            /\b(?:true|false|else)\b/.test(claimTextLower) &&
            /\b(?:money|format(?:s|ting)?|branch|else)\b/.test(claimTextLower)) ||
          (/\belse\b/.test(claimTextLower) && /\bmoney\b/.test(claimTextLower) && !/\bmoney_with_currency\b/.test(claimTextLower)) ||
          (/\bmobile\b/.test(claimTextLower) && /\b(?:branch|variant|mode|path|navigation)\b/.test(claimTextLower)) ||
          /\bdefault\b.{0,50}\b(?:case|branch|block|path|variant)\b/.test(claimTextLower) ||
          /\b(?:case|branch|block|path|variant)\b.{0,50}\bdefault\b/.test(claimTextLower);

        const genericTokenOverlap = !requiresSpecializedEvidence &&
          hasMeaningfulTokenOverlap(claim.claim, combinedEvidence);

        const evidenceSupportsClaim =
          genericTokenOverlap ||
          supportsTransparentHeaderTemplateRuleClaim(claim.claim, evidenceItems) ||
          supportsHeaderDrawerMenuCaptureClaim(claim.claim, evidenceItems) ||
          supportsTransparentNotStickyTransitionClaim(claim.claim, evidenceItems) ||
          supportsNavigationBarBranchGuardClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsAssignmentExtractionClaim(claim.claim, evidenceItems) ||
          supportsLiquidCaseStatementClaim(claim.claim, evidenceItems) ||
          supportsLiquidCaseBranchLabelClaim(claim.claim, evidenceItems) ||
          supportsDefaultLiquidCaseBranchClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsComparePriceFlagClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsJsonSettingDefinitionClaim(claim.claim, evidenceItems) ||
          supportsLiquidSectionSchemaSettingDefinitionClaim(claim.claim, evidenceItems, file) ||
          supportsStickyCssRuleClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyCssContentClassDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderObserverClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderScrollStateClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderSettingEvaluationClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyConnectedCallbackAttributeReadClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyConnectedCallbackStickyObserverInitClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyObserverThresholdClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyAttributeBooleanObserverCallClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyObserverBooleanCallDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyScrollContainerListenerClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyScrollListenerDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyRafDescriptionClaim(claim.claim, evidenceItems, sourceSegments, actualSource) ||
          supportsStickyDesktopCssDescriptionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyIntersectionObserverDatasetUpdateClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderLiquidSettingAssignmentClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyHeaderAttributeOutputClaim(claim.claim, evidenceItems) ||
          supportsStickyIntersectionObserverFieldClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyIntersectionObserverBehaviorClaim(claim.claim, evidenceItems) ||
          supportsStickyScrollRafHandlerClaim(claim.claim, evidenceItems) ||
          supportsStickyConnectedCallbackSetupClaim(claim.claim, evidenceItems) ||
          supportsStickyConnectedCallbackBreakpointListenerClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyBreakpointAttributeReadClaim(claim.claim, evidenceItems) ||
          supportsStickyBreakpointObserverResetClaim(claim.claim, evidenceItems, sourceSegments, actualSource) ||
          supportsStickyObserverStateOrOffscreenBranchClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyObserverThemeColorClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyIntersectionStateAssignmentClaim(claim.claim, evidenceItems) ||
          supportsStickyOffscreenAssignmentClaim(claim.claim, evidenceItems, sourceSegments, actualSource) ||
          supportsStickyBreakpointRebindClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyDisconnectedCleanupClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsStickyScrollDirectionAssignmentClaim(claim.claim, evidenceItems) ||
          supportsUseCurrencySettingAssignmentClaim(claim.claim, evidenceItems) ||
          supportsProductPageQuantityBreakBranchClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsModuleScriptLoadingClaim(claim.claim, evidenceItems) ||
          supportsBlankProductFallbackPriceClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsCurrencyFormattingBranchesClaim(claim.claim, evidenceItems) ||
          supportsTrueCurrencyBranchClaim(
            claim.claim, evidenceItems, evidenceContexts, sourceSegments
          ) ||
          supportsFalseCurrencyBranchClaim(
            claim.claim, evidenceItems, evidenceContexts, sourceSegments
          ) ||
          supportsMoneyElseBranchClaim(
            claim.claim, evidenceItems, evidenceContexts, sourceSegments
          ) ||
          supportsUnitPriceRenderConditionClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsMobileCaseLabelClaim(claim.claim, evidenceItems) ||
          supportsMobileVariantRenderClaim(claim.claim, evidenceItems, sourceSegments) ||
          supportsVolumePricingDetectionClaim(claim.claim, evidenceItems, sourceSegments);

        if (
          claimSubjectMatchesSourceFile &&
          claimIsAtomic &&
          !hasUnsupportedControlFlow &&
          existsInSentContext &&
          existsInActualSource &&
          evidenceSupportsClaim
        ) {
          verified = true;
          verificationMode = "direct_source";

          verificationReason =
            "Direct evidence exists in one contiguous Gemini source range and the actual file, with all meaningful claim concepts represented.";
        } else {
          // ----------------------------------------
          // VERIFIED DEPENDENCY RELATIONSHIP FALLBACK
          // ----------------------------------------

          const possibleTargets =
            extractThemeFileReferences(claim.claim, knownThemePaths);

          const matchingDependency = claimSubjectMatchesSourceFile && !explicitFileAllowlist
            ? possibleTargets
            .map((target) =>
              findVerifiedDependencyRelationship(
                file,
                target,
                claim.claim,
                verifiedDependencies
              )
            )
            .find((dependency): dependency is VerifiedDependency => {
              if (!dependency) return false;

              return isPureDependencyRelationshipStatement(
                claim.claim,
                file,
                dependency.to,
                dependency.relationship
              );
            })
            : undefined;

          if (matchingDependency) {
            verified = true;
            verificationMode = "dependency_relationship";

            verificationReason =
              `Direct evidence was insufficient, but the caller file has a verified dependency relationship: ${matchingDependency.relationship} -> ${normalizePath(matchingDependency.to)}. This verifies only the dependency relationship, not the target file's internal implementation.`;
          } else if (!existsInSentContext) {
            verificationReason =
              `Evidence was not present within a single contiguous source range sent to Gemini (segments=${sourceSegments.length}, sentChars=${sentSource.length}, actualSourceMatch=${existsInActualSource}, claimOverlap=${evidenceSupportsClaim}). No matching verified dependency relationship was found.`;
          } else if (!existsInActualSource) {
            verificationReason =
              "Evidence was not found in the actual source file, and no matching verified dependency relationship was found.";
          } else if (!claimSubjectMatchesSourceFile) {
            verificationReason =
              `The claim attributes a source action to a different file than its verified evidence file (${file}).`;
          } else if (!claimIsAtomic) {
            verificationReason =
              "Claim combines multiple actions or coordinated facts. Split it into separate atomic claims with separate evidence before verification.";
          } else if (hasUnsupportedControlFlow) {
            verificationReason =
              "Claim adds a condition, branch, position/order, default, or variant qualifier that is not demonstrated by the supplied source excerpt.";
          } else {
            verificationReason =
              "Evidence text did not contain all meaningful identifiers or concepts supporting the claim wording, and no matching verified dependency relationship was found.";
          }
        }
      }
    } catch {
      verificationReason =
        "Unable to read the actual source file.";
    }
  }

  return {
    ...claim,
    file,
    verified,
    verificationReason,
    verificationMode,
  };
}
// --------------------------------------------------
// NORMALIZE EVIDENCE
// --------------------------------------------------

function normalizeEvidence(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}


// --------------------------------------------------
// SPLIT ANSWER LINES AT CLAIM-TERMINATED STATEMENTS
// --------------------------------------------------

/**
 * Recognize structural headings without treating ordinary numbered factual
 * statements as headings. In particular, section labels such as
 * "1. Header Navigation Architecture" must not be counted as uncited facts.
 */
function getNormalizedSectionHeading(trimmed: string): string {
  return trimmed
    .replace(/^#{1,6}\s*/, "")
    .replace(/^\d{1,2}[.)]\s+/, "")
    .replace(/^\*\*|\*\*$/g, "")
    .replace(/^`|`$/g, "")
    .replace(/[:：]+$/, "")
    .trim()
    .toLowerCase();
}

function isBackendCoverageHeading(trimmed: string): boolean {
  const title = getNormalizedSectionHeading(trimmed);
  return /\bsource\s+coverage\b/.test(title) ||
    /\bmissing\s+(?:or\s+)?incomplete\s+source\s+coverage\b/.test(title) ||
    /\bverified\s+findings(?:\s+and\s+limitations)?\b/.test(title) ||
    /\bfindings\s+and\s+limitations\b/.test(title);
}

function isVerifiedDependencyMapHeading(trimmed: string): boolean {
  return /\bverified\s+dependency\s+map\b/.test(
    getNormalizedSectionHeading(trimmed)
  );
}

function dependencyMapSectionHasClaimMarkers(lines: string[], headingIndex: number): boolean {
  let sawStatement = false;
  for (let index = headingIndex + 1; index < lines.length; index++) {
    const candidate = lines[index].trim();
    if (isStructuralMarkdownLine(candidate)) break;
    if (!candidate) continue;
    sawStatement = true;
    if (!/\[\[C[0-9]+(?:\s*,\s*C[0-9]+)*\]\]/.test(candidate)) {
      return false;
    }
  }
  return sawStatement;
}

function isStructuralMarkdownLine(trimmed: string): boolean {
  if (
    /^#{1,6}\s/.test(trimmed) ||
    /^[-*_]{3,}$/.test(trimmed) ||
    /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(trimmed)
  ) {
    return true;
  }

  const numberedHeading = trimmed.match(/^\d{1,2}[.)]\s+(.+)$/);
  if (!numberedHeading) return false;

  const title = numberedHeading[1].trim();
  if (
    title.length === 0 ||
    title.length > 90 ||
    /[.!?;:]$/.test(title) ||
    /\[\[|\b(?:checks?|renders?|assigns?|sets?|uses?|loads?|formats?|determines?|detects?|compares?|calculates?|computes?|imports?|includes?)\b/i.test(title)
  ) {
    return false;
  }

  const words = title.match(/[A-Za-z][A-Za-z'-]*/g) ?? [];
  if (words.length < 2 || words.length > 10) return false;

  const headingConnectors = new Set([
    "a", "an", "and", "as", "at", "for", "from", "in", "of", "or", "the", "to", "with",
  ]);

  // Require title-style capitalization for substantive words. Lowercase
  // articles/conjunctions are allowed (for example, "Verified Findings and Limitations").
  return words.every((word) =>
    headingConnectors.has(word.toLowerCase()) || /^[A-Z]/.test(word)
  );
}

function splitAnswerIntoCitationStatements(line: string): string[] {
  const markerEndings = [...line.matchAll(/\[\[[^\]]+\]\][.!?]?(?=\s|$)/g)];

  if (markerEndings.length === 0) return [line];

  const statements: string[] = [];
  let cursor = 0;

  for (const match of markerEndings) {
    const start = match.index ?? cursor;
    const end = start + match[0].length;
    const statement = line.slice(cursor, end).trim();

    if (statement) statements.push(statement);

    cursor = end;
    while (cursor < line.length && /\s/.test(line[cursor])) {
      cursor++;
    }
  }

  const remainder = line.slice(cursor).trim();
  if (remainder) statements.push(remainder);

  return statements.length > 0 ? statements : [line];
}

// --------------------------------------------------
// ANSWER-TO-CLAIM TRACEABILITY ENFORCEMENT
// --------------------------------------------------

type AnswerClaimCoverageResult = {
  answer: string;
  retainedLines: number;
  removedLines: number;
  coverageBlockedLines: number;
  pathMismatchLines: number;
  malformedMarkerLines: number;
  uncitedStatementLines: number;
  suppressedModelSectionLines: number;
  unrelatedClaimLines: number;
  fallbackClaimLinesRetained: number;
  fallbackClaimLinesRejected: number;
  unrelatedClaimDiagnostics: Array<{
    claimId: string;
    claim: string;
    statement: string;
    missingTokens: string[];
    matchedTokens: string[];
  }>;
  pathMismatchDiagnostics: Array<{
    statement: string;
    claimIds: string[];
    referencedPaths: string[];
    unsupportedPaths: string[];
    candidateEdges: Array<{
      from: string;
      to: string;
      relationship: string;
      verified: boolean;
      claimVerificationMode: string;
      pureRelationship: boolean;
      tokenOverlap: boolean;
    }>;
  }>;
  invalidClaimReferences: string[];
  ignoredMetadataClaimIds: string[];
  usedClaimIds: string[];
  verifiedDependencyReferenceFiles: string[];
  usedVerifiedClaimFallback: boolean;
};

type SourceCoverageEntry = {
  file: string;
  totalLength: number;
  includedCharacters: number;
  coveragePercent: number;
  retrievedRanges: RetrievedRange[];
  complete: boolean;
};

function enforceAnswerClaimReferences(
  answer: string,
  claims: VerifiedAnswerClaim[],
  sourceCoverage: SourceCoverageEntry[] = [],
  verifiedDependencies: VerifiedDependency[] = [],
  backendMetadataClaimIds: Set<string> = new Set(),
  allowVerifiedClaimFallback = true
): AnswerClaimCoverageResult {
  const verifiedById = new Map(
    claims
      .filter((claim) => claim.verified && claim.id.trim())
      .map((claim) => [claim.id.trim(), claim])
  );

  const lines = answer.split(/\r?\n/);
  const kept: string[] = [];
  const invalidClaimReferences = new Set<string>();
  const ignoredMetadataIdsUsed = new Set<string>();
  const usedClaimIds = new Set<string>();
  const sourceCoverageByFile = new Map(
    sourceCoverage.map((entry) => [normalizePath(entry.file), entry])
  );
  const knownThemePaths = new Set<string>([
    ...sourceCoverage.map((entry) => entry.file),
    ...claims.map((claim) => claim.file),
    ...verifiedDependencies.flatMap((dependency) => [dependency.from, dependency.to]),
  ]);
  const verifiedDependencyReferenceFiles = new Set<string>();

  let retainedLines = 0;
  let removedLines = 0;
  let coverageBlockedLines = 0;
  let pathMismatchLines = 0;
  let malformedMarkerLines = 0;
  let uncitedStatementLines = 0;
  let suppressedModelSectionLines = 0;
  let suppressingBackendOnlySection = false;
  let unrelatedClaimLines = 0;
  const unrelatedClaimDiagnostics: AnswerClaimCoverageResult["unrelatedClaimDiagnostics"] = [];
  const pathMismatchDiagnostics: AnswerClaimCoverageResult["pathMismatchDiagnostics"] = [];

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
    const originalLine = lines[lineIndex];
    const trimmed = originalLine.trim();

    // Coverage sections are backend-owned. Strip the complete model-generated
    // section, not just its heading, so metadata prose cannot become uncited noise.
    if (isBackendCoverageHeading(trimmed)) {
      suppressingBackendOnlySection = true;
      removedLines++;
      suppressedModelSectionLines++;
      continue;
    }

    // Keep a dependency map only when its bullets carry claim citations. An
    // entirely uncited map is removed as one section instead of counting every
    // raw edge as a separate uncited statement.
    if (isVerifiedDependencyMapHeading(trimmed) &&
        !dependencyMapSectionHasClaimMarkers(lines, lineIndex)) {
      suppressingBackendOnlySection = true;
      removedLines++;
      suppressedModelSectionLines++;
      continue;
    }

    if (suppressingBackendOnlySection) {
      if (isStructuralMarkdownLine(trimmed)) {
        suppressingBackendOnlySection = false;
      } else {
        if (trimmed) {
          removedLines++;
          suppressedModelSectionLines++;
        }
        continue;
      }
    }

    if (!trimmed) {
      kept.push(originalLine);
      continue;
    }

    // Structural Markdown lines are not independent factual assertions.
    // Keep numbered section headings as structure, while numbered sentences
    // still pass through citation verification.
    if (isStructuralMarkdownLine(trimmed)) {
      kept.push(originalLine);
      continue;
    }

    // A paragraph can contain several individually cited sentences. Validate
    // each citation-terminated statement separately so one bad citation does
    // not discard neighboring statements with independently verified claims.
    const statements = splitAnswerIntoCitationStatements(originalLine);
    const keptStatements: string[] = [];
    const bulletPrefix = trimmed.match(/^(?:[-*+]\s+|\d+[.)]\s+)/)?.[0] ?? "";

    for (const statement of statements) {
      const statementText = statement.trim();
      if (!statementText) continue;

      // Do not allow prose to claim complete/full source coverage when the
      // backend's exact-range manifest says a referenced file was partial.
      const assertsCompleteSource =
        /\b(?:complete|full|entire)\s+(?:source(?:\s+code)?|file|contents?)\b|\bfully\s+(?:available|provided|included|loaded|retrieved|analyzed)\b/i.test(statementText);
      const explicitlySaysIncomplete =
        /\b(?:not|never|cannot|can't|isn't|wasn't|weren't)\b.{0,45}\b(?:complete|full|fully|entire)\b/i.test(statementText) ||
        /\b(?:complete|full|entire)\s+(?:source(?:\s+code)?|file|contents?)\b.{0,45}\b(?:not|never|unavailable|missing)\b/i.test(statementText);

      if (assertsCompleteSource && !explicitlySaysIncomplete) {
        const paths = extractThemeFileReferences(statementText, knownThemePaths);
        const unsupportedCompletePaths = paths.filter((file) =>
          sourceCoverageByFile.get(file)?.complete !== true
        );

        if (paths.length === 0 || unsupportedCompletePaths.length > 0) {
          coverageBlockedLines++;
          removedLines++;
          unsupportedCompletePaths.forEach((file) =>
            invalidClaimReferences.add(`INCOMPLETE_SOURCE:${file}`)
          );
          continue;
        }
      }

      // Accept [[C1]][[C2]] and [[C1, C2]]. Ranges such as [[C1 to C5]]
      // remain invalid because they can cite unrelated claims.
      const markerMatches = [...statementText.matchAll(/\[\[([^\]]+)\]\]/g)];
      const ids: string[] = [];
      let hasMalformedMarker = false;

      for (const marker of markerMatches) {
        const markerValues = marker[1]
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean);

        if (markerValues.length === 0) {
          hasMalformedMarker = true;
          invalidClaimReferences.add(marker[1].trim() || "<empty>");
          continue;
        }

        for (const value of markerValues) {
          if (!/^C[0-9]+$/.test(value)) {
            hasMalformedMarker = true;
            invalidClaimReferences.add(value);
            continue;
          }
          ids.push(value);
        }
      }

      const allUniqueIds = [...new Set(ids)];
      const metadataIds = allUniqueIds.filter((id) => backendMetadataClaimIds.has(id));
      metadataIds.forEach((id) => ignoredMetadataIdsUsed.add(id));
      const uniqueIds = allUniqueIds.filter((id) => !backendMetadataClaimIds.has(id));

      if (hasMalformedMarker) {
        malformedMarkerLines++;
        removedLines++;
        continue;
      }

      // Backend-only source coverage claims are not source-code evidence.
      if (uniqueIds.length === 0 && metadataIds.length > 0) {
        coverageBlockedLines++;
        removedLines++;
        continue;
      }

      if (uniqueIds.length === 0) {
        // No marker is an uncited statement, not a malformed marker. Keep
        // diagnostics truthful so the metric identifies syntax errors only.
        if (markerMatches.length === 0) {
          uncitedStatementLines++;
        } else {
          malformedMarkerLines++;
        }
        removedLines++;
        continue;
      }

      const allReferencesVerified = uniqueIds.every((id) => verifiedById.has(id));
      if (!allReferencesVerified) {
        uniqueIds.forEach((id) => {
          if (!verifiedById.has(id)) invalidClaimReferences.add(id);
        });
        removedLines++;
        continue;
      }

      const citedClaims = uniqueIds
        .map((id) => verifiedById.get(id))
        .filter((claim): claim is VerifiedAnswerClaim => Boolean(claim));

      // Dependency-only claims may have no ordinary descriptive tokens after
      // the source/target paths and relationship verb are removed. In that
      // case, require the exact verified edge and a relationship-only sentence.
      const statementFilePaths = extractThemeFileReferences(statementText, knownThemePaths);
      const unrelatedClaimIds = citedClaims
        .filter((claim) => {
          const directTokenAlignment = hasMeaningfulTokenOverlap(
            claim.claim,
            statementText
          );
          const useCurrencyElseAlignment =
            /\buse_currency\b/.test(claim.claim.toLowerCase()) &&
            /\belse\b/.test(claim.claim.toLowerCase()) &&
            /\buse_currency\b/.test(statementText.toLowerCase()) &&
            /\bfalse\b/.test(statementText.toLowerCase()) &&
            getClaimEvidenceTokens(claim.claim)
              .filter((token) => token !== "else")
              .every((token) => getClaimEvidenceTokens(statementText).includes(token));
          const stickyScrollRafAlignment = isStickyScrollRafStatementAligned(
            claim.claim,
            statementText
          );
          const stickyHeaderSettingAlignment = isStickyHeaderSettingStatementAligned(
            claim.claim,
            statementText
          );
          const stickyCssDescriptionAlignment = isStickyCssDescriptionStatementAligned(
            claim.claim,
            statementText
          );
          const stickyObserverConstructionAlignment = isStickyObserverConstructionStatementAligned(
            claim.claim,
            statementText
          );
          const stickyIntersectionStateAlignment = isStickyIntersectionStateStatementAligned(
            claim.claim,
            statementText
          );
          const stickyHeaderAttributeOutputAlignment = isStickyHeaderAttributeOutputStatementAligned(
            claim.claim,
            statementText
          );
          const stickyDesktopCssAlignment = isStickyDesktopCssStatementAligned(
            claim.claim,
            statementText
          );
          const stickyConnectedCallbackInitAlignment = isStickyConnectedCallbackInitStatementAligned(
            claim.claim,
            statementText
          );
          const stickyObserverThresholdAlignment = isStickyObserverThresholdStatementAligned(
            claim.claim,
            statementText
          );
          const stickyOffscreenAssignmentAlignment = isStickyOffscreenAssignmentStatementAligned(
            claim.claim,
            statementText
          );
          if (
            directTokenAlignment ||
            useCurrencyElseAlignment ||
            stickyScrollRafAlignment ||
            stickyHeaderSettingAlignment ||
            stickyCssDescriptionAlignment ||
            stickyObserverConstructionAlignment ||
            stickyIntersectionStateAlignment ||
            stickyHeaderAttributeOutputAlignment ||
            stickyDesktopCssAlignment ||
            stickyConnectedCallbackInitAlignment ||
            isStickyScrollListenerStatementAligned(claim.claim, statementText) ||
            stickyObserverThresholdAlignment ||
            stickyOffscreenAssignmentAlignment
          ) {
            return false;
          }

          const supportedByDependency = statementFilePaths.some((targetFile) => {
            const dependency = findVerifiedDependencyRelationship(
              claim.file,
              targetFile,
              statementText,
              verifiedDependencies
            );

            return Boolean(
              dependency &&
              isPureDependencyRelationshipStatement(
                statementText,
                claim.file,
                dependency.to,
                dependency.relationship
              )
            );
          });

          return !supportedByDependency;
        })
        .map((claim) => claim.id);

      if (unrelatedClaimIds.length > 0) {
        const statementTokens = new Set(getClaimEvidenceTokens(statementText));
        for (const claimId of unrelatedClaimIds) {
          const mismatchedClaim = citedClaims.find((claim) => claim.id === claimId);
          if (!mismatchedClaim) continue;

          const claimTokens = getClaimEvidenceTokens(mismatchedClaim.claim);
          const matchedTokens = claimTokens.filter((token) => statementTokens.has(token));
          const missingTokens = claimTokens.filter((token) => !statementTokens.has(token));

          if (unrelatedClaimDiagnostics.length < 20) {
            unrelatedClaimDiagnostics.push({
              claimId,
              claim: mismatchedClaim.claim.slice(0, 260),
              statement: statementText.slice(0, 360),
              missingTokens,
              matchedTokens,
            });
          }
        }

        unrelatedClaimLines++;
        removedLines++;
        unrelatedClaimIds.forEach((id) =>
          invalidClaimReferences.add(`UNRELATED_CLAIM:${id}`)
        );
        continue;
      }

      // A target snippet path is allowed only when a verified caller claim
      // and a verified dependency edge support that explicit relationship.
      if (statementFilePaths.length > 0) {
        const unsupportedPaths = statementFilePaths.filter((file) => {
          const directlySupported = citedClaims.some((claim) =>
            normalizePath(claim.file) === file && sourceCoverageByFile.has(file)
          );

          const supportedAsDependencyTarget = citedClaims.some((claim) => {
            const dependency = findVerifiedDependencyRelationship(
              claim.file,
              file,
              statementText,
              verifiedDependencies
            );

            if (!dependency) return false;

            // The verified caller evidence already proves the source-side
            // edge. Pure relationship-only statements are accepted directly;
            // richer statements additionally need a verified direct-source
            // claim, the caller path, and claim-to-sentence relevance.
            return canAuthorizeDependencyTargetPath(
              statementText,
              statementFilePaths,
              claim,
              dependency
            );
          });

          return !directlySupported && !supportedAsDependencyTarget;
        });

        if (unsupportedPaths.length > 0) {
          pathMismatchLines++;
          removedLines++;
          if (pathMismatchDiagnostics.length < 20) {
            const candidateEdges = citedClaims.flatMap((claim) =>
              verifiedDependencies
                .filter((dependency) =>
                  normalizePath(dependency.from) === normalizePath(claim.file) &&
                  statementFilePaths.includes(normalizePath(dependency.to))
                )
                .map((dependency) => ({
                  from: normalizePath(dependency.from),
                  to: normalizePath(dependency.to),
                  relationship: dependency.relationship,
                  verified: dependency.verified,
                  claimVerificationMode: claim.verificationMode,
                  pureRelationship: isPureDependencyRelationshipStatement(
                    statementText,
                    claim.file,
                    dependency.to,
                    dependency.relationship
                  ),
                  tokenOverlap: hasMeaningfulTokenOverlap(claim.claim, statementText),
                }))
            );
            pathMismatchDiagnostics.push({
              statement: statementText.slice(0, 500),
              claimIds: uniqueIds,
              referencedPaths: statementFilePaths,
              unsupportedPaths,
              candidateEdges,
            });
          }
          unsupportedPaths.forEach((file) => {
            const existsAsSource = sourceCoverageByFile.has(file);
            invalidClaimReferences.add(
              existsAsSource
                ? `NO_MATCHING_CLAIM_OR_DEPENDENCY:${file}`
                : `SOURCE_NOT_INCLUDED_OR_DEPENDENCY_UNVERIFIED:${file}`
            );
          });
          continue;
        }
      }

      // Keep an explicit backend trace of dependency targets before citation
      // markers are removed from the user-facing answer. The final path audit
      // must not try to reconstruct claim IDs from marker-free prose.
      for (const targetFile of statementFilePaths) {
        const supportedDependency = citedClaims.some((claim) => {
          const dependency = findVerifiedDependencyRelationship(
            claim.file,
            targetFile,
            statementText,
            verifiedDependencies
          );

          if (!dependency) return false;

          return canAuthorizeDependencyTargetPath(
            statementText,
            statementFilePaths,
            claim,
            dependency
          );
        });

        if (supportedDependency) {
          verifiedDependencyReferenceFiles.add(targetFile);
        }
      }

      uniqueIds.forEach((id) => usedClaimIds.add(id));
      keptStatements.push(
        statementText.replace(/\s*\[\[[^\]]+\]\]/g, "").trim()
      );
      retainedLines++;
    }

    if (keptStatements.length > 0) {
      // If earlier sentences in a Markdown list item were removed, retain the
      // list marker on the first surviving sentence.
      if (bulletPrefix && !/^[-*+]\s+|^\d+[.)]\s+/.test(keptStatements[0])) {
        keptStatements[0] = `${bulletPrefix}${keptStatements[0]}`;
      }
      kept.push(keptStatements.join(" "));
    }
  }

  let sanitizedAnswer = kept.join("\n").trim();
  let usedVerifiedClaimFallback = false;
  let fallbackClaimLinesRetained = 0;
  let fallbackClaimLinesRejected = 0;

  // Gemini can bundle several individually verified claims into one sentence
  // with unrelated citations. The sentence is rejected above. Preserve unused
  // verified claims only as exact claim text with its own marker, then run those
  // generated lines through the same traceability/path checks in a second pass.
  // This prevents fallback from becoming an uncited or unvalidated escape hatch.
  if (allowVerifiedClaimFallback) {
    const fallbackClaims = [...verifiedById.values()].filter(
      (claim) => !usedClaimIds.has(claim.id)
    );

    if (fallbackClaims.length > 0) {
      const fallbackHeading = retainedLines === 0
        ? "### Verified source findings"
        : "### Additional verified source findings";
      const fallbackLines = fallbackClaims.map((claim) =>
        `- **${claim.file}** — ${claim.claim} [[${claim.id}]]`
      );
      const fallbackResult = enforceAnswerClaimReferences(
        [fallbackHeading, "", ...fallbackLines].join("\n"),
        claims,
        sourceCoverage,
        verifiedDependencies,
        backendMetadataClaimIds,
        false
      );

      fallbackClaimLinesRetained = fallbackResult.retainedLines;
      fallbackClaimLinesRejected = fallbackResult.removedLines;
      if (fallbackResult.answer.trim()) {
        sanitizedAnswer = [sanitizedAnswer, fallbackResult.answer]
          .filter(Boolean)
          .join("\n\n");
      }

      retainedLines += fallbackResult.retainedLines;
      removedLines += fallbackResult.removedLines;
      coverageBlockedLines += fallbackResult.coverageBlockedLines;
      pathMismatchLines += fallbackResult.pathMismatchLines;
      malformedMarkerLines += fallbackResult.malformedMarkerLines;
      uncitedStatementLines += fallbackResult.uncitedStatementLines;
      suppressedModelSectionLines += fallbackResult.suppressedModelSectionLines;
      unrelatedClaimLines += fallbackResult.unrelatedClaimLines;

      fallbackResult.unrelatedClaimDiagnostics.forEach((diagnostic) => {
        if (unrelatedClaimDiagnostics.length < 20) {
          unrelatedClaimDiagnostics.push(diagnostic);
        }
      });
      fallbackResult.pathMismatchDiagnostics.forEach((diagnostic) => {
        if (pathMismatchDiagnostics.length < 20) {
          pathMismatchDiagnostics.push(diagnostic);
        }
      });
      fallbackResult.invalidClaimReferences.forEach((value) =>
        invalidClaimReferences.add(value)
      );
      fallbackResult.ignoredMetadataClaimIds.forEach((value) =>
        ignoredMetadataIdsUsed.add(value)
      );
      fallbackResult.usedClaimIds.forEach((value) => {
        usedClaimIds.add(value);
        // The original model sentence may have attached this valid claim to an
        // unrelated statement, but the exact fallback sentence has now passed
        // the full validator. Do not report that rejected first attempt as an
        // invalid reference in the final sanitized answer.
        invalidClaimReferences.delete(`UNRELATED_CLAIM:${value}`);
      });
      fallbackResult.verifiedDependencyReferenceFiles.forEach((value) =>
        verifiedDependencyReferenceFiles.add(value)
      );
      usedVerifiedClaimFallback = fallbackClaimLinesRetained > 0;
    }
  }

  return {
    answer: sanitizedAnswer,
    retainedLines,
    removedLines,
    coverageBlockedLines,
    pathMismatchLines,
    malformedMarkerLines,
    uncitedStatementLines,
    suppressedModelSectionLines,
    unrelatedClaimLines,
    fallbackClaimLinesRetained,
    fallbackClaimLinesRejected,
    unrelatedClaimDiagnostics,
    pathMismatchDiagnostics,
    invalidClaimReferences: [...invalidClaimReferences],
    ignoredMetadataClaimIds: [...ignoredMetadataIdsUsed],
    usedClaimIds: [...usedClaimIds],
    verifiedDependencyReferenceFiles: [...verifiedDependencyReferenceFiles],
    usedVerifiedClaimFallback,
  };
}

// --------------------------------------------------
// BUILD TRUSTED SOURCE COVERAGE SUMMARY
// --------------------------------------------------

function buildSourceCoverageMarkdown(
  entries: SourceCoverageEntry[]
): string {
  if (entries.length === 0) return "";

  const completeFiles = entries.filter((entry) => entry.complete);
  const partialFiles = entries.filter((entry) => !entry.complete);

  const formatFiles = (files: SourceCoverageEntry[]) =>
    files.map((entry) => {
      const ranges = entry.retrievedRanges
        .map((range) => `${range.startOffset}-${range.endOffset}`)
        .join(", ");

      return `- \`${entry.file}\`: ${entry.complete
        ? `complete (${entry.totalLength} characters)`
        : `partial (${entry.coveragePercent}% supplied; ranges ${ranges || "none"} of ${entry.totalLength})`
        }`;
    });

  return [
    "",
    "",
    "### Source coverage (calculated from the exact ranges sent to Gemini)",
    ...(completeFiles.length > 0
      ? ["", "**Complete source files:**", ...formatFiles(completeFiles)]
      : []),
    ...(partialFiles.length > 0
      ? ["", "**Partial source files:**", ...formatFiles(partialFiles)]
      : []),
  ].join("\n");
}

// --------------------------------------------------
// MAIN POST API
// --------------------------------------------------

export async function POST(request: NextRequest) {
  try {
    // -------------------------------------
    // STEP 0: Validate request
    // -------------------------------------

    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        {
          success: false,
          error: "GEMINI_API_KEY is not configured.",
        },
        { status: 500 }
      );
    }

    const body = await request.json();

    const prompt = body.prompt;

    const history = Array.isArray(body.history)
      ? body.history
      : [];

    if (
      !prompt ||
      typeof prompt !== "string" ||
      !prompt.trim()
    ) {
      return NextResponse.json(
        {
          success: false,
          error: "Prompt is required",
        },
        { status: 400 }
      );
    }

    // -------------------------------------
    // STEP 1: Resolve explicit source scope before broad search
    // -------------------------------------

    const themeFileManifest = await getThemeFiles();
    const explicitFileRequest = resolveExplicitFileRequest(prompt, themeFileManifest);
    const explicitOnly = explicitFileRequest.enabled;
    const explicitFileAllowlist = explicitOnly
      ? new Set(explicitFileRequest.canonicalPaths)
      : null;

    console.log("\n========== EXPLICIT FILE REQUEST RESOLUTION ==========");
    console.log({
      explicitOnly,
      detectedPaths: explicitFileRequest.detectedPaths,
      canonicalPaths: explicitFileRequest.canonicalPaths,
      invalidPaths: explicitFileRequest.invalidPaths,
    });

    if (explicitOnly && explicitFileRequest.invalidPaths.length > 0) {
      return NextResponse.json({
        success: false,
        error: "One or more explicitly requested theme files are invalid or missing. No fallback search was performed.",
        explicitOnly: true,
        detectedPaths: explicitFileRequest.detectedPaths,
        resolvedPaths: explicitFileRequest.canonicalPaths,
        invalidPaths: explicitFileRequest.invalidPaths,
        filesAnalyzed: [],
        filesIncludedInContext: [],
        filesSkippedFromContext: [],
      }, { status: 400 });
    }

    if (explicitOnly && explicitFileRequest.canonicalPaths.length > MAX_FILES) {
      return NextResponse.json({
        success: false,
        error: `The explicit request contains ${explicitFileRequest.canonicalPaths.length} files, exceeding the current limit of ${MAX_FILES}. No files were loaded.`,
        explicitOnly: true,
        detectedPaths: explicitFileRequest.detectedPaths,
        resolvedPaths: explicitFileRequest.canonicalPaths,
        filesAnalyzed: [],
        filesIncludedInContext: [],
        filesSkippedFromContext: [],
      }, { status: 400 });
    }

    if (explicitOnly && explicitFileRequest.canonicalPaths.length === 0) {
      return NextResponse.json({
        success: false,
        error: "An explicit file-analysis request was detected, but no valid Horizon theme file path could be resolved. No fallback search was performed.",
        explicitOnly: true,
        detectedPaths: explicitFileRequest.detectedPaths,
        filesAnalyzed: [],
        filesIncludedInContext: [],
        filesSkippedFromContext: [],
      }, { status: 400 });
    }

    // Explicit-only mode does not call broad semantic search at all.
    const searchResults = explicitOnly ? [] : await searchTheme(prompt);

    if (!explicitOnly && searchResults.length === 0) {
      return NextResponse.json({
        success: true,
        message:
          "I could not find relevant source files in the Horizon theme for this request.",
        claims: [],
        evidenceSummary: {
          total: 0,
          verified: 0,
          unverified: 0,
        },
        filesAnalyzed: [],
        filesIncludedInContext: [],
        filesSkippedFromContext: [],
        dependencyGraph: [],
        relevantDependencies: [],
        verificationSummary: {
          total: 0,
          verified: 0,
          unverified: 0,
          relevant: 0,
        },
      });
    }

    const selectedFiles = new Map<string, ThemeFile>();

    const isProductCardIntent =
      isProductCardArchitectureIntent(prompt);
    const isStickyHeaderIntent =
      isStickyHeaderArchitectureIntent(prompt);
    const explicitlyAsksForHeaderNavigation =
      /\b(menu|navigation|nav|drawer|mega[\s-]?menu)\b/i.test(prompt);

    // -------------------------------------
    // SELECT ARCHITECTURAL ROOT FILES
    // -------------------------------------

    const rootFilePriorities = explicitOnly
      ? [...explicitFileRequest.canonicalPaths]
      : getArchitectureRootFiles(prompt);

    // Map search results for quick lookup
    const searchResultMap = new Map(
      searchResults.map((file) => [
        file.file,
        file,
      ])
    );

    const initialFiles: ThemeFile[] = [];
    let directlyReferencedStickyAssets = new Set<string>();

    // -------------------------------------
    // FIRST: ADD ARCHITECTURAL ROOT FILES
    // -------------------------------------

    for (const filePath of rootFilePriorities) {
      let file = searchResultMap.get(filePath);

      // Architecture entry points are loaded by exact path. They must not
      // disappear merely because broad keyword matching ranked other files
      // (for example product-card files) above them.
      if (!file) {
        try {
          const content = await readThemeFile(filePath);
          file = {
            file: filePath,
            content,
            score: 1000,
          };
        } catch {
          console.log(
            "REQUESTED ARCHITECTURE ROOT NOT FOUND:",
            filePath
          );
          continue;
        }
      }

      const rootFile = {
        ...file,
        score: Math.max(file.score, 1000),
      };

      initialFiles.push(rootFile);
      selectedFiles.set(rootFile.file, rootFile);
    }

    if (isStickyHeaderIntent && !explicitOnly) {
      directlyReferencedStickyAssets = getDirectlyReferencedThemeAssetFiles(
        selectedFiles.get("sections/header.liquid")?.content ?? "",
        searchResults
      );
      console.log("DIRECTLY REFERENCED STICKY-HEADER ASSETS:", [
        ...directlyReferencedStickyAssets,
      ]);
    }

    // -------------------------------------
    // SECOND: ADD HIGHLY RELEVANT FILES
    // -------------------------------------

    if (!explicitOnly) {
      const maxScore = searchResults[0]?.score ?? 0;

      const relevanceThreshold = maxScore * 0.35;

      const MAX_INITIAL_FILES = 7;

      for (const result of searchResults) {
      if (
        initialFiles.length >= MAX_INITIAL_FILES
      ) {
        break;
      }

      if (
        selectedFiles.has(result.file)
      ) {
        continue;
      }

      if (
        result.score < relevanceThreshold
      ) {
        continue;
      }

      // Avoid unrelated header and filter files for product-card analysis.
      if (
        isProductCardIntent &&
        (
          result.file.includes("header") ||
          result.file.includes("filters")
        )
      ) {
        continue;
      }

      // Sticky-header analysis should not spend its initial context on mega-menu
      // and drawer internals unless the user explicitly asks about navigation.
      if (
        isStickyHeaderIntent &&
        !explicitlyAsksForHeaderNavigation &&
        /(?:header-menu|header-drawer|mega-menu|overflow-list)/i.test(result.file)
      ) {
        continue;
      }

      if (isStickyHeaderIntent) {
        const normalizedResultPath = normalizePath(result.file);
        const stickyRelevant =
          normalizedResultPath === "sections/header.liquid" ||
          normalizedResultPath === "assets/base.css" ||
          directlyReferencedStickyAssets.has(normalizedResultPath);
        if (!stickyRelevant) continue;
      }

      initialFiles.push(result);

      selectedFiles.set(
        result.file,
        result
      );
      }
    }

    console.log(
      "\n========== SMART ROOT FILES =========="
    );

    console.table(
      initialFiles.map(({ file, score }) => ({
        file,
        score,
      }))
    );

    // -------------------------------------
    // STEP 2: Extract dependencies
    // -------------------------------------

    let dependencies: Dependency[] = [];

    if (!explicitOnly) {
      for (const file of selectedFiles.values()) {
        const extracted = extractDependencies(
          file.file,
          file.content,
          prompt
        );

        dependencies.push(...extracted);
      }

      dependencies = uniqueDependencies(dependencies);
    } else {
      console.log("EXPLICIT-ONLY: dependency extraction and dependency graph expansion disabled.");
    }

    // -------------------------------------
    // STEP 3: Recursive dependency loading
    // -------------------------------------

    const dependencyQueue: {
      to: string;
      score: number;
    }[] = [];

    const queuedFiles = new Set<string>();

    function isThemeFileDependency(
      to: string
    ): boolean {
      return (
        to.startsWith("snippets/") ||
        to.startsWith("blocks/") ||
        to.startsWith("sections/")
      );
    }

    for (const dependency of (explicitOnly ? [] : dependencies)) {
      if (
        isThemeFileDependency(dependency.to) &&
        dependency.relevanceScore >= MIN_DEPENDENCY_SCORE &&
        !queuedFiles.has(dependency.to)
      ) {
        dependencyQueue.push({
          to: dependency.to,
          score: dependency.relevanceScore,
        });

        queuedFiles.add(dependency.to);
      }
    }

    console.log(
      "\n========== RECURSIVE DEPENDENCY LOADING =========="
    );

    let loadedDependencyCount = 0;

    while (!explicitOnly && dependencyQueue.length > 0) {
      dependencyQueue.sort(
        (a, b) => b.score - a.score
      );

      const nextDependency = dependencyQueue.shift()!;

      const target = nextDependency.to;

      if (selectedFiles.has(target)) {
        continue;
      }

      // Reserve one slot for settings_schema.json.

      const reservedFileCount = 1;

      if (
        selectedFiles.size >=
        MAX_FILES - reservedFileCount
      ) {
        console.log(
          "MAX FILE LIMIT REACHED:",
          target
        );

        continue;
      }

      try {
        const content = await readThemeFile(target);

        const themeFile: ThemeFile = {
          file: target,
          content,
          score: 0,
        };

        selectedFiles.set(target, themeFile);

        loadedDependencyCount++;

        console.log(
          "DEPENDENCY LOADED:",
          target
        );

        const extractedDependencies =
          extractDependencies(
            target,
            content,
            prompt
          );

        dependencies.push(...extractedDependencies);

        for (const dependency of extractedDependencies) {
          if (
            isThemeFileDependency(dependency.to) &&
            dependency.relevanceScore >= MIN_DEPENDENCY_SCORE &&
            !queuedFiles.has(dependency.to)
          ) {
            dependencyQueue.push({
              to: dependency.to,
              score: dependency.relevanceScore,
            });

            queuedFiles.add(dependency.to);
          }
        }
      } catch {
        console.log(
          "DEPENDENCY FILE NOT FOUND:",
          target
        );
      }
    }

    dependencies = uniqueDependencies(dependencies);

    console.log(
      "Total recursively loaded files:",
      loadedDependencyCount
    );

    console.log(
      "Total unique dependencies:",
      dependencies.length
    );

    // -------------------------------------
    // STEP 4: Load settings schema
    // -------------------------------------

    let schemaContent = "";

    if (!explicitOnly) {
      try {
        schemaContent = await readThemeFile(
          "config/settings_schema.json"
        );

        if (
          !selectedFiles.has(
            "config/settings_schema.json"
          )
        ) {
          selectedFiles.set(
            "config/settings_schema.json",
            {
              file: "config/settings_schema.json",
              content: schemaContent,
              score: 0,
            }
          );
        }
      } catch {
        console.warn(
          "Could not read settings_schema.json"
        );
      }
    } else {
      console.log("EXPLICIT-ONLY: automatic config/settings_schema.json loading disabled.");
    }

    // Strict maximum file limit.

    if (selectedFiles.size > MAX_FILES) {
      const removableFile = Array.from(
        selectedFiles.values()
      )
        .filter(
          (file) =>
            file.file !== "config/settings_schema.json" &&
            !initialFiles.some(
              (root) => root.file === file.file
            )
        )
        .sort((a, b) => a.score - b.score)[0];

      if (removableFile) {
        selectedFiles.delete(removableFile.file);
      }
    }

    // -------------------------------------
    // STEP 5: Add setting definitions
    // -------------------------------------

    if (schemaContent) {
      const definitions =
        buildSettingDefinitionDependencies(
          dependencies,
          schemaContent,
          prompt
        );

      dependencies = uniqueDependencies([
        ...dependencies,
        ...definitions,
      ]);
    }

    // -------------------------------------
    // STEP 6: Verify dependencies
    // -------------------------------------

    const verifiedDependencies: VerifiedDependency[] = [];

    for (const dependency of dependencies) {
      const verification = await verifyDependency(
        dependency,
        schemaContent,
        dependencies
      );

      verifiedDependencies.push(verification);
    }

    const verifiedCount =
      verifiedDependencies.filter(
        (dependency) => dependency.verified
      ).length;

    // -------------------------------------
    // STEP 7: Filter relevant dependencies
    // -------------------------------------

    const relevantDependencies =
      filterRelevantDependencies(
        verifiedDependencies,
        initialFiles
      );

    console.log(
      "\n========== RELEVANT DEPENDENCIES =========="
    );

    console.table(
      relevantDependencies.map((dependency) => ({
        from: dependency.from,
        to: dependency.to,
        relationship: dependency.relationship,
        score: dependency.relevanceScore,
        category: classifyDependency(dependency),
      }))
    );

    console.log(
      "Total dependencies:",
      verifiedDependencies.length
    );

    console.log(
      "Verified:",
      verifiedCount
    );

    console.log(
      "Relevant:",
      relevantDependencies.length
    );

    // -------------------------------------
    // STEP 8: Filter relevant theme files
    // -------------------------------------

    const allLoadedFiles = Array.from(
      selectedFiles.values()
    );

    const relevantFiles = filterRelevantThemeFiles(
      allLoadedFiles,
      initialFiles,
      relevantDependencies,
      explicitFileAllowlist
    );

    const filesAnalyzed = Array.from(
      selectedFiles.keys()
    );

    // -------------------------------------
    // STEP 9: Build initial context
    // -------------------------------------

    const {
      context: themeContext,
      filesIncludedInContext,
      filesSkippedFromContext,
      includedSourceContent,
      truncatedFiles,
      filePriorities: contextFilePriorities,
    } = buildThemeContext(
      relevantFiles,
      relevantDependencies,
      INITIAL_CONTEXT_BUDGET,
      rootFilePriorities,
      prompt
    );


    // -------------------------------------
    // STEP 9.5: Shared retrieval state
    // -------------------------------------

    const onDemandBudgetCap = Math.max(
      0,
      Math.min(
        ON_DEMAND_CONTEXT_BUDGET,
        MAX_CONTEXT_SIZE - themeContext.length
      )
    );

    const retrievalState = createRetrievalState(
      themeContext.length,
      onDemandBudgetCap,
      MAX_ON_DEMAND_FILES,
      explicitFileAllowlist
    );

    // Seed fileStates for every file already in the initial context.
    for (const filePath of filesIncludedInContext) {
      const includedContent = includedSourceContent.get(filePath);
      if (!includedContent) continue;

      const fullFile =
        selectedFiles.get(filePath) ??
        relevantFiles.find((f) => f.file === filePath);
      if (!fullFile) continue;

      const totalLength = fullFile.content.length;
      const end = Math.min(includedContent.length, totalLength);

      retrievalState.fileStates.set(filePath, {
        file: filePath,
        totalLength,
        retrievedRanges: end > 0 ? [{ startOffset: 0, endOffset: end }] : [],
        nextOffset: end,
        hasMore: end < totalLength,
        isComplete: end >= totalLength,
      });

      retrievalState.fullContents.set(filePath, fullFile.content);
    }

    // Cache full contents and mark deps extracted for every loaded file.
    // Files loaded in STEP 2/3 already had deps extracted from full content.
    for (const file of selectedFiles.values()) {
      if (!retrievalState.fullContents.has(file.file)) {
        retrievalState.fullContents.set(file.file, file.content);
      }
      retrievalState.processedDependencyRanges.set(file.file, [
        { startOffset: 0, endOffset: file.content.length },
      ]);
    }

    console.log("\n========== SHARED RETRIEVAL STATE INITIALIZED ==========");
    console.log({
      initialUsedBudget: retrievalState.usedBudget,
      onDemandBudget: retrievalState.remainingBudget,
      totalBudget: retrievalState.totalBudget,
      seededFileStates: retrievalState.fileStates.size,
      seededFullContents: retrievalState.fullContents.size,
    });

    // -------------------------------------
    // STEP 10: Debug report
    // -------------------------------------

    console.log(
      "\n========== SEARCH RESULTS =========="
    );

    console.table(
      searchResults.slice(0, 10).map(
        ({ file, score }) => ({
          file,
          score,
        })
      )
    );

    console.log(
      "\n========== ALL LOADED FILES =========="
    );

    console.table(filesAnalyzed);

    console.log(
      "\n========== FILES SENT TO GEMINI =========="
    );

    console.table(filesIncludedInContext);

    console.log(
      "\n========== FILES SKIPPED FROM CONTEXT =========="
    );

    console.table(filesSkippedFromContext);

    console.log(
      "\n========== CONTEXT BUDGET =========="
    );

    console.log({
      maxContextBudget: MAX_CONTEXT_SIZE,
      initialContextBudget: INITIAL_CONTEXT_BUDGET,
      onDemandReservedBudget: ON_DEMAND_CONTEXT_BUDGET,
      initialContextUsed: themeContext.length,
      initialContextRemaining:
        INITIAL_CONTEXT_BUDGET - themeContext.length,
    });

    console.log(
      "\n========== SUMMARY =========="
    );

    console.table({
      totalSearchResults: searchResults.length,
      initialFilesSelected: initialFiles.length,
      totalFilesLoaded: filesAnalyzed.length,
      relevantFiles: relevantFiles.length,
      filesSentToGemini: filesIncludedInContext.length,
      filesSkipped: filesSkippedFromContext.length,
      totalDependencies: verifiedDependencies.length,
      verified: verifiedCount,
      unverified:
        verifiedDependencies.length - verifiedCount,
      relevantDependencies: relevantDependencies.length,
    });

    // -------------------------------------
    // STEP 11: Prepare Gemini history
    // -------------------------------------

    const validHistory = (explicitOnly ? [] : history)
      .filter(
        (message: any) =>
          message &&
          typeof message.content === "string" &&
          ["user", "assistant", "model"].includes(
            message.role
          )
      )
      .slice(-10)
      .map((message: any) => ({
        role:
          message.role === "assistant"
            ? "model"
            : message.role,
        parts: [
          {
            text: message.content,
          },
        ],
      }));

    const availableThemeFiles = new Set(
      themeFileManifest
    );



    const contents: any[] = [
      ...validHistory,
      {
        role: "user",
        parts: [
          {
            text: `
USER REQUEST:
${prompt}

HORIZON THEME SOURCE CODE:
${themeContext}

FILES ANALYZED BY BACKEND:
${JSON.stringify(filesAnalyzed, null, 2)}

FILES INCLUDED IN GEMINI CONTEXT:
${JSON.stringify(filesIncludedInContext, null, 2)}

FILES SKIPPED FROM CONTEXT:
${JSON.stringify(filesSkippedFromContext, null, 2)}

RELEVANT VERIFIED DEPENDENCY GRAPH:
${JSON.stringify(relevantDependencies, null, 2)}

VERIFICATION SUMMARY:
${verifiedCount} of ${verifiedDependencies.length} dependencies verified.

AVAILABLE THEME FILE PATHS:
${JSON.stringify(explicitOnly ? [...(explicitFileAllowlist ?? [])] : [...availableThemeFiles], null, 2)}

IMPORTANT:
Only use files included in GEMINI CONTEXT as source-code evidence.
Do not assume files in FILES ANALYZED BY BACKEND were sent to Gemini.
${explicitOnly ? `
EXPLICIT-ONLY FILE SCOPE (MANDATORY):
The user explicitly selected these source files and no others:
${JSON.stringify([...(explicitFileAllowlist ?? [])], null, 2)}
Do not claim to have inspected, read, or verified any other theme file.
Do not make claims whose claim.file is outside this exact allowlist.
Do not infer the internal behavior of referenced dependencies that were not selected.
Use only the selected file source text supplied above as evidence.
` : ""}
`,
          },
        ],
      },
    ];


    // -------------------------------------
    // STEP 12: Truncated manifest + planner
    // -------------------------------------

    // --------------------------------------------------
    // PRELOAD VERIFIED DEPENDENCY TARGETS FOR ON-DEMAND
    // --------------------------------------------------

    const plannerFilePriorities = new Map(contextFilePriorities);
    const dependencyPriorityByFile = new Map<string, number>();

    const relationshipPriority: Record<string, number> = {
      "renders theme block": 180,
      "defines preset theme block": 160,
      "renders snippet": 80,
      "uses block setting": 60,
      "uses global setting": 40,
      "defines setting": 30,
    };

    const settingsSchemaRequested = isSettingsSchemaIntent(prompt);

    for (const dependency of (explicitOnly ? [] : relevantDependencies)) {
      if (!dependency.verified) continue;

      // Rendering dependencies point to their target file. A setting
      // definition instead points from config/settings_schema.json to a
      // synthetic setting:global.* node, so preload the source file.
      const target = normalizePath(
        dependency.relationship === "defines setting"
          ? dependency.from
          : dependency.to
      );
      const isThemeDependency =
        target.startsWith("snippets/") ||
        target.startsWith("blocks/") ||
        target.startsWith("sections/") ||
        (dependency.relationship === "defines setting" && target.startsWith("config/"));

      if (
        !isThemeDependency ||
        !ALLOWED_EXTENSIONS.includes(path.extname(target)) ||
        !availableThemeFiles.has(target)
      ) {
        continue;
      }

      const settingsIntentBoost =
        settingsSchemaRequested &&
        dependency.relationship === "defines setting" &&
        target === "config/settings_schema.json"
          ? 1200
          : 0;

      const priority =
        dependency.relevanceScore * 1.5 +
        (relationshipPriority[dependency.relationship] ?? 0) +
        settingsIntentBoost;

      dependencyPriorityByFile.set(
        target,
        Math.max(dependencyPriorityByFile.get(target) ?? 0, priority)
      );
    }

    // Apply priorities even when a dependency source file is already loaded.
    // Previously only newly-preloaded dependency targets received this map,
    // which meant config/settings_schema.json remained low-priority precisely
    // when its full content was already available in shared retrieval state.
    for (const [file, priority] of dependencyPriorityByFile) {
      plannerFilePriorities.set(
        file,
        Math.max(plannerFilePriorities.get(file) ?? 0, priority)
      );
    }

    const dependencyPreloadCandidates = [...dependencyPriorityByFile.entries()]
      .filter(([file]) => !retrievalState.fullContents.has(file))
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, MAX_ON_DEMAND_FILES * 2);

    for (const [file, priority] of dependencyPreloadCandidates) {
      const content = await ensureFullContent(retrievalState, file);

      if (content === null || content.length === 0) continue;

      plannerFilePriorities.set(
        file,
        Math.max(plannerFilePriorities.get(file) ?? 0, priority)
      );

      console.log("ON-DEMAND DEPENDENCY CANDIDATE LOADED:", {
        file,
        totalLength: content.length,
        priorityScore: plannerFilePriorities.get(file),
      });
    }

    const plannerManifest = buildPlannerFileManifest(
      retrievalState,
      relevantFiles,
      plannerFilePriorities
    );

    const fileLengths = new Map<string, number>(
      plannerManifest.files.map((file) => [
        file.file,
        file.totalLength,
      ])
    );

    console.log("\n========== COMPLETE PLANNER FILE MANIFEST ==========");
    console.log(JSON.stringify(plannerManifest, null, 2));

    // The planner receives exact lengths for every loaded file,
    // including files skipped entirely from initial context.
    const planningResponse = await generateContentWithRetry({
      model: MODEL,
      contents,
      config: {
        responseMimeType: "application/json",

        responseSchema: {
          type: "OBJECT",
          properties: {
            need_chunks: {
              type: "ARRAY",
              items: {
                type: "OBJECT",
                properties: {
                  file: { type: "STRING" },
                  start_offset: { type: "NUMBER" },
                  max_chars: { type: "NUMBER" },
                },
                required: ["file"],
              },
            },
          },
          required: ["need_chunks"],
        },

        systemInstruction: `
You are a Shopify Horizon Theme source-code analyst.

Analyze the user's request and the supplied theme context.

Your task is ONLY to identify which additional source chunks
are required to provide a fully verified answer.

Rules:

1. Check which relevant source ranges are missing from GEMINI CONTEXT.
2. Request only chunks that are genuinely needed.
3. Use exact file paths from AVAILABLE THEME FILE PATHS.
4. Never invent file paths.
5. Request chunks from a maximum of ${MAX_ON_DEMAND_FILES} unique files.
6. The COMPLETE LOADED FILE MANIFEST below gives the exact
   totalLength for every file available to the retrieval system.
7. Never guess a file length or invent an offset.
8. Any numeric start_offset MUST satisfy:
   0 <= start_offset < totalLength for that exact file.
9. If the user needs the final max_chars characters of a file,
   calculate start_offset = Math.max(0, totalLength - max_chars).
10. If a file has already-retrieved ranges, avoid requesting
    content entirely inside retrievedRanges.
11. For a file with no retrieved ranges, start_offset may be 0
    or another valid offset based on the user's request.
12. If start_offset is omitted, the backend retrieves from
    the next available source range.
13. Request only files listed in the COMPLETE LOADED FILE MANIFEST.
14. First classify the user's request as NARROW or BROAD.
    A request to analyze/review/understand the theme generally,
    identify architecture, summarize major functionality, or
    inspect the source as a whole is BROAD.
15. For a BROAD request, do not conclude that three or fewer
    initially included files represent the whole theme. Request
    additional chunks from the highest-priority files that are
    not fully included, until the available on-demand budget or
    unique-file limit prevents further retrieval.
16. For a NARROW request, request only source needed to verify
    the specific task, behavior, setting, or dependency.
17. Prefer files with the highest priorityScore. Skip a file
    only when it is irrelevant to the user's request or its
    complete relevant source is already included.
18. For each selected file, use its nextOffset when valid and
    request up to 6000 characters. Do not request an already
    retrieved range. If nextOffset is inside the file, prefer it
    over inventing another offset.
19. An empty need_chunks array is valid only when the supplied
    source is sufficient for the full scope of the user's request,
    or no additional relevant source is available in this manifest.
20. Do not request unrelated files.
21. Return no more than ${MAX_ON_DEMAND_FILES} unique files.

COMPLETE LOADED FILE MANIFEST:
${JSON.stringify(plannerManifest.files, null, 2)}

Return ONLY valid JSON:

{
  "need_chunks": [
    {
      "file": "sections/header.liquid",
      "start_offset": 18000,
      "max_chars": 6000
    }
  ]
}

If no additional chunks are needed:

{
  "need_chunks": []
}
`,
      },
    });

    const planningText = planningResponse.text ?? "";

    console.log(
      "\n========== RAW ON-DEMAND PLANNER RESPONSE ==========\n",
      planningText
    );

    const rawPlannerRequestedChunks = parsePlannerResponse(
      planningText,
      new Set(plannerManifest.files.map((file) => file.file)),
      fileLengths
    );

    const availableCoverageDomains = explicitOnly
      ? []
      : getAvailableCoverageDomains(prompt, plannerManifest.files);

    // Settings definitions are source objects in config/settings_schema.json,
    // not ordinary dependency.to files. Retrieve the exact objects referenced
    // by included source files instead of asking the planner to guess offsets.
    const settingDefinitionRequests = explicitOnly
      ? []
      : buildSettingDefinitionChunkRequests(
          prompt,
          relevantDependencies,
          retrievalState
        );
    const liquidSectionSettingRequests = explicitOnly
      ? []
      : buildLiquidSchemaSettingDefinitionRequests(
          prompt,
          relevantDependencies,
          retrievalState
        );
    const deterministicSettingRequests = [
      ...settingDefinitionRequests,
      ...liquidSectionSettingRequests,
    ];
    const settingDefinitionFiles = new Set(
      deterministicSettingRequests.map((request) => request.file)
    );

    // Reserve unique-file slots for requested coverage areas and deterministic
    // setting-definition ranges before accepting planner-generated files.
    const plannerRequestLimit = Math.max(
      0,
      MAX_ON_DEMAND_FILES - availableCoverageDomains.length - settingDefinitionFiles.size
    );

    const plannerRequestedChunks = rawPlannerRequestedChunks
      .filter((request) => !settingDefinitionFiles.has(request.file))
      // The sticky-header toggle is defined in sections/header.liquid's
      // `{% schema %}` block, not in the global theme settings schema.
      // Avoid spending the limited on-demand budget on that unrelated file.
      .filter((request) => !(
        availableCoverageDomains.includes("sticky-header") &&
        normalizePath(request.file).toLowerCase() === "config/settings_schema.json"
      ))
      .slice(0, plannerRequestLimit)
      .filter((request) => isPathAllowedByScope(request.file, explicitFileAllowlist));

    const plannedRequests = [
      ...deterministicSettingRequests,
      ...plannerRequestedChunks,
    ];
    const plannedUniqueFiles = new Set(
      plannedRequests.map((request) => request.file)
    );
    const remainingCoverageSlots = Math.max(
      0,
      MAX_ON_DEMAND_FILES - plannedUniqueFiles.size
    );

    if (rawPlannerRequestedChunks.length > plannerRequestedChunks.length) {
      console.log("PLANNER REQUESTS TRIMMED TO PRESERVE TARGETED / DOMAIN COVERAGE:", {
        rawPlannerRequested: rawPlannerRequestedChunks.length,
        acceptedPlannerRequests: plannerRequestedChunks.length,
        reservedCoverageDomains: availableCoverageDomains,
        targetedSettingFiles: [...settingDefinitionFiles],
      });
    }

    // Explicit-only mode disables dependency discovery, not completion of the
    // exact files the user explicitly selected. Let the deterministic backend
    // coverage planner request missing ranges from the current manifest; the
    // allowlist filter below remains the hard boundary, so no unselected file
    // can be retrieved. In explicit-only mode, do not reserve slots for
    // dependency/settings discovery: all available slots belong to selected
    // files whose source is still incomplete.
    const explicitIncompleteFileCount = explicitOnly
      ? plannerManifest.files.filter((file) =>
          isPathAllowedByScope(file.file, explicitFileAllowlist) &&
          file.hasMore &&
          file.totalLength > 0
        ).length
      : 0;

    const backendCoverageRequests = buildBackendCoverageRequests(
      plannerManifest.files,
      plannedRequests,
      explicitOnly
        ? Math.min(MAX_ON_DEMAND_FILES, explicitIncompleteFileCount)
        : remainingCoverageSlots,
      // Keep explicit-only retrieval inside the user's allowlist, but retain
      // the detected domain so targeted source windows can be selected. For a
      // sticky-header request this prioritizes actual sticky CSS selectors and
      // declarations in base.css instead of spending the entire budget on the
      // next sequential ranges. The allowlist remains the hard retrieval boundary.
      prompt,
      retrievalState.fullContents
    );

    const coverageCandidateRequests = [
      ...plannedRequests,
      ...backendCoverageRequests,
    ];

    const requestedChunks = allocateChunkRequestBudgets(
      coverageCandidateRequests.filter((item) => isPathAllowedByScope(item.file, explicitFileAllowlist)),
      plannerManifest.files.filter((item) => isPathAllowedByScope(item.file, explicitFileAllowlist)),
      retrievalState.remainingBudget
    );

    console.log("\n========== PLANNER + BACKEND COVERAGE REQUESTS ==========");
    console.log({
      coverageDomains: availableCoverageDomains,
      rawPlannerRequestedChunks,
      settingDefinitionRequests,
      liquidSectionSettingRequests,
      plannerRequestedChunks,
      backendCoverageRequests,
      coverageCandidateRequests,
      budgetAllocatedRequests: requestedChunks,
      budgetAvailable: retrievalState.remainingBudget,
      budgetAllocated: requestedChunks.reduce(
        (total, item) => total + item.max_chars,
        0
      ),
    });





    // -------------------------------------
    // STEP 12.1: Retrieve planner-requested chunks
    // -------------------------------------

    const plannerChunks: RetrievedChunk[] = [];

    for (const req of requestedChunks) {
      if (retrievalState.remainingBudget <= 0) {
        console.log(
          "ON-DEMAND BUDGET EXHAUSTED — STOPPING PLANNER RETRIEVAL"
        );
        break;
      }

      const fullContent = await ensureFullContent(
        retrievalState,
        req.file
      );

      if (fullContent === null) {
        console.log(
          "PLANNER REQUESTED FILE NOT FOUND:",
          req.file
        );
        continue;
      }

      const chunk = retrieveChunk(
        retrievalState,
        req.file,
        fullContent,
        req.start_offset,
        req.max_chars
      );

      if (!chunk) {
        console.log(
          "PLANNER CHUNK RETRIEVAL SKIPPED:",
          {
            file: req.file,
            requestedStart: req.start_offset,
            requestedMaxChars: req.max_chars,
          }
        );
        continue;
      }

      plannerChunks.push(chunk);
    }

    // Explicit-only retrieval gets a bounded second pass over the same
    // allowlisted files. The first allocation can leave budget unused because
    // each candidate is capped at MAX_CHUNK_SIZE; continue from the next
    // uncovered offset while meaningful budget and incomplete files remain.
    if (explicitOnly) {
      const MAX_EXPLICIT_COVERAGE_ROUNDS = 4;
      for (let round = 2; round <= MAX_EXPLICIT_COVERAGE_ROUNDS; round++) {
        if (retrievalState.remainingBudget < 1200) break;

        const nextRequests: PlannerChunkRequest[] = Array.from(
          retrievalState.fileStates.values()
        )
          .filter((fileState) =>
            isPathAllowedByScope(fileState.file, explicitFileAllowlist) &&
            fileState.hasMore &&
            fileState.nextOffset < fileState.totalLength
          )
          .sort((a, b) => a.file.localeCompare(b.file))
          .slice(0, MAX_ON_DEMAND_FILES)
          .map((fileState) => ({
            file: fileState.file,
            start_offset: fileState.nextOffset,
            max_chars: Math.min(
              MAX_CHUNK_SIZE,
              fileState.totalLength - fileState.nextOffset
            ),
          }));

        if (nextRequests.length === 0) break;

        const currentManifest: PlannerManifestFile[] = nextRequests.map((request) => {
          const fileState = retrievalState.fileStates.get(request.file)!;
          const original = plannerManifest.files.find((item) => item.file === request.file);
          return {
            file: fileState.file,
            totalLength: fileState.totalLength,
            priorityScore: original?.priorityScore ?? 1,
            includedInContext: true,
            retrievedRanges: [...fileState.retrievedRanges],
            nextOffset: fileState.nextOffset,
            hasMore: fileState.hasMore,
            isComplete: fileState.isComplete,
          };
        });

        const roundRequests = allocateChunkRequestBudgets(
          nextRequests,
          currentManifest,
          retrievalState.remainingBudget
        );

        console.log(`EXPLICIT-ONLY ADDITIONAL COVERAGE ROUND ${round}:`, {
          candidates: nextRequests,
          allocatedRequests: roundRequests,
          budgetBefore: retrievalState.remainingBudget,
        });

        if (roundRequests.length === 0) break;

        let roundRetrieved = 0;
        for (const request of roundRequests) {
          const fullContent = await ensureFullContent(retrievalState, request.file);
          if (fullContent === null) continue;
          const chunk = retrieveChunk(
            retrievalState,
            request.file,
            fullContent,
            request.start_offset,
            request.max_chars
          );
          if (!chunk) continue;
          plannerChunks.push(chunk);
          roundRetrieved++;
          if (!filesAnalyzed.includes(chunk.file)) filesAnalyzed.push(chunk.file);
          if (!filesIncludedInContext.includes(chunk.file)) filesIncludedInContext.push(chunk.file);
          const skippedIdx = filesSkippedFromContext.indexOf(chunk.file);
          if (skippedIdx !== -1) filesSkippedFromContext.splice(skippedIdx, 1);
        }

        console.log(`EXPLICIT-ONLY ADDITIONAL COVERAGE ROUND ${round} RESULT:`, {
          requested: roundRequests.length,
          retrieved: roundRetrieved,
          usedBudget: retrievalState.usedBudget,
          remainingBudget: retrievalState.remainingBudget,
        });
        if (roundRetrieved === 0) break;
      }
    }

    console.log(
      "\n========== CHUNK RETRIEVAL PROGRESS =========="
    );

    console.log({
      stage: "planner",
      requestedChunks: requestedChunks.length,
      retrievedChunks: plannerChunks.length,
      usedBudget: retrievalState.usedBudget,
      remainingBudget: retrievalState.remainingBudget,
      uniqueFiles: retrievalState.uniqueFiles.size,
    });

    for (const chunk of plannerChunks) {
      if (!filesAnalyzed.includes(chunk.file)) {
        filesAnalyzed.push(chunk.file);
      }

      if (!filesIncludedInContext.includes(chunk.file)) {
        filesIncludedInContext.push(chunk.file);
      }

      const skippedIdx = filesSkippedFromContext.indexOf(
        chunk.file
      );

      if (skippedIdx !== -1) {
        filesSkippedFromContext.splice(skippedIdx, 1);
      }
    }






    // -------------------------------------
    // STEP 23: DEPENDENCY CHUNK EXPANSION (shared state)
    // -------------------------------------

    console.log("\n========== DEPENDENCY CHUNK EXPANSION ==========");

    const MAX_DEPENDENCY_EXPANSION_ROUNDS = 3;

    function getDependencyKey(dep: Dependency): string {
      return [dep.from, dep.to, dep.relationship, dep.evidence].join("::");
    }

    const verifiedDependencyKeys = new Set(
      verifiedDependencies.map(getDependencyKey)
    );

    let expansionQueue: RetrievedChunk[] = explicitOnly ? [] : [...plannerChunks];

    if (explicitOnly) {
      console.log("EXPLICIT-ONLY: dependency chunk expansion skipped.");
    }

    for (
      let round = 1;
      !explicitOnly && round <= MAX_DEPENDENCY_EXPANSION_ROUNDS;
      round++
    ) {
      console.log(`\n--- DEPENDENCY EXPANSION ROUND ${round} ---`);

      if (expansionQueue.length === 0) {
        console.log("No new chunks to expand");
        break;
      }

      // Extract dependencies only from the genuinely new portions
      // of each newly retrieved chunk.
      const newlyExtractedDependencies: Dependency[] = [];

      for (const chunk of expansionQueue) {



        const newContent = extractNewContentForDependencies(
          retrievalState,
          chunk
        );

        if (!newContent) continue;


        try {
          const extracted = extractDependencies(
            chunk.file,
            newContent,
            prompt
          );

          newlyExtractedDependencies.push(...extracted);

          // Mark range as processed only after successful extraction.
          retrievalState.processedDependencyRanges.set(
            chunk.file,
            mergeRanges([
              ...(retrievalState.processedDependencyRanges.get(
                chunk.file
              ) ?? []),
              {
                startOffset: chunk.startOffset,
                endOffset: chunk.endOffset,
              },
            ])
          );

        } catch (error) {
          console.error("DEPENDENCY EXTRACTION FAILED:", {
            file: chunk.file,
            startOffset: chunk.startOffset,
            endOffset: chunk.endOffset,
            error,
          });
        }








      }

      if (newlyExtractedDependencies.length === 0) {
        console.log("No new dependencies extracted from new chunks");
        break;
      }

      dependencies = uniqueDependencies([
        ...dependencies,
        ...newlyExtractedDependencies,
      ]);

      if (schemaContent) {
        const definitions = buildSettingDefinitionDependencies(
          dependencies,
          schemaContent,
          prompt
        );
        dependencies = uniqueDependencies([...dependencies, ...definitions]);
      }

      // Verify any newly added dependencies
      let newlyVerifiedCount = 0;
      for (const dep of dependencies) {
        const key = getDependencyKey(dep);
        if (verifiedDependencyKeys.has(key)) continue;
        const verification = await verifyDependency(
          dep,
          schemaContent,
          dependencies
        );
        verifiedDependencies.push(verification);
        verifiedDependencyKeys.add(key);
        if (verification.verified) newlyVerifiedCount++;
      }

      // Update relevant dependency graph with the current roots
      const availableRootFiles: ThemeFile[] = Array.from(
        retrievalState.fileStates.keys()
      ).map((f) => ({ file: f, content: "", score: 0 }));

      const updatedRelevantDependencies = filterRelevantDependencies(
        verifiedDependencies,
        [...initialFiles, ...availableRootFiles]
      );
      relevantDependencies.splice(
        0,
        relevantDependencies.length,
        ...updatedRelevantDependencies
      );

      // Choose dependency files that are not fully retrieved yet.
      const dependencyCandidates = relevantDependencies
        .filter((dep): dep is VerifiedDependency => {
          if (!dep.verified) return false;
          if (dep.relevanceScore < MIN_DEPENDENCY_SCORE) return false;
          return (
            dep.to.startsWith("snippets/") ||
            dep.to.startsWith("blocks/") ||
            dep.to.startsWith("sections/")
          );
        })
        .filter((dep) => {
          const target = normalizePath(dep.to);
          if (!availableThemeFiles.has(target)) return false;
          if (!ALLOWED_EXTENSIONS.includes(path.extname(target)))
            return false;

          const fs = retrievalState.fileStates.get(target);

          if (fs && fs.isComplete) return false;

          const alreadyRetrieved = retrievalState.retrievedChunks
            .filter(chunk => chunk.file === target)
            .reduce(
              (total, chunk) => total + chunk.content.length,
              0
            );

          if (
            alreadyRetrieved >= MAX_ON_DEMAND_FILE_SIZE
          ) {
            return false;
          }

          if (
            !retrievalState.uniqueFiles.has(target) &&
            retrievalState.uniqueFiles.size >= retrievalState.maxOnDemandFiles
          ) {
            return false;
          }
          return true;
        })
        .sort((a, b) => b.relevanceScore - a.relevanceScore);

      const candidatePaths = [
        ...new Set(
          dependencyCandidates.map((d) => normalizePath(d.to))
        ),
      ];

      console.log("NEW VERIFIED DEPENDENCIES:", newlyVerifiedCount);
      console.log("DEPENDENCY CANDIDATES:", candidatePaths);

      if (candidatePaths.length === 0) {
        console.log("NO MORE RELEVANT DEPENDENCY FILES");
        break;
      }

      const nextQueue: RetrievedChunk[] = [];

      for (const target of candidatePaths) {
        if (retrievalState.remainingBudget <= 0) {
          console.log("ON-DEMAND CONTEXT BUDGET REACHED — STOPPING EXPANSION");
          break;
        }
        if (
          !retrievalState.uniqueFiles.has(target) &&
          retrievalState.uniqueFiles.size >= retrievalState.maxOnDemandFiles
        ) {
          console.log("MAX ON-DEMAND UNIQUE FILES REACHED");
          break;
        }

        const fullContent = await ensureFullContent(retrievalState, target);
        if (fullContent === null) continue;

        const chunk = retrieveChunk(
          retrievalState,
          target,
          fullContent,
          null,
          MAX_CHUNK_SIZE
        );
        if (!chunk) continue;

        nextQueue.push(chunk);

        if (!filesAnalyzed.includes(chunk.file))
          filesAnalyzed.push(chunk.file);
        if (!filesIncludedInContext.includes(chunk.file))
          filesIncludedInContext.push(chunk.file);
        const skippedIdx = filesSkippedFromContext.indexOf(chunk.file);
        if (skippedIdx !== -1)
          filesSkippedFromContext.splice(skippedIdx, 1);
      }

      console.log(
        `Round ${round} retrieved ${nextQueue.length} new dependency chunks`
      );
      console.log({
        round,
        usedBudget: retrievalState.usedBudget,
        remainingBudget: retrievalState.remainingBudget,
        uniqueFiles: retrievalState.uniqueFiles.size,
        totalRetrievedChunks: retrievalState.retrievedChunks.length,
      });

      expansionQueue = nextQueue;

      if (expansionQueue.length === 0) {
        console.log("Expansion queue empty — stopping");
        break;
      }
    }

    // --------------------------------------------------
    // v35: FINAL EXPLICIT FILE SCOPE AUDIT + DEFENSIVE PRUNING
    // --------------------------------------------------
    if (explicitOnly && explicitFileAllowlist) {
      const allowed = (file: string) => isPathAllowedByScope(file, explicitFileAllowlist);
      const scopeViolations: string[] = [];
      const auditArray = (label: string, values: string[]) => {
        for (const file of values) if (!allowed(file)) scopeViolations.push(`${label}: ${file}`);
      };

      auditArray("filesAnalyzed", filesAnalyzed);
      auditArray("filesIncludedInContext", filesIncludedInContext);
      auditArray("filesSkippedFromContext", filesSkippedFromContext);
      auditArray("selectedFiles", [...selectedFiles.keys()]);
      auditArray("initialFiles", initialFiles.map((file) => file.file));
      auditArray("relevantFiles", relevantFiles.map((file) => file.file));
      auditArray("plannerManifest.files", plannerManifest.files.map((file) => file.file));
      auditArray("fileLengths", [...fileLengths.keys()]);
      auditArray("requestedChunks", requestedChunks.map((request) => request.file));
      auditArray("plannerChunks", plannerChunks.map((chunk) => chunk.file));
      auditArray("retrievalState.uniqueFiles", [...retrievalState.uniqueFiles]);
      auditArray("retrievalState.fileStates", [...retrievalState.fileStates.keys()]);
      auditArray("retrievalState.fullContents", [...retrievalState.fullContents.keys()]);
      auditArray("retrievalState.retrievedChunks", retrievalState.retrievedChunks.map((chunk) => chunk.file));

      if (scopeViolations.length > 0) {
        console.error("EXPLICIT FILE SCOPE VIOLATIONS DETECTED:", scopeViolations);
      }

      for (const file of [...selectedFiles.keys()]) if (!allowed(file)) selectedFiles.delete(file);
      for (const file of [...retrievalState.fileStates.keys()]) if (!allowed(file)) retrievalState.fileStates.delete(file);
      for (const file of [...retrievalState.fullContents.keys()]) if (!allowed(file)) retrievalState.fullContents.delete(file);
      for (const file of [...retrievalState.processedDependencyRanges.keys()]) if (!allowed(file)) retrievalState.processedDependencyRanges.delete(file);
      retrievalState.retrievedChunks = retrievalState.retrievedChunks.filter((chunk) => allowed(chunk.file));
      retrievalState.uniqueFiles = new Set([...retrievalState.uniqueFiles].filter(allowed));

      for (const [file] of [...includedSourceContent.entries()]) if (!allowed(file)) includedSourceContent.delete(file);
      filesAnalyzed.splice(0, filesAnalyzed.length, ...filesAnalyzed.filter(allowed));
      filesIncludedInContext.splice(0, filesIncludedInContext.length, ...filesIncludedInContext.filter(allowed));
      filesSkippedFromContext.splice(0, filesSkippedFromContext.length, ...filesSkippedFromContext.filter(allowed));
      initialFiles.splice(0, initialFiles.length, ...initialFiles.filter((file) => allowed(file.file)));
      relevantFiles.splice(0, relevantFiles.length, ...relevantFiles.filter((file) => allowed(file.file)));
      dependencies = [];
      verifiedDependencies.splice(0, verifiedDependencies.length);
      relevantDependencies.splice(0, relevantDependencies.length);

      console.log("\n========== EXPLICIT FILE SCOPE AUDIT ==========");
      console.log({
        explicitOnly: true,
        allowedFiles: [...explicitFileAllowlist],
        scopeViolations,
        filesAnalyzed,
        filesIncludedInContext,
        filesSkippedFromContext,
        selectedFiles: [...selectedFiles.keys()],
        plannerManifestFiles: plannerManifest.files.map((file) => file.file),
        fileLengthKeys: [...fileLengths.keys()],
        requestedChunks: requestedChunks.map((request) => request.file),
        plannerChunks: plannerChunks.map((chunk) => chunk.file),
        retrievalState: {
          uniqueFiles: [...retrievalState.uniqueFiles],
          fileStates: [...retrievalState.fileStates.keys()],
          fullContents: [...retrievalState.fullContents.keys()],
          retrievedChunks: retrievalState.retrievedChunks.map((chunk) => chunk.file),
        },
      });
    }

    dependencies = uniqueDependencies(dependencies);

    const finalRelevantDependencies = filterRelevantDependencies(
      verifiedDependencies,
      [
        ...initialFiles,
        ...Array.from(retrievalState.fileStates.keys()).map((f) => ({
          file: f,
          content: "",
          score: 0,
        })),
      ]
    );
    relevantDependencies.splice(
      0,
      relevantDependencies.length,
      ...finalRelevantDependencies
    );

    // -------------------------------------
    // STEP 12.2: Rebuild includedSourceContent and append chunks
    // -------------------------------------

    // Rebuild includedSourceContent from the shared state so the
    // verifier sees exactly the merged ranges that Gemini sees.
    const rebuiltIncludedSourceContent =
      buildIncludedSourceContentFromState(retrievalState);

    includedSourceContent.clear();
    for (const [file, content] of rebuiltIncludedSourceContent.entries()) {
      includedSourceContent.set(file, content);
    }

    const retrievalCoverage = explicitOnly
      ? []
      : buildRetrievalCoverageReport(
          prompt,
          plannerManifest.files,
          retrievalState
        );

    const sourceCoverageManifest = filesIncludedInContext.map((file) => {
      const fileManifest = plannerManifest.files.find(
        (item) => item.file === file
      );
      const fileState = retrievalState.fileStates.get(file);
      const totalLength = fileManifest?.totalLength ?? fileState?.totalLength ?? 0;
      const retrievedRanges = fileState?.retrievedRanges ?? [];
      const includedCharacters = retrievedRanges.reduce(
        (total, range) => total + (range.endOffset - range.startOffset),
        0
      );
      const complete = hasCompleteSourceCoverage(totalLength, retrievedRanges);

      return {
        file,
        totalLength,
        includedCharacters,
        coveragePercent: totalLength > 0
          ? Number(((includedCharacters / totalLength) * 100).toFixed(1))
          : 0,
        retrievedRanges,
        complete,
      };
    });

    if (explicitOnly && explicitFileAllowlist) {
      const coverageScopeViolations = sourceCoverageManifest
        .filter((entry) => !isPathAllowedByScope(entry.file, explicitFileAllowlist))
        .map((entry) => entry.file);
      if (coverageScopeViolations.length > 0) {
        console.error("EXPLICIT FILE SCOPE VIOLATIONS IN SOURCE COVERAGE MANIFEST:", coverageScopeViolations);
        for (let index = sourceCoverageManifest.length - 1; index >= 0; index--) {
          if (!isPathAllowedByScope(sourceCoverageManifest[index].file, explicitFileAllowlist)) {
            sourceCoverageManifest.splice(index, 1);
          }
        }
      }
      console.log("EXPLICIT FILE SCOPE COVERAGE AUDIT:", {
        allowedFiles: [...explicitFileAllowlist],
        sourceCoverageFiles: sourceCoverageManifest.map((entry) => entry.file),
      });
    }

    if (retrievalState.retrievedChunks.length > 0) {
      const chunksSection = retrievalState.retrievedChunks
        .map((chunk) => {
          return `=== ${chunk.file} [range ${chunk.startOffset}-${chunk.endOffset} of ${chunk.totalLength}] ===
hasMore: ${chunk.hasMore}
isComplete: ${chunk.isComplete}

${chunk.content}
`;
        })
        .join("\n");

      const lastUserPart = contents[contents.length - 1].parts[0];
      lastUserPart.text += `

ADDITIONAL ON-DEMAND SOURCE CHUNKS:

${chunksSection}

UPDATED VERIFIED DEPENDENCY GRAPH:

${JSON.stringify(relevantDependencies, null, 2)}

FILES INCLUDED IN GEMINI CONTEXT:
${JSON.stringify(filesIncludedInContext, null, 2)}

FILES SKIPPED FROM CONTEXT:
${JSON.stringify(filesSkippedFromContext, null, 2)}

IMPORTANT:
- The chunks above are the ONLY additional source content available.
- Files listed in the initial context remain valid.
- Files NOT in the initial context and NOT in these chunks are NOT available.
- Do not claim that a file was fully analyzed if only a chunk was retrieved.
- Do not make any claims about the internal API implementation,
  architecture, or budget management — you cannot see the server code.
- Only use the exact source content above as evidence.
`;
    }

    // Always give Gemini the backend-calculated source coverage so it
    // cannot describe a partial chunk as a complete file analysis.
    const finalUserTextPart = contents[contents.length - 1].parts[0];
    const dependencyCallerAllowlist = explicitOnly
      ? []
      : sourceCoverageManifest
          .filter((entry) => entry.includedCharacters > 0)
          .map((entry) => entry.file);

    finalUserTextPart.text += `

SOURCE COVERAGE STATUS (calculated from the exact source ranges supplied):
${JSON.stringify(sourceCoverageManifest, null, 2)}

DEPENDENCY CLAIM CALLER ALLOWLIST:
${JSON.stringify(dependencyCallerAllowlist, null, 2)}
Only files in this allowlist may be used as the \"file\" caller for a dependency claim.
Do not create dependency claims from graph edges whose caller file is absent from this allowlist,
even if that file appears in the dependency graph or recursive-loading list.

STRICT COVERAGE RULES:
- A file is COMPLETE only when its complete flag is true above.
- If complete is false, describe it only as partial/incomplete and state the supplied ranges when relevant.
- Never say that full/complete source was available or fully analyzed unless complete is true.
- A file absent from this coverage list was not supplied to you as source content.
- Do not infer completeness from a file path appearing in a dependency graph, manifest, or search result.
`;

    // -------------------------------------
    // Final retrieval summary
    // -------------------------------------

    console.log("\n========== FINAL CHUNK RETRIEVAL SUMMARY ==========");
    console.log({
      usedBudget: retrievalState.usedBudget,
      remainingBudget: retrievalState.remainingBudget,
      totalBudget: retrievalState.totalBudget,
      uniqueFiles: [...retrievalState.uniqueFiles],
      retrievedChunkCount: retrievalState.retrievedChunks.length,
      retrievalCoverage,
      fileStates: Array.from(retrievalState.fileStates.entries()).map(
        ([file, fs]) => ({
          file,
          totalLength: fs.totalLength,
          retrievedRanges: fs.retrievedRanges,
          nextOffset: fs.nextOffset,
          hasMore: fs.hasMore,
          isComplete: fs.isComplete,
        })
      ),
    });

    // -------------------------------------
    // STEP 13: Final Gemini response
    // -------------------------------------

    const response =
      await generateContentWithRetry({
        model: MODEL,
        contents,
        config: {
          responseMimeType: "application/json",

          responseSchema: {
            type: "OBJECT",
            properties: {
              answer: {
                type: "STRING",
              },

              claims: {
                type: "ARRAY",
                items: {
                  type: "OBJECT",
                  properties: {
                    id: {
                      type: "STRING",
                    },

                    claim: {
                      type: "STRING",
                    },

                    file: {
                      type: "STRING",
                    },

                    evidence: {
                      type: "ARRAY",
                      items: {
                        type: "STRING",
                      },
                    },
                  },
                  required: [
                    "id",
                    "claim",
                    "file",
                    "evidence",
                  ],
                },
              },
            },

            required: [
              "answer",
              "claims",
            ],
          },

          systemInstruction: `
You are Horizon AI, an expert Shopify Horizon Theme Developer Agent.

Analyze the actual Shopify Horizon theme source code.

Return ONLY valid JSON in this exact structure:

{
  "answer": "Your complete explanation in markdown format.",
  "claims": [
    {
      "id": "C1",
      "claim": "A factual statement about the theme source code.",
      "file": "snippets/example.liquid",
      "evidence": ["Exact source code text copied from that file."]
    }
  ]
}

RULES:

1. Use only source code provided in the context.
2. Never invent file paths, settings, variables or relationships.
3. EVERY factual statement about theme source code in the answer must have a matching
   entry in the claims array. This includes each factual bullet, numbered item,
   dependency statement, file behavior, and code explanation.
4. ATOMIC CLAIM REQUIREMENT (MANDATORY): Each claim must assert exactly ONE
   independently verifiable behavior or relationship. Never combine multiple
   actions, settings, snippets, or independent outcomes into one claim.
   Create separate claim IDs for each fact. When one code block handles several
   distinct template cases (home, product, collection), create a separate claim
   for each case. For dependency claims, create one claim per caller/target edge.
   Do NOT combine variant-price lookup, compare-at comparison, currency handling,
   or multiple snippet renders into one claim.
   Examples of REQUIRED splitting:
   - A transparent-header template rule may be one narrowly scoped claim only
     when one contiguous excerpt contains its exact template guard, the
     \`assign transparent = 'always'\` assignment, and its matching
     \`transparent_text_color_setting\` assignment. Otherwise split the facts.
   - Do not write "uses a case statement to determine markup". State only:
     "The file uses a Liquid \`case\` on \`variant\`."
   - Do not combine a render relationship with a separate branch check. Keep
     the setting check and the caller-to-target render edge as separate facts.
5. The file must exist in FILES INCLUDED IN GEMINI CONTEXT or ADDITIONAL FILES RETRIEVED ON DEMAND.
6. Evidence MUST be an ARRAY of one or more exact source-code excerpts copied
   verbatim from the provided context. Do not paraphrase or repair the code.
7. Every excerpt must independently be an exact contiguous substring of the
   named file. If one atomic behavior depends on two separate code locations,
   include both exact excerpts as separate evidence array items.
   NEVER insert ellipses ("..." or "…"), omitted-code placeholders, or
   synthetic line gaps inside an evidence excerpt. Copy each exact range separately.
8. Do not invent evidence. Do not use one excerpt to imply unrelated behavior.
   For dependency claims, create exactly ONE caller-to-target relationship per
   claim ID. Never combine multiple render targets in one dependency claim.
9. Keep each excerpt short and specific; include only excerpts that directly
   support the single behavior asserted by that claim.
10. If a claim cannot be supported by source code, do not include it as a verified claim.
10A. STICKY HEADER SOURCE EVIDENCE: When analyzing \`enable_sticky_header\`, quote the complete setting object as one evidence item, including its opening and closing braces. Do not split the object into isolated property lines. Identify the literal option values exactly as written (for example, \`scroll-up\`); do not substitute an option's translation label (for example, \`t:options.on_scroll_up\`) for its value. Each setting-field claim must identify \`enable_sticky_header\` and state only the fields actually quoted.
10B. STICKY LIQUID BRANCHES: Quote the exact source syntax. If code is inside Shopify's \`{% liquid %}\` block, inner \`if\`, \`elsif\`, \`assign\`, and \`endif\` statements are plain lines; do NOT wrap them in invented \`{% if %}\` or \`{% elsif %}\` tags. Copy source lines verbatim. Create separate atomic claims for the \`always\` and \`scroll-up\` assignments, with each exact condition and matching assignment in the same evidence excerpt. Do not infer what \`never\` does from a missing branch.
10B-1. LIQUID EVIDENCE SYNTAX: Never repair, normalize, or convert Liquid syntax in evidence. A \`{% liquid %}\` block, a normal \`{% if ... %}\` tag, and plain Liquid statements are different source forms. Preserve the form actually present in the supplied source.
10C. STICKY CSS: For each CSS claim, quote the exact selector and its complete relevant declaration block from the same contiguous excerpt. State \`position\`, \`top\`, or \`z-index\` only when that declaration is present in the quoted rule. Do not infer that a selector belongs to the sticky header from its name alone. Keep \`.sticky-content\` and \`.sticky-content--desktop\` as separate atomic claims; when mentioning a top offset, preserve that detail in the matching answer sentence.
10D. STICKY JAVASCRIPT: Treat \`#observeStickyPosition\` and \`#handleWindowScroll\` as method names only. Claim an IntersectionObserver only when evidence includes the actual \`new IntersectionObserver\` construction; claim a scroll listener only when evidence includes the actual listener registration (such as \`addEventListener('scroll', ...)\`) and its relevant state update. Keep observer and scroll-listener claims separate.
10D-1. STICKY METHOD SCOPE: For callback, lifecycle, breakpoint-handler, or observer-branch claims, quote enough exact contiguous source to include the method name/opening, relevant condition, and described operation. If relevant lines are separated by an unretrieved source gap, request the missing range; never combine non-contiguous ranges into one evidence excerpt.
10E. STICKY SCROLL LISTENER CONDITION: If the listener is registered only inside a condition such as \`stickyMode === 'scroll-up' || stickyMode === 'always'\`, include both literal mode values and that exact OR condition in the claim and in the same contiguous evidence excerpt as \`addEventListener('scroll', ...)\`. Use wording such as "when stickyMode is 'scroll-up' or 'always'." Never summarize this as merely "when stickyMode is set" or "when stickyMode is present".
10E-1. STICKY CALLBACK INITIALIZATION: When explaining \`connectedCallback\`, keep the \`getAttribute('sticky')\` read and the \`#observeStickyPosition(stickyMode === 'always')\` call in one exact evidence excerpt. State both operations in the claim.
10E-2. STICKY OBSERVER THRESHOLD: When describing the observer threshold, quote the exact \`threshold: alwaysSticky ? 1 : 0\` config and the observer construction/use from \`#observeStickyPosition\`. Keep threshold claims separate from callback-state assignment claims.
10F. STICKY OBSERVER ELSE BRANCH: To claim that \`#offscreen\` is updated when \`alwaysSticky\` is false, quote the paired \`if (alwaysSticky) { ... } else { ... }\` control flow, including the \`#offscreen\` assignment, in one contiguous excerpt. Do not infer false solely from an isolated \`else\` block. If the paired block is not present in the supplied context, omit this false-branch claim.
10G. STICKY CLAIM-TO-ANSWER ALIGNMENT: Each final sentence must preserve every condition and operation in its cited claim. For scroll listeners, state the literal \`scroll-up\` OR \`always\` guard; if you cannot quote both modes and the registration, omit the claim. For observer threshold claims, do not mention \`root: getIntersectionRoot()\` unless the final sentence also states that root behavior and the evidence quotes it. Keep \`connectedCallback\` attribute reading, boolean observer initialization, observer threshold, observer state assignment, and scroll updates as separate facts.
11. Do not treat relevance scores as proof.
11A. Never add inferred qualifiers such as "when", "only when", "default branch",
     "desktop/mobile variant", or "truthy" unless the cited evidence includes the
     corresponding Liquid/JavaScript control-flow condition. A render argument or
     data attribute containing "mobile" does not prove a mobile-only branch.
11B. Describe direct assignments precisely (for example, "assigns price from
     selected_variant.price"). Do not describe a value as calculated, formatted,
     or conditionally selected unless the exact operation/branch appears in evidence.
11C. When describing an if/else outcome, include the condition and the relevant
     branch operation in the evidence. Prefer separate claims for independent
     branch outcomes; do not infer false solely from the word else without
     showing the paired condition in the same contiguous source context. For
     the use_currency false/else branch, quote one contiguous excerpt that
     starts at if use_currency and continues through else to the
     assign price = price | money operation whenever that source is present.
11D. For a render said to occur in a mobile/default/desktop variant or inside a
     specific condition, include the active Liquid condition/case/when tag and
     the render statement in one evidence excerpt. A data attribute or render
     argument containing a variant name is not proof of branch placement.
11E. Do not add positional or ordering claims such as "at the top", "at the
     bottom", "first", "last", "before", or "after" unless the evidence includes
     enough contiguous source context to prove that position.
11F. When a source contains a script tag with a known asset URL and
     type="module", describe only that it loads the named asset as a module
     script. Do not add a location such as "at the bottom of the template"
     unless the cited excerpt itself includes enough surrounding source to prove it.
11G. MANDATORY PRICE-FILE COVERAGE CHECKLIST: When snippets/price.liquid is
     included and the corresponding exact source is present, you MUST create
     separate claim IDs AND separate answer statements for all three behaviors:
     (a) the use_currency true branch using money_with_currency;
     (b) the use_currency else branch using money. For (b), cite the paired
         if use_currency condition, else, and money assignment together
         in one contiguous evidence excerpt; and
     (c) the unit-price snippet render guarded by both
         selected_variant.unit_price and show_unit_price.
     For each behavior, cite a contiguous excerpt containing its condition/branch
     and operation. Do not omit any checklist item merely because another related
     claim was easier to cite. If a required source excerpt is unavailable, do not
     invent the claim; explicitly identify that coverage limitation instead.
11G-1. Never compose a paired if/else quote by joining separate lines or
       separate source locations. The quoted text must match the source as one
       exact contiguous excerpt (whitespace differences only). If the paired
       use_currency condition and money assignment cannot be quoted exactly,
       omit the false-branch claim and state that the supplied evidence does
       not establish that branch.
11H. When describing the 1999 price assignment for a blank product_resource,
     cite one contiguous excerpt containing both the condition if
     product_resource == blank and the assignment assign price = 1999. Do not
     split those adjacent lines into separate evidence items when they appear
     together in the supplied source.
11I. Treat \`when 'mobile'\` and \`when 'navigation_bar'\` as distinct case
     branches. Do not call the navigation_bar branch a mobile variant unless
     the exact source branch itself proves that relationship. Describe each
     branch using its own literal branch value. If your evidence contains only
     \`{% when 'mobile' %}\`, state only that the file contains that literal
     Liquid case label; do not claim that it renders or handles mobile navigation.
11I-1. To claim that the mobile case renders \`header-drawer\`, include both the
       exact \`when 'mobile'\` tag and the \`render 'header-drawer'\` statement in
       one contiguous evidence excerpt. Otherwise report only the case label.
11I-2. For the \`when 'navigation_bar'\` case, use the exact phrase
       "the navigation_bar case" or "the navigation bar case". Do NOT use the
       word "mobile" in that claim or answer sentence. The presence of a
       navigation-bar setting does not establish that the branch is mobile.
11I-3. A statement that the navigation_bar case checks
       \`block_settings.navigation_bar == true\` must be a separate claim from
       any render relationship. Do not say that \`contrast-override\` or another
       snippet is rendered "when" that setting is true unless ONE contiguous
       evidence excerpt includes the \`when 'navigation_bar'\` tag, the setting
       guard, and the exact render statement. If that full excerpt is not quoted,
       state only the generic caller-to-target render relationship in the
       dependency map.
11I-4. If claiming that \`block_settings.navigation_bar == true\` is checked
       inside the navigation_bar case, the same evidence excerpt must include
       both the literal \`when 'navigation_bar'\` tag and the exact \`if
       block_settings.navigation_bar == true\` guard. A guard-only quote proves
       the guard exists, not which case contains it.
11I-5. For each transparent-header template rule (home/index, product, or
       collection), keep the answer sentence and claim narrowly tied to the one
       matching template guard. If the sentence also names the text-color
       assignment, quote the guard, \`assign transparent = 'always'\`, and the
       matching \`transparent_text_color_setting\` assignment together.
11I-6. When describing the transparent/sticky interaction, state only that
       \`transparent\` is assigned \`not-sticky\` under the exact
       \`if transparent and sticky\` guard. Keep that guard and assignment in
       one evidence excerpt.
11I-7. In \`sections/header.liquid\`, the mobile drawer invocation is captured
       as \`drawer_menu\` using \`content_for 'block'\`, \`_header-menu\`, and
       \`variant: 'mobile'\`. Describe this as a drawer_menu capture, not as a
       \`when 'mobile'\` case render. Quote the complete capture invocation.
11I-8. For product and collection transparent-header rules, accept the exact
       \`elsif\` template guard as the start of the quoted branch. Do not omit
       the matching template name or text-color setting identifier.
11W. Do not add a broad introductory summary that restates several later
       claims without preserving their exact subjects and identifiers. Begin
       with the first exact independently supported source fact instead.
11V. CLAIM-TO-SENTENCE TRACEABILITY: Preserve the concrete identifiers and
       subjects from each claim in the corresponding answer sentence. Do not
       replace \`drawer menu\` with a generic phrase such as "menu variants";
       do not omit \`product.handle\` when the claim is about handle-based
       currency selection; and do not paraphrase a literal Liquid \`variant\`
       case into a generic statement that loses the variant reference. If a
       sentence cannot be aligned to its cited claim without guessing, split the
       sentence or omit it. Do not cite several adjacent claims merely because
       they concern the same file or feature.
11L. MANDATORY VOLUME-PRICING EVIDENCE CHECKLIST: If \`snippets/price.liquid\`
     is supplied, create separate atomic claims for the product-card and
     product-page quantity-break checks when their source is present.
     (a) Product-card claim: evidence must include the \`if is_product_card\`
         branch and the exact \`assign has_volume_pricing =
         product_resource.quantity_price_breaks_configured?\` operation.
         In the claim prose, stop immediately after "product_resource's
         quantity-price-break configuration". Never append a dot, suffix, or
         stray word after the literal identifier ending in \`?\` (for example,
         do not produce \`quantity_price_breaks_configured?. molds\`).
     (b) Product-page claim: one exact evidence excerpt must include both the
         \`if selected_variant.quantity_price_breaks.size > 0\` condition and
         \`assign has_volume_pricing = true\` operation. Prefer to include the
         enclosing \`if is_product_card\` and its matching \`else\` in that excerpt.
         If the quote begins at \`else\`, backend verification must independently
         confirm from the same contiguous source segment that this is the
         \`else\` branch of \`if is_product_card\` and that the quoted assignment
         is inside it. Never infer product-page scope from the variable name
         or quantity-break expression alone.
11M. For the currency \`else\` branch, do not cite only \`else\` plus \`money\`.
     Include the paired \`if use_currency\` condition and the money assignment
     in one exact contiguous evidence excerpt. State explicitly that this is
     the \`use_currency\` else branch. For the true branch, spell the filter
     exactly as \`money_with_currency\`; never use a shortened or fused phrase.
11M-1. A plain assignment such as \`assign use_currency = false\` is a
       separate initialization fact, not the money-formatting else branch.
       Describe it as "initializes use_currency to false" and cite the exact
       assignment. Likewise, describe \`assign use_currency = true\` only
       with the exact setting/condition excerpt that governs that assignment.
       Do not attach \`money\` or \`money_with_currency\` claims to these
       boolean-setting assignment claims unless the same excerpt proves it.
11N. COMPARE-AT PRICE FLAG: Use the exact field name \`show_compare_price\`.
     Never output malformed names such as \`show_expr compare price\`.
     Prefer one atomic claim: "The file sets show_compare_price to true when
     compare_at_price is greater than price." Evidence must contain the exact
     \`if compare_at_price > price\` condition and \`assign show_compare_price = true\`.
11O. CURRENCY SETTING ASSIGNMENTS: For a setting-driven flag assignment, do not
     describe the same operation as a multi-action explanation. Use direct
     wording such as "The settings.currency_code_enabled_product_cards branch
     assigns use_currency to true." Cite the exact setting and assignment.
11P. LIQUID CASE LABELS: For \`{% case variant %}\`, state "uses a Liquid case on
     variant". For a branch label, state only that the file contains the literal
     \`when 'mobile'\` or \`when 'navigation_bar'\` label unless its exact branch and
     operation are both quoted.
11Q. PRODUCT-PAGE QUANTITY BREAK: Include both the exact
     \`if selected_variant.quantity_price_breaks.size > 0\` condition and
     \`assign has_volume_pricing = true\` in one evidence excerpt. You may start
     the excerpt at \`else\` only when the source context also contains the
     enclosing \`if is_product_card\`; backend verification will check that the
     assignment is actually inside that \`else\` branch. Describe the behavior
     as the product-page quantity-break check without broadening its condition.
11R. PRODUCT-CARD QUANTITY BREAK: Use one direct assignment claim, e.g. "For
     product cards, has_volume_pricing is assigned from the product resource's
     quantity-price-break configuration." The evidence must show the
     \`if is_product_card\` guard and the exact assignment.
11S. DEPENDENCY CLAIMS: A dependency edge may be claimed only when its caller
     file is named in DEPENDENCY CLAIM CALLER ALLOWLIST and the exact edge is
     supported by the supplied source excerpt. Do not infer that the caller's
     file contents were supplied merely because the file was recursively loaded.
11T. DUPLICATE EDGES: A caller-to-target render edge must appear only once in
     the answer. If it appears in the architecture explanation with a verified
     branch condition, do not repeat it in the dependency map. Otherwise list
     the generic edge only in the dependency map.
11U. DEFAULT LIQUID CASE BRANCH: Describe an operation as occurring in the
     default case branch only when the exact quoted operation is inside the
     \`{% else %}\` branch of the active \`{% case %}\`. A nearby \`render\` or script
     line alone does not prove the default branch. Keep generic dependency
     statements free of default-branch qualifiers unless that context is quoted.
11J. Claim prose must end with a complete, meaningful phrase. Do not append
     partial identifiers, fragments, unfinished words, or stray suffixes
     (for example, do not turn \`quantity_price_breaks_configured?\` into
     \`quantity_price_breaks_configured?.expl\`). Copy code only in evidence;
     describe the behavior in clean natural language.
11K. Never repeat the same caller-to-target render edge in more than one
     answer section. If the architecture section describes the render with a
     proven branch condition, omit that edge from the dependency map. Otherwise,
     omit the render from the architecture section and list the generic edge
     once in the dependency map. Do not add "default branch", "inside submenu",
     or similar qualifiers to a generic render edge.
12. Do not claim that a file was analyzed by Gemini if it was not included in its context.
13. Explain the relationship between settings, sections, blocks and snippets when relevant.
14. Do not modify any files.
15. Never claim that a file was changed or saved.
16. Clearly distinguish confirmed facts from reasonable explanations.
17. If the available source is insufficient, explicitly say so in the answer.
18. Analyze Shopify Horizon theme source code ONLY.

19. You do NOT have access to the Agent backend implementation.
Never claim that backend API routes, server-side functions,
internal architecture, retrieval systems, context budgets,
or application infrastructure have been implemented,
modified, or updated.

20. Never describe internal backend implementation as an
existing feature unless its source code is explicitly
provided in the context.

21. Only describe functionality supported by the actual
Shopify Horizon theme source code provided to you.

22. This task is about Shopify theme source only. Do not discuss backend
availability, retrieval behavior, context budgets, or internal API details.

23. Never invent files, functions, code changes,
or implementation details.
24. Do not include a factual source-code statement in the answer unless
    the same statement is represented in claims with exact supporting evidence.
25. Never cite a JavaScript/CSS/Liquid file name merely because it is common
    in Shopify themes; the exact file must appear in the supplied context.
26. If evidence does not support a statement, omit the statement or explicitly
    label it as unverified and explain what source file is missing.
27. Give every claim a unique sequential ID: C1, C2, C3, and so on.
28. Put each factual sentence or factual bullet on its own line.
29. Every factual sentence or bullet MUST end with EXACTLY ONE claim marker,
    for example: [[C1]]. One sentence = one claim ID = one independently verifiable fact.
30. Put each claim ID in its own marker. NEVER group IDs as [[C1, C2]],
    and NEVER use ranges such as [[C1 to C5]].
30A. CRITICAL: Never attach two or more different claim IDs to the same factual
     sentence or bullet. Do not write one combined sentence about home, product,
     and collection templates with [[C1]][[C2]][[C3]]. Write three separate
     sentences, each with its own single marker. Apply the same rule to
     always/scroll-up, product-card/product-page, and logo/menu facts.
30B. Do not combine "captures" with "renders" unless the claim evidence proves
     both operations. A claim saying "captures logo" does not support a sentence
     saying the logo is both captured and rendered.
31. Every marker must refer to an ID in the claims array, and the referenced
    single atomic claim must support that exact sentence or bullet. Use exactly
    one claim marker per factual sentence. Never reuse a broad claim to cover
    multiple independent facts. Do not attach a second marker unless that claim
    independently supports the full sentence; being from the same file or
    appearing in the same section is not enough.
32. Do not put multiple independent facts in one sentence or bullet.
32A. CLAIM-TO-SENTENCE ALIGNMENT: Keep each answer sentence close to the exact
     wording of its cited claim. Reuse the same concrete subject, target, action,
     identifiers, and meaningful qualifiers. Do not cite a claim merely because
     it discusses the same file or topic. Every cited claim must independently
     support the complete sentence; remove extra claim IDs rather than grouping
     claims that support only separate parts. When wording differs, rewrite the
     sentence to match the verified claim instead of broadening the claim.
32B. EXACT CLAIM WORDING: When a dependency relationship or source finding cannot
     be described without adding qualifiers, use the exact wording of the atomic
     claim as a standalone sentence and cite only that claim's ID.
32C. SOURCE-FILE ATTRIBUTION: If a claim names a theme file as the subject of an
     action (for example, "sections/header.liquid renders ..."), its file field must
     be that exact source file. Do not attribute behavior to a caller or target file
     whose source does not contain the quoted operation.
33. Do not put unreferenced factual statements in tables, captions, blockquotes,
    introductions, conclusions, or paragraphs.
34. Headings and non-factual structural labels do not need claim markers.
35. Do not use claim markers for claims whose evidence is absent from the
    exact source context sent to you.
36. Use SOURCE COVERAGE STATUS as the authority for completeness. Never describe
    a file as complete/full unless its complete field is true. A verified code
    excerpt proves only that excerpt, not that the whole file was supplied.
37. SOURCE COVERAGE STATUS is backend-supplied metadata, not theme source code.
    Do not create claims whose evidence is hasMore, isComplete, coveragePercent,
    retrievedRanges, or other manifest metadata. The backend appends the trusted
    coverage summary after your answer.
38. For a render/dependency statement, put the claim file as the caller file
    containing the render statement. The dependency target may be named in the
    answer, but do not claim the target's internal behavior unless its own source
    was supplied and the evidence comes from that target file.
39. Never use a claim ID merely because it exists. Each cited claim must support
    the exact sentence and must come from the file where its evidence appears.
40. Do not add a sentence about the Agent backend being unavailable; it is outside
    the requested Shopify theme analysis.
41. Do not repeat the same source behavior as multiple claims merely because it
    appears in more than one answer section. A single source behavior should have
    one claim ID and one factual statement. The dependency map should list only
    distinct caller-to-target edges not already stated elsewhere.
42. For dependency-map statements, use the narrow relationship wording
    "<caller file> renders <target file>" unless the exact contiguous evidence
    also proves the additional qualifier. Do not add "standard menu", "inside
    submenu", "mobile-only", "default branch", or similar context from a
    render line that only proves the generic render relationship.
43. Never cite an incomplete/truncated excerpt as proof of a conditional or
    positional behavior. If a quote ends at an unfinished render tag or in the
    middle of an argument, either provide a complete exact excerpt with the
    relevant condition/container, or state only the generic dependency edge
    when that edge is independently verified.
44. Claims must describe concrete behavior or relationships present in the
    Shopify theme source. Do not create claims about analysis methodology,
    evidence verification, retrieved context, source limitations, or what the
    Agent does; source-coverage limitations are appended by the backend.
45. Do NOT write a "Missing or Incomplete Source Coverage", "Source Coverage",
    "Verified Findings and Limitations", or similar section. Do NOT list which
    files are complete/partial, describe percentages/ranges, or add closing
    statements saying the answer is verified/traceable/supported by source.
    The backend appends the authoritative coverage section after your answer.
46. DEPENDENCY MAP CITATIONS: Every dependency-map edge must have its own
    atomic claim object and its matching [[C#]] marker in the corresponding
    answer bullet. Never list uncited caller-to-target paths. If you cannot
    provide a verified claim and matching citation for every edge, omit the
    entire dependency-map section; do not fall back to uncited bullets.
47. LITERAL MOBILE LABEL: If the only evidence is \`{% when 'mobile' %}\`, use
    wording such as "contains the literal when 'mobile' case label". Do not
    describe mobile behavior, rendering, or desktop/mobile layout from that
    label alone. For a mobile header-drawer render claim, quote the when tag
    and the render statement in one contiguous excerpt.
48. TRANSPARENT HEADER: For home, product, or collection, quote the exact
    matching setting/template guard and \`assign transparent = 'always'\`.
    Mention the text-color setting only when the sentence also states that
    behavior and the matching color assignment is in the same quote.
49. MENU RENDER QUALIFIERS: A quote containing only \`render 'overflow-list'\`
    or \`render 'mega-menu-list'\` supports only the generic caller-to-target
    render edge. Do not call it desktop/default or place it inside a submenu
    unless the same contiguous evidence proves that exact context.
50. CURRENCY ELSE EVIDENCE: Do not create a \`use_currency\` else-branch claim
    unless one evidence item begins at \`if use_currency\` and includes its
    paired \`else\` and the \`| money\` assignment. An \`else\`-only quote is not
    sufficient; omit the claim when the complete quote is unavailable.
51. PRODUCT-PAGE QUANTITY BREAK: Do not create a product-page volume-pricing
    claim unless one quote starts at the outer \`if is_product_card\` and reaches
    its paired \`else\`, nested quantity-break check, assignment, and closing
    \`endif\` statements. An \`else\`-only quote is insufficient; omit the claim.
The answer field should contain a practical explanation for the user.

The claims array should contain important factual claims
that can be checked against actual source files.

If no source evidence is available, return an empty claims array.
`,
        },
      });

    // -------------------------------------
    // STEP 14: Parse Gemini response
    // -------------------------------------

    const responseText =
      response.text?.trim() || "";

    console.log(
      "\n========== RAW FINAL GEMINI RESPONSE ==========\n",
      responseText
    );

    let parsedResponse: unknown;

    try {
      parsedResponse = parseGeminiJson(responseText);

      if (
        typeof parsedResponse !== "object" ||
        parsedResponse === null ||
        Array.isArray(parsedResponse)
      ) {
        throw new Error(
          "Gemini returned an invalid JSON object."
        );
      }
    } catch (error) {
      console.error(
        "GEMINI JSON PARSE ERROR:",
        error
      );

      return NextResponse.json(
        {
          success: false,
          error:
            "Gemini returned an invalid JSON response.",
          details:
            error instanceof Error
              ? error.message
              : "Unknown JSON parsing error.",
        },
        { status: 502 }
      );
    }

    const parsedObject =
      parsedResponse as Record<string, unknown>;

    const answer =
      typeof parsedObject.answer === "string" &&
        parsedObject.answer.trim().length > 0
        ? parsedObject.answer
        : "No response generated.";

    // -------------------------------------
    // STEP 15: Validate claims structure
    // -------------------------------------

    const seenAnswerClaimIds = new Set<string>();

    const parsedAnswerClaims: AnswerClaim[] =
      Array.isArray(parsedObject.claims)
        ? parsedObject.claims.filter(
          (claim: unknown): claim is AnswerClaim => {
            if (
              typeof claim !== "object" ||
              claim === null ||
              Array.isArray(claim)
            ) {
              return false;
            }

            const item =
              claim as Record<string, unknown>;

            if (
              typeof item.id !== "string" ||
              !/^C[0-9]+$/.test(item.id) ||
              typeof item.claim !== "string" ||
              typeof item.file !== "string" ||
              !(
                typeof item.evidence === "string" ||
                (
                  Array.isArray(item.evidence) &&
                  item.evidence.length > 0 &&
                  item.evidence.every((entry) => typeof entry === "string")
                )
              ) ||
              seenAnswerClaimIds.has(item.id)
            ) {
              return false;
            }

            seenAnswerClaimIds.add(item.id);
            return true;
          }
        )
        : [];

    const backendMetadataClaimIds = new Set(
      parsedAnswerClaims
        .filter(isBackendCoverageMetadataClaim)
        .map((claim) => claim.id)
    );

    const answerClaimsWithoutMetadata = parsedAnswerClaims.filter(
      (claim) => !backendMetadataClaimIds.has(claim.id)
    );
    const outOfScopeModelClaims = explicitOnly && explicitFileAllowlist
      ? answerClaimsWithoutMetadata.filter((claim) => !isPathAllowedByScope(claim.file, explicitFileAllowlist))
      : [];
    if (outOfScopeModelClaims.length > 0) {
      console.warn("EXPLICIT FILE SCOPE: rejecting out-of-scope model claims:", outOfScopeModelClaims.map((claim) => ({ id: claim.id, file: claim.file })));
    }
    const answerClaims = explicitOnly && explicitFileAllowlist
      ? answerClaimsWithoutMetadata.filter((claim) => isPathAllowedByScope(claim.file, explicitFileAllowlist))
      : answerClaimsWithoutMetadata;

    if (backendMetadataClaimIds.size > 0) {
      console.log("IGNORED BACKEND-ONLY COVERAGE CLAIM IDS:", [
        ...backendMetadataClaimIds,
      ]);
    }

    // -------------------------------------
    // STEP 16: Verify answer claims
    // -------------------------------------

    const includedFilesSet = new Set(
      filesIncludedInContext
    );

    const verifiedAnswerClaims: VerifiedAnswerClaim[] = [];

    for (const claim of answerClaims) {


      const verification = await verifyAnswerClaim(
        claim,
        availableThemeFiles,
        includedFilesSet,
        includedSourceContent,
        verifiedDependencies,
        explicitFileAllowlist
      );

      verifiedAnswerClaims.push(verification);
    }

    const verifiedClaimCount =
      verifiedAnswerClaims.filter(
        (claim) => claim.verified
      ).length;

    const unverifiedClaimCount =
      verifiedAnswerClaims.length - verifiedClaimCount;

    const answerClaimCoverage = enforceAnswerClaimReferences(
      answer,
      verifiedAnswerClaims,
      sourceCoverageManifest,
      verifiedDependencies,
      backendMetadataClaimIds
    );

    const finalContentAnswer = answerClaimCoverage.answer;
    const sourceCoverageMarkdown = buildSourceCoverageMarkdown(
      sourceCoverageManifest
    );
    const knownThemePaths = new Set<string>([
      ...availableThemeFiles,
      ...verifiedDependencies.flatMap((dependency) => [dependency.from, dependency.to]),
    ]);
    const referencedSourcePaths = extractThemeFileReferences(finalContentAnswer, knownThemePaths);
    const originalReferencedSourcePaths = extractThemeFileReferences(answer, knownThemePaths);
    const usedClaimIds = new Set(answerClaimCoverage.usedClaimIds);
    const usedVerifiedClaims = verifiedAnswerClaims.filter(
      (claim) => claim.verified && usedClaimIds.has(claim.id)
    );
    const verifiedDependencyReferenceFiles = new Set(
      answerClaimCoverage.verifiedDependencyReferenceFiles.map(normalizePath)
    );

    const sourceReferenceAudit = referencedSourcePaths.map((file) => {
      const exists = availableThemeFiles.has(file);
      const included = includedFilesSet.has(file);
      const hasVerifiedClaim = usedVerifiedClaims.some(
        (claim) => normalizePath(claim.file) === file
      );
      const hasVerifiedDependencyEdge =
        verifiedDependencyReferenceFiles.has(normalizePath(file));

      let status: string = "verified";
      let reason = "At least one emitted claim for this path passed evidence verification.";

      if (!exists) {
        status = "file_not_found";
        reason = "The referenced path does not exist in the scanned theme file list.";
      } else if (hasVerifiedClaim) {
        status = "verified";
        reason = "A claim supported by this file's own source passed evidence verification.";
      } else if (hasVerifiedDependencyEdge) {
        status = "verified_dependency_reference";
        reason = included
          ? "The caller's source proves this dependency edge. The target's internal behavior is not independently verified unless supported by a claim from the target file."
          : "The caller's source proves this dependency edge, but the target's source was not included in Gemini context; its internal behavior is not verified.";
      } else if (!included) {
        status = "not_in_gemini_context";
        reason = "The path exists, but its source was not included in Gemini context and no cited verified dependency edge supports the reference.";
      } else {
        status = "no_verified_claim";
        reason = "The path was mentioned in the answer, but no claim from this file or verified caller-to-target edge supports it.";
      }

      return { file, status, reason };
    });

    const unverifiedSourceReferences = sourceReferenceAudit.filter(
      (item) => item.status !== "verified" &&
        item.status !== "verified_dependency_reference"
    );

    const dependencyOnlySourceReferences = sourceReferenceAudit.filter(
      (item) => item.status === "verified_dependency_reference"
    );

    const verificationNoticeParts: string[] = [];

    if (unverifiedClaimCount > 0) {
      verificationNoticeParts.push(
        `${unverifiedClaimCount} of ${verifiedAnswerClaims.length} source claims did not pass evidence verification. Treat those claims as unverified.`
      );
    }

    if (unverifiedSourceReferences.length > 0) {
      verificationNoticeParts.push(
        `The following file references in the answer could not be verified: ${unverifiedSourceReferences
          .map((item) => `\`${item.file}\` (${item.reason})`)
          .join("; ")}. Treat any related explanation as unverified.`
      );
    }

    if (dependencyOnlySourceReferences.length > 0) {
      verificationNoticeParts.push(
        `These dependency references are verified from their caller files, but their target implementations are not independently verified: ${dependencyOnlySourceReferences
          .map((item) => `\`${item.file}\``)
          .join(", ")}.`
      );
    }

    if (answerClaimCoverage.ignoredMetadataClaimIds.length > 0) {
      verificationNoticeParts.push(
        `Backend source-coverage metadata claims were ignored because coverage is appended from the retrieval manifest: ${answerClaimCoverage.ignoredMetadataClaimIds.join(", ")}.`
      );
    }

    const uncitedRemovedLines = Math.max(
      0,
      answerClaimCoverage.removedLines - answerClaimCoverage.suppressedModelSectionLines
    );
    if (uncitedRemovedLines > 0) {
      verificationNoticeParts.push(
        `${uncitedRemovedLines} answer line(s) were removed because they did not reference a source claim that passed evidence verification.`
      );
    }
    if (answerClaimCoverage.suppressedModelSectionLines > 0) {
      verificationNoticeParts.push(
        `${answerClaimCoverage.suppressedModelSectionLines} line(s) in model-generated coverage or uncited dependency-map sections were suppressed; the backend supplies the authoritative coverage summary.`
      );
    }

    if (answerClaimCoverage.invalidClaimReferences.length > 0) {
      verificationNoticeParts.push(
        `Invalid or unverified claim markers were found: ${answerClaimCoverage.invalidClaimReferences.join(", ")}.`
      );
    }

    const answerVerificationNotice = verificationNoticeParts.length > 0
      ? `\n\n**Source verification notice:** ${verificationNoticeParts.join(" ")}`
      : "";

    const finalAnswer = `${finalContentAnswer}${sourceCoverageMarkdown}${answerVerificationNotice}`;

    console.log(
      "\n========== ANSWER EVIDENCE VERIFICATION =========="
    );

    console.table({
      totalClaims: verifiedAnswerClaims.length,
      verifiedClaims: verifiedClaimCount,
      unverifiedClaims: unverifiedClaimCount,
    });

    console.table(
      verifiedAnswerClaims.map((claim) => ({
        claim: claim.claim,
        file: claim.file,
        verified: claim.verified,
        reason: claim.verificationReason,
      }))
    );

    console.log("ANSWER SOURCE PATH AUDIT:", sourceReferenceAudit);
    console.log("ANSWER CLAIM-LINE TRACEABILITY:", {
      originalReferencedSourcePaths,
      finalReferencedSourcePaths: referencedSourcePaths,
      retainedLines: answerClaimCoverage.retainedLines,
      removedLines: answerClaimCoverage.removedLines,
      coverageBlockedLines: answerClaimCoverage.coverageBlockedLines,
      pathMismatchLines: answerClaimCoverage.pathMismatchLines,
      malformedMarkerLines: answerClaimCoverage.malformedMarkerLines,
      uncitedStatementLines: answerClaimCoverage.uncitedStatementLines,
      suppressedModelSectionLines: answerClaimCoverage.suppressedModelSectionLines,
      unrelatedClaimLines: answerClaimCoverage.unrelatedClaimLines,
      fallbackClaimLinesRetained: answerClaimCoverage.fallbackClaimLinesRetained,
      fallbackClaimLinesRejected: answerClaimCoverage.fallbackClaimLinesRejected,
      unrelatedClaimDiagnostics: answerClaimCoverage.unrelatedClaimDiagnostics,
      pathMismatchDiagnostics: answerClaimCoverage.pathMismatchDiagnostics,
      usedClaimIds: answerClaimCoverage.usedClaimIds,
      verifiedDependencyReferenceFiles: answerClaimCoverage.verifiedDependencyReferenceFiles,
      invalidClaimReferences: answerClaimCoverage.invalidClaimReferences,
      ignoredMetadataClaimIds: answerClaimCoverage.ignoredMetadataClaimIds,
      usedVerifiedClaimFallback: answerClaimCoverage.usedVerifiedClaimFallback,
    });

    const finalVerifiedCount =
      verifiedDependencies.filter(
        (dependency) => dependency.verified
      ).length;


    // -------------------------------------
    // FINAL RESPONSE
    // -------------------------------------

    return NextResponse.json({
      success: true,
      explicitOnly,
      explicitFileAllowlist: explicitOnly ? [...(explicitFileAllowlist ?? [])] : [],

      message: finalAnswer,

      claims: verifiedAnswerClaims,

      evidenceSummary: {
        total: verifiedAnswerClaims.length,
        verified: verifiedClaimCount,
        unverified: unverifiedClaimCount,
      },

      answerVerification: {
        explicitSourcePaths: referencedSourcePaths.length,
        verifiedSourcePaths: sourceReferenceAudit.filter(
          (item) => item.status === "verified"
        ).length,
        unverifiedSourcePaths: unverifiedSourceReferences,
        sourcePathCoverageComplete: unverifiedSourceReferences.length === 0,
        claimLineTraceability: {
          retainedLines: answerClaimCoverage.retainedLines,
          removedLines: answerClaimCoverage.removedLines,
          coverageBlockedLines: answerClaimCoverage.coverageBlockedLines,
          pathMismatchLines: answerClaimCoverage.pathMismatchLines,
          malformedMarkerLines: answerClaimCoverage.malformedMarkerLines,
          uncitedStatementLines: answerClaimCoverage.uncitedStatementLines,
          suppressedModelSectionLines: answerClaimCoverage.suppressedModelSectionLines,
          unrelatedClaimLines: answerClaimCoverage.unrelatedClaimLines,
          fallbackClaimLinesRetained: answerClaimCoverage.fallbackClaimLinesRetained,
          fallbackClaimLinesRejected: answerClaimCoverage.fallbackClaimLinesRejected,
          unrelatedClaimDiagnostics: answerClaimCoverage.unrelatedClaimDiagnostics,
          pathMismatchDiagnostics: answerClaimCoverage.pathMismatchDiagnostics,
          usedClaimIds: answerClaimCoverage.usedClaimIds,
          invalidClaimReferences: answerClaimCoverage.invalidClaimReferences,
          ignoredMetadataClaimIds: answerClaimCoverage.ignoredMetadataClaimIds,
          usedVerifiedClaimFallback: answerClaimCoverage.usedVerifiedClaimFallback,
        },
        directlyVerifiedSourcePaths: sourceReferenceAudit.filter(
          (item) => item.status === "verified"
        ).length,
        dependencyVerifiedSourcePaths: dependencyOnlySourceReferences.length,
        note: "Claim evidence is checked against exact source ranges, actual files, bounded same-range Liquid control-flow context, structural assignment/branch patterns, and term relevance. A dependency reference is verified from the caller file, but this does not verify the target file's internal behavior. This is not a complete semantic entailment proof.",
      },

      retrievalCoverage,

      filesAnalyzed,

      filesIncludedInContext,

      filesSkippedFromContext,

      dependencyGraph: verifiedDependencies,

      relevantDependencies,

      verificationSummary: {
        total: verifiedDependencies.length,
        verified: finalVerifiedCount,
        unverified:
          verifiedDependencies.length - finalVerifiedCount,
        relevant: relevantDependencies.length,
      },


    });
  } catch (error) {
    console.error(
      "HORIZON AGENT ERROR:",
      error
    );

    const status = getGeminiErrorStatus(error);

    if (
      status === 429 ||
      status === 503 ||
      status === 504
    ) {
      return NextResponse.json(
        {
          success: false,
          error:
            "Gemini service is temporarily unavailable. Please try again shortly.",
          details:
            error instanceof Error
              ? error.message
              : "Gemini temporary service error.",
        },
        { status: status === 429 ? 429 : 503 }
      );
    }

    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : "Something went wrong",
      },
      { status: 500 }
    );
  }
}
