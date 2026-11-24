import { createHash } from "node:crypto";
import { Buffer } from "node:buffer";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";

export type FigmaDesignAsset = {
  fileName: string;
  nodeId: string;
  nodeName: string;
  imageRef: string;
  mimeType: string;
  width: number;
  height: number;
  imageData: string;
};

export type FigmaDesignContext = {
  fileName: string;
  fileKey: string;
  nodeId: string;
  nodeName: string;
  summary: string;
  imageData: string;
  imageMimeType: string;
  assets: FigmaDesignAsset[];
};

export class FigmaConnectionError extends Error {
  status: number;
  retryAfterSeconds?: number;
  planTier?: string;
  rateLimitType?: string;
  upgradeLink?: string;

  constructor(
    message: string,
    status = 502,
    details?: {
      retryAfterSeconds?: number;
      planTier?: string;
      rateLimitType?: string;
      upgradeLink?: string;
    }
  ) {
    super(message);
    this.name = "FigmaConnectionError";
    this.status = status;
    this.retryAfterSeconds = details?.retryAfterSeconds;
    this.planTier = details?.planTier;
    this.rateLimitType = details?.rateLimitType;
    this.upgradeLink = details?.upgradeLink;
  }
}

type FigmaNode = {
  id?: string;
  name?: string;
  type?: string;
  characters?: string;
  style?: {
    fontFamily?: string;
    fontSize?: number;
    fontWeight?: number | string;
    lineHeightPx?: number;
    letterSpacing?: number;
  };
  size?: { x?: number; y?: number };
  absoluteBoundingBox?: { width?: number; height?: number };
  layoutMode?: string;
  itemSpacing?: number;
  paddingTop?: number;
  paddingRight?: number;
  paddingBottom?: number;
  paddingLeft?: number;
  primaryAxisAlignItems?: string;
  counterAxisAlignItems?: string;
  cornerRadius?: number;
  fills?: Array<Record<string, any>>;
  strokes?: Array<Record<string, any>>;
  children?: FigmaNode[];
};

type ParsedFigmaUrl = {
  fileKey: string;
  nodeId: string | null;
};

type FigmaCacheRecord = {
  version: 2;
  savedAt: number;
  context: FigmaDesignContext;
};

const FIGMA_CACHE_DIRECTORY = path.join(process.cwd(), ".horizon-cache", "figma");
const figmaContextMemoryCache = new Map<string, FigmaDesignContext>();
const figmaContextLoadsInFlight = new Map<string, Promise<FigmaDesignContext>>();

function getFigmaCacheKey(fileKey: string, nodeId: string | null): string {
  return fileKey + "-" + (nodeId ? nodeId.replace(/:/g, "-") : "auto");
}

function getFigmaCachePath(fileKey: string, nodeId: string | null): string {
  return path.join(FIGMA_CACHE_DIRECTORY, getFigmaCacheKey(fileKey, nodeId) + ".json");
}

function isFigmaDesignContext(value: unknown): value is FigmaDesignContext {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return typeof item.fileName === "string" &&
    typeof item.fileKey === "string" &&
    typeof item.nodeId === "string" &&
    typeof item.nodeName === "string" &&
    typeof item.summary === "string" &&
    typeof item.imageData === "string" &&
    typeof item.imageMimeType === "string" &&
    Array.isArray(item.assets) &&
    item.assets.every((asset) => {
      if (!asset || typeof asset !== "object") return false;
      const entry = asset as Record<string, unknown>;
      return typeof entry.fileName === "string" &&
        typeof entry.nodeId === "string" &&
        typeof entry.nodeName === "string" &&
        typeof entry.imageRef === "string" &&
        typeof entry.mimeType === "string" &&
        typeof entry.width === "number" &&
        typeof entry.height === "number" &&
        typeof entry.imageData === "string";
    });
}

async function readCachedFigmaDesignContext(
  fileKey: string,
  nodeId: string | null
): Promise<FigmaDesignContext | null> {
  const key = getFigmaCacheKey(fileKey, nodeId);
  const inMemory = figmaContextMemoryCache.get(key);
  if (inMemory) return inMemory;

  try {
    const raw = await readFile(getFigmaCachePath(fileKey, nodeId), "utf8");
    const record = JSON.parse(raw) as FigmaCacheRecord;
    if (record.version !== 2 || !isFigmaDesignContext(record.context)) return null;
    figmaContextMemoryCache.set(key, record.context);
    return record.context;
  } catch (error) {
    const code = error && typeof error === "object"
      ? (error as NodeJS.ErrnoException).code
      : undefined;
    if (code !== "ENOENT") {
      console.warn("Could not read the local Figma design cache:", error);
    }
    return null;
  }
}

async function writeCachedFigmaDesignContext(
  fileKey: string,
  nodeId: string | null,
  context: FigmaDesignContext
): Promise<void> {
  await mkdir(FIGMA_CACHE_DIRECTORY, { recursive: true });
  const cachePath = getFigmaCachePath(fileKey, nodeId);
  const temporaryPath = cachePath + "." + process.pid + "." + Date.now() + ".tmp";
  const record: FigmaCacheRecord = { version: 2, savedAt: Date.now(), context };
  await writeFile(temporaryPath, JSON.stringify(record), "utf8");
  await rename(temporaryPath, cachePath);
}

function getFigmaToken(): string {
  const token = process.env.FIGMA_ACCESS_TOKEN?.trim();
  if (!token) {
    throw new FigmaConnectionError(
      "Figma Agent connection is not configured. Add FIGMA_ACCESS_TOKEN to .env.local using a Figma personal access token with the file_content:read scope, then restart Horizon AI.",
      409
    );
  }
  return token;
}

function parseFigmaDesignUrl(value: string): ParsedFigmaUrl {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new FigmaConnectionError("The saved Figma URL is not valid. Reload the Figma Design link in the dashboard.", 400);
  }

  const hostname = url.hostname.toLowerCase();
  const isFigmaHost = hostname === "figma.com" || hostname.endsWith(".figma.com");
  const match = url.pathname.match(/^\/(?:design|file)\/([a-z0-9]{22,128})(?:\/|$)/i);
  if (url.protocol !== "https:" || !isFigmaHost || !match || url.username || url.password || url.port) {
    throw new FigmaConnectionError("Agent design reading currently supports Figma Design share links only. Paste a https://www.figma.com/design/... link.", 400);
  }

  const rawNodeId = url.searchParams.get("node-id");
  const nodeId = rawNodeId ? rawNodeId.replace(/-/g, ":") : null;
  if (nodeId && !/^\d+:\d+$/.test(nodeId)) {
    throw new FigmaConnectionError("The Figma link contains an invalid node-id. Copy the link directly from the homepage frame in Figma.", 400);
  }

  return { fileKey: match[1], nodeId };
}

async function figmaJson<T>(endpoint: string): Promise<T> {
  const response = await fetch("https://api.figma.com/v1" + endpoint, {
    method: "GET",
    headers: {
      "X-Figma-Token": getFigmaToken(),
      Accept: "application/json",
    },
    cache: "no-store",
    signal: AbortSignal.timeout(20000),
  });

  const payload = await response.json().catch(() => ({})) as Record<string, any>;
  if (!response.ok) {
    const detail = typeof payload.err === "string" ? payload.err : "";
    if (response.status === 401) {
      throw new FigmaConnectionError(
        "Figma rejected the API token. Check that FIGMA_ACCESS_TOKEN is a valid, non-expired token." +
          (detail ? " Figma: " + detail : ""),
        403
      );
    }
    if (response.status === 403) {
      throw new FigmaConnectionError(
        "Figma denied access to this file. Confirm the token includes file_content:read and its Figma account can view the file." +
          (detail ? " Figma: " + detail : ""),
        403
      );
    }
    if (response.status === 404) {
      throw new FigmaConnectionError("Figma could not find this file. Check the share URL and confirm the token's account has access to it.", 404);
    }
    if (response.status === 429) {
      const retryAfterHeader = response.headers.get("retry-after");
      const retryAfterValue = retryAfterHeader ? Number(retryAfterHeader) : NaN;
      const retryAfterSeconds = Number.isFinite(retryAfterValue) && retryAfterValue > 0
        ? Math.ceil(retryAfterValue)
        : undefined;
      const planTier = response.headers.get("x-figma-plan-tier") || undefined;
      const rateLimitType = response.headers.get("x-figma-rate-limit-type") || undefined;
      const upgradeLink = response.headers.get("x-figma-upgrade-link") || undefined;
      const details = [
        "Figma REST API rate limit reached.",
        retryAfterSeconds
          ? "Figma says to retry after " + retryAfterSeconds + " seconds."
          : "Figma did not provide a short retry delay; this may be a monthly seat/plan limit.",
        planTier ? "File plan: " + planTier + "." : "",
        rateLimitType === "low"
          ? "Figma reports a low-rate seat class (View/Collab)."
          : rateLimitType === "high"
            ? "Figma reports a high-rate seat class (Dev/Full)."
            : "",
        upgradeLink ? "Review Figma seat/plan limits: " + upgradeLink : "",
        "Horizon AI will cache a successful design read locally so later edits do not repeatedly call the Figma API.",
      ].filter(Boolean).join(" ");
      throw new FigmaConnectionError(details, 429, {
        retryAfterSeconds,
        planTier,
        rateLimitType,
        upgradeLink,
      });
    }
    throw new FigmaConnectionError(
      "Figma API request failed with HTTP " + response.status + (detail ? ": " + detail : "."),
      response.status >= 400 && response.status < 500 ? response.status : 502
    );
  }

  return payload as T;
}

function nodeWidth(node: FigmaNode): number {
  return node.absoluteBoundingBox?.width ?? node.size?.x ?? 0;
}

function nodeHeight(node: FigmaNode): number {
  return node.absoluteBoundingBox?.height ?? node.size?.y ?? 0;
}

function chooseHomepageFrame(document: FigmaNode): FigmaNode | null {
  const candidates: Array<{ node: FigmaNode; score: number }> = [];
  const visit = (node: FigmaNode, pageName: string, depth: number) => {
    if (depth > 3) return;
    const type = String(node.type || "");
    const name = String(node.name || "");
    const lowerName = name.toLowerCase();
    const width = nodeWidth(node);
    const height = nodeHeight(node);
    if (["FRAME", "COMPONENT", "INSTANCE", "SECTION"].includes(type) && width >= 560 && height >= 350) {
      let score = Math.min(width, 2200) / 30 + Math.min(height, 8000) / 300;
      if (/\b(homepage|home page|home desktop|desktop home|storefront home)\b/.test(lowerName)) score += 240;
      else if (/\b(home|landing|main page|ecommerce|e-commerce|shop home)\b/.test(lowerName)) score += 130;
      if (/\b(desktop|web)\b/.test(lowerName)) score += 35;
      if (/\b(mobile|tablet|product card|checkout|cart|footer|header)\b/.test(lowerName)) score -= 60;
      if (/\b(home|homepage|storefront)\b/.test(pageName.toLowerCase())) score += 25;
      candidates.push({ node, score });
    }
    for (const child of node.children ?? []) visit(child, pageName, depth + 1);
  };

  for (const page of document.children ?? []) {
    if (page.type === "PAGE") {
      for (const child of page.children ?? []) visit(child, String(page.name || ""), 0);
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  return candidates[0]?.node ?? null;
}

function colorToHex(color: Record<string, any>): string {
  const toByte = (value: unknown) => {
    const number = typeof value === "number" ? value : 0;
    return Math.max(0, Math.min(255, Math.round(number * 255))).toString(16).padStart(2, "0");
  };
  return "#" + toByte(color.r) + toByte(color.g) + toByte(color.b);
}

function summarizeNodeTree(root: FigmaNode): string {
  const lines: string[] = [];
  let visited = 0;

  const visit = (node: FigmaNode, depth: number) => {
    if (visited >= 650 || depth > 8) return;
    visited += 1;
    const indent = "  ".repeat(depth);
    const width = Math.round(nodeWidth(node));
    const height = Math.round(nodeHeight(node));
    const details: string[] = [];
    if (width || height) details.push(width + "x" + height + "px");
    if (node.layoutMode) details.push("layout=" + node.layoutMode);
    if (typeof node.itemSpacing === "number") details.push("gap=" + node.itemSpacing);
    if (node.paddingTop || node.paddingRight || node.paddingBottom || node.paddingLeft) {
      details.push("padding=" + [node.paddingTop ?? 0, node.paddingRight ?? 0, node.paddingBottom ?? 0, node.paddingLeft ?? 0].join("/"));
    }
    if (node.primaryAxisAlignItems) details.push("mainAlign=" + node.primaryAxisAlignItems);
    if (node.counterAxisAlignItems) details.push("crossAlign=" + node.counterAxisAlignItems);
    if (typeof node.cornerRadius === "number") details.push("radius=" + node.cornerRadius);

    const solidFills = (node.fills ?? [])
      .filter((fill) => fill.visible !== false && fill.type === "SOLID" && fill.color)
      .slice(0, 3)
      .map((fill) => colorToHex(fill.color) + (typeof fill.opacity === "number" && fill.opacity < 1 ? "@" + fill.opacity.toFixed(2) : ""));
    if (solidFills.length) details.push("fill=" + solidFills.join(","));

    if (node.type === "TEXT") {
      const text = String(node.characters || "").replace(/\s+/g, " ").trim().slice(0, 220);
      if (text) details.push("text=" + JSON.stringify(text));
      const style = node.style ?? {};
      const typography = [
        style.fontFamily ? "font=" + style.fontFamily : "",
        style.fontSize ? "size=" + style.fontSize : "",
        style.fontWeight ? "weight=" + style.fontWeight : "",
        style.lineHeightPx ? "lineHeight=" + style.lineHeightPx : "",
        style.letterSpacing ? "letterSpacing=" + style.letterSpacing : "",
      ].filter(Boolean);
      if (typography.length) details.push(typography.join(" "));
    }

    lines.push(indent + "- " + (node.name || node.type || "Layer") + " [" + (node.type || "UNKNOWN") + "]" + (details.length ? " — " + details.join("; ") : ""));
    for (const child of node.children ?? []) visit(child, depth + 1);
  };

  visit(root, 0);
  if (visited >= 650) lines.push("... design hierarchy shortened after 650 nodes ...");
  return lines.join("\n").slice(0, 26000);
}

type FigmaImageUse = {
  nodeId: string;
  nodeName: string;
  imageRef: string;
  width: number;
  height: number;
};

function findNamedFigmaNode(root: FigmaNode, targetName: string): FigmaNode | null {
  const target = targetName.trim().toLowerCase();
  const stack: FigmaNode[] = [root];
  let visited = 0;
  while (stack.length > 0 && visited < 12000) {
    const node = stack.pop()!;
    visited += 1;
    if (String(node.name || "").trim().toLowerCase() === target) return node;
    for (const child of node.children ?? []) stack.push(child);
  }
  return null;
}

function collectFigmaImageUses(root: FigmaNode): FigmaImageUse[] {
  const byImageRef = new Map<string, FigmaImageUse>();
  const stack: FigmaNode[] = [root];
  let visited = 0;

  while (stack.length > 0 && visited < 12000) {
    const node = stack.pop()!;
    visited += 1;
    for (const fill of node.fills ?? []) {
      if (
        fill.type !== "IMAGE" ||
        typeof fill.imageRef !== "string" ||
        !fill.imageRef ||
        fill.visible === false ||
        (typeof fill.opacity === "number" && fill.opacity <= 0)
      ) {
        continue;
      }

      if (!node.id || byImageRef.has(fill.imageRef)) continue;
      byImageRef.set(fill.imageRef, {
        nodeId: node.id,
        nodeName: node.name || "Figma image",
        imageRef: fill.imageRef,
        width: Math.round(nodeWidth(node)),
        height: Math.round(nodeHeight(node)),
      });
    }

    for (const child of node.children ?? []) stack.push(child);
  }

  return [...byImageRef.values()];
}

function getImageExtension(mimeType: string, imageUrl: string): string | null {
  const mime = mimeType.toLowerCase().split(";")[0].trim();
  if (mime === "image/png") return "png";
  if (mime === "image/jpeg" || mime === "image/jpg") return "jpg";
  if (mime === "image/webp") return "webp";
  if (mime === "image/gif") return "gif";
  if (mime === "image/svg+xml") return "svg";

  try {
    const extension = path.extname(new URL(imageUrl).pathname).toLowerCase().replace(".", "");
    return ["png", "jpg", "jpeg", "webp", "gif", "svg"].includes(extension)
      ? (extension === "jpeg" ? "jpg" : extension)
      : null;
  } catch {
    return null;
  }
}

async function downloadFigmaImageAssets(
  fileKey: string,
  imageUses: FigmaImageUse[]
): Promise<FigmaDesignAsset[]> {
  if (imageUses.length === 0) return [];

  // This is Figma's Tier 2 image-fills endpoint. It returns temporary source
  // URLs for the imageRef values found on the selected frame's image fills.
  const fillResponse = await figmaJson<{ images?: Record<string, string> }>(
    "/files/" + fileKey + "/images"
  );
  const imageUrls = fillResponse.images ?? {};
  const selectedUses = imageUses.slice(0, 8);
  const assets: FigmaDesignAsset[] = [];
  let totalBytes = 0;

  for (const use of selectedUses) {
    const imageUrl = imageUrls[use.imageRef];
    if (!imageUrl) continue;

    const response = await fetch(imageUrl, {
      cache: "no-store",
      signal: AbortSignal.timeout(20000),
    });
    if (!response.ok) continue;

    const mimeType = response.headers.get("content-type")?.split(";")[0].trim() || "application/octet-stream";
    const extension = getImageExtension(mimeType, imageUrl);
    if (!extension) continue;

    const imageBuffer = Buffer.from(await response.arrayBuffer());
    if (imageBuffer.length === 0 || imageBuffer.length > 8 * 1024 * 1024) continue;
    if (totalBytes + imageBuffer.length > 20 * 1024 * 1024) break;
    totalBytes += imageBuffer.length;

    const nodeToken = use.nodeId.replace(/[^0-9]/g, "-").replace(/-+/g, "-");
    const refToken = createHash("sha256").update(use.imageRef).digest("hex").slice(0, 10);
    assets.push({
      fileName: "figma-" + nodeToken + "-" + refToken + "." + extension,
      nodeId: use.nodeId,
      nodeName: use.nodeName,
      imageRef: use.imageRef,
      mimeType,
      width: use.width,
      height: use.height,
      imageData: imageBuffer.toString("base64"),
    });
  }

  return assets;
}

function getFigmaFileNameFromUrl(value: string): string {
  try {
    const lastSegment = new URL(value).pathname.split("/").filter(Boolean).at(-1) || "";
    return decodeURIComponent(lastSegment).replace(/[-_]+/g, " ").trim() || "Figma design";
  } catch {
    return "Figma design";
  }
}

export async function verifyFigmaConnection(
  value: string
): Promise<{ fileName: string; fileKey: string; cached: boolean }> {
  const { fileKey, nodeId } = parseFigmaDesignUrl(value);
  getFigmaToken();

  // This endpoint is intentionally local-only. It confirms configuration and
  // checks whether the design is already cached; it does not spend a Tier 1
  // Figma REST request just to paint the dashboard's connection badge.
  const cachedContext = await readCachedFigmaDesignContext(fileKey, nodeId);
  return {
    fileName: cachedContext?.fileName || getFigmaFileNameFromUrl(value),
    fileKey,
    cached: Boolean(cachedContext),
  };
}

async function fetchFigmaDesignContext(
  value: string,
  parsed: ParsedFigmaUrl
): Promise<FigmaDesignContext> {
  getFigmaToken();
  const { fileKey, nodeId: requestedNodeId } = parsed;
  let fileName = getFigmaFileNameFromUrl(value);
  let targetNodeId = requestedNodeId;

  // Only fetch the full file tree when the URL does not identify a frame.
  // A frame-specific URL already gives us the exact node ID and avoids one
  // additional Tier 1 GET file request.
  if (!targetNodeId) {
    const file = await figmaJson<{ name?: string; document?: FigmaNode }>(
      "/files/" + fileKey + "?depth=2"
    );
    if (!file.document) {
      throw new FigmaConnectionError("Figma returned no design document for this file.", 502);
    }
    fileName = file.name || fileName;
    const selected = chooseHomepageFrame(file.document);
    if (!selected?.id) {
      throw new FigmaConnectionError("Figma file is accessible, but no homepage-sized frame was found. Copy a link directly to the homepage frame (Share/Copy link with the frame selected) and load that link.", 422);
    }
    targetNodeId = selected.id;
  }

  const nodeResponse = await figmaJson<{
    nodes?: Record<string, { document?: FigmaNode }>;
  }>("/files/" + fileKey + "/nodes?ids=" + encodeURIComponent(targetNodeId) + "&depth=8");
  const root = nodeResponse.nodes?.[targetNodeId]?.document;
  if (!root) {
    throw new FigmaConnectionError("Figma could not read the selected homepage frame. Select the main homepage frame and copy its link again.", 422);
  }

  const imageResponse = await figmaJson<{ images?: Record<string, string | null> }>(
    "/images/" + fileKey + "?ids=" + encodeURIComponent(targetNodeId) + "&format=png&scale=0.8"
  );
  const imageUrl = imageResponse.images?.[targetNodeId];
  if (!imageUrl) {
    throw new FigmaConnectionError("Figma returned the homepage layers but could not export a screenshot for the selected frame.", 422);
  }

  const imageResponseBody = await fetch(imageUrl, {
    cache: "no-store",
    signal: AbortSignal.timeout(20000),
  });
  if (!imageResponseBody.ok) {
    throw new FigmaConnectionError("Figma's temporary screenshot URL could not be downloaded. Refresh the design and try again.", 502);
  }
  const imageBuffer = Buffer.from(await imageResponseBody.arrayBuffer());
  if (imageBuffer.byteLength === 0 || imageBuffer.byteLength > 7 * 1024 * 1024) {
    throw new FigmaConnectionError("The homepage screenshot is empty or exceeds the 7 MB design-context limit. Link a smaller homepage frame and try again.", 413);
  }

  const heroAssetRoot = findNamedFigmaNode(root, "Button/Top") ?? root;
  const heroImageUses = collectFigmaImageUses(heroAssetRoot);
  // Some Figma file-node responses keep component-instance children on the
  // page tree. If the instance subtree is collapsed, fall back to the selected
  // frame's visible image fills; the hero is the first top-level section.
  const imageUses = heroImageUses.length > 0
    ? heroImageUses
    : collectFigmaImageUses(root);
  const assets = await downloadFigmaImageAssets(fileKey, imageUses);

  if (imageUses.length > 0 && assets.length === 0) {
    throw new FigmaConnectionError(
      "The Figma hero contains image fills, but Horizon AI could not download the original image assets. Check the file_content:read scope and retry after any Figma API limit resets.",
      502
    );
  }

  console.info("HORIZON FIGMA IMAGE ASSETS DOWNLOADED", {
    fileKey,
    heroFrame: heroAssetRoot.name || root.name || "Homepage frame",
    imageFillsFound: imageUses.length,
    imageAssetsDownloaded: assets.length,
    imageBytes: assets.reduce((total, asset) => total + Math.floor(asset.imageData.length * 0.75), 0),
  });

  return {
    fileName,
    fileKey,
    nodeId: targetNodeId,
    nodeName: root.name || "Homepage frame",
    summary: summarizeNodeTree(root),
    imageData: imageBuffer.toString("base64"),
    imageMimeType: imageResponseBody.headers.get("content-type")?.split(";")[0] || "image/png",
    assets,
  };
}

export async function loadFigmaDesignContext(value: string): Promise<FigmaDesignContext> {
  const parsed = parseFigmaDesignUrl(value);
  const key = getFigmaCacheKey(parsed.fileKey, parsed.nodeId);
  const cached = await readCachedFigmaDesignContext(parsed.fileKey, parsed.nodeId);
  if (cached) {
    console.info("HORIZON FIGMA CONTEXT CACHE HIT", {
      fileKey: parsed.fileKey,
      nodeId: cached.nodeId,
    });
    return cached;
  }

  const existingLoad = figmaContextLoadsInFlight.get(key);
  if (existingLoad) return existingLoad;

  const loadPromise = (async () => {
    const context = await fetchFigmaDesignContext(value, parsed);
    await writeCachedFigmaDesignContext(parsed.fileKey, parsed.nodeId, context);
    figmaContextMemoryCache.set(key, context);
    console.info("HORIZON FIGMA CONTEXT CACHED LOCALLY", {
      fileKey: context.fileKey,
      nodeId: context.nodeId,
    });
    return context;
  })();

  figmaContextLoadsInFlight.set(key, loadPromise);
  try {
    return await loadPromise;
  } finally {
    figmaContextLoadsInFlight.delete(key);
  }
}
