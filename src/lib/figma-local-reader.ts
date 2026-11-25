import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { nodeId, parseFig } from "openfig-core";

const FIGMA_DIRECTORY = path.resolve(process.cwd(), "figma");
const PREFERRED_FILE = path.join(FIGMA_DIRECTORY, "design.fig");
const CACHE_DIRECTORY = path.resolve(process.cwd(), ".horizon-cache", "figma-assets");
const THEME_ASSET_DIRECTORY = path.resolve(process.cwd(), "themes", "horizon", "assets");
const MAX_FILE_BYTES = 80 * 1024 * 1024;
const MAX_TREE_CHARACTERS = 76000;
const MAX_TREE_NODES = 7000;
const MAX_INLINE_IMAGE_COUNT = 18;
const MAX_INLINE_IMAGE_BYTES = 12 * 1024 * 1024;

export class LocalFigDesignError extends Error {
  status: number;

  constructor(message: string, status = 422) {
    super(message);
    this.name = "LocalFigDesignError";
    this.status = status;
  }
}

export type LocalFigImageAsset = {
  name: string;
  mimeType: "image/png" | "image/jpeg" | "image/webp" | "image/gif";
  bytes: number;
  sha256: string;
  themeAssetPath?: string;
  data?: string;
};

export type LocalFigDesignContext = {
  fileName: string;
  filePath: string;
  formatVersion: number | null;
  nodeCount: number;
  pageCount: number;
  pageNames: string[];
  frameCount: number;
  imageCount: number;
  frames: Array<{ id: string; name: string; page: string; width: number | null; height: number | null }>;
  tree: string;
  images: LocalFigImageAsset[];
};

function getNodeId(node: any): string | null {
  try {
    return nodeId(node) || null;
  } catch {
    return null;
  }
}

function safeName(value: string): string {
  return value.replace(/[^a-zA-Z0-9._-]+/g, "_").replace(/^\.+/, "").slice(0, 100) || "asset";
}

function getImageMime(bytes: Uint8Array, name: string): LocalFigImageAsset["mimeType"] | null {
  const lower = name.toLowerCase();
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return "image/png";
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "image/jpeg";
  if (bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) return "image/gif";
  if (bytes[0] === 0x52 && bytes[1] === 0x49 && bytes[2] === 0x46 && bytes[3] === 0x46 &&
      bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) return "image/webp";
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return "image/jpeg";
  if (lower.endsWith(".gif")) return "image/gif";
  if (lower.endsWith(".webp")) return "image/webp";
  return null;
}

async function resolveFigFile(): Promise<string> {
  try {
    const preferred = await stat(PREFERRED_FILE);
    if (preferred.isFile()) return PREFERRED_FILE;
  } catch {
    // A single .fig file with its original name is also accepted during the rename transition.
  }

  let names: string[] = [];
  try {
    names = (await readdir(FIGMA_DIRECTORY)).filter((name) => name.toLowerCase().endsWith(".fig"));
  } catch {
    throw new LocalFigDesignError("No local Figma folder was found. Add the original design file at figma/design.fig.", 404);
  }

  if (names.length === 0) {
    throw new LocalFigDesignError("Local Figma file not found. Add the original .fig file at figma/design.fig.", 404);
  }
  if (names.length > 1) {
    throw new LocalFigDesignError("More than one .fig file exists in figma/. Keep the target design named design.fig so Horizon can select it deterministically.", 409);
  }
  return path.join(FIGMA_DIRECTORY, names[0]);
}

function serializableValue(value: any, depth = 0, seen = new WeakSet<object>()): any {
  if (value === null || value === undefined) return value;
  if (typeof value === "string") return value.length > 1400 ? value.slice(0, 1400) + "…[truncated]" : value;
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (typeof value === "bigint") return value.toString();
  if (typeof value !== "object") return undefined;
  if (value instanceof Uint8Array || Buffer.isBuffer(value)) return "[binary " + value.byteLength + " bytes]";
  if (depth >= 5) return "[nested object omitted]";
  if (seen.has(value)) return "[circular]";
  seen.add(value);

  if (Array.isArray(value)) {
    return value.slice(0, 40).map((entry) => serializableValue(entry, depth + 1, seen));
  }

  const output: Record<string, any> = {};
  for (const [key, entry] of Object.entries(value).slice(0, 70)) {
    if (/^(?:guid|parentIndex|phase|commandsBlob|rawChunks|compiledSchema|schema|nodeMap|childrenMap|blobs)$/i.test(key)) continue;
    if (/^(?:imageData|thumbnailBytes|encodedBytes)$/i.test(key)) continue;
    const serialized = serializableValue(entry, depth + 1, seen);
    if (serialized !== undefined) output[key] = serialized;
  }
  return output;
}

function nodeProperties(node: any): Record<string, any> {
  const output: Record<string, any> = {};
  const relevantKey = /(?:size|transform|paint|fill|stroke|effect|opacity|visible|corner|text|font|line|letter|align|layout|axis|spacing|padding|clip|constraint|component|shape|radius|character|style|blend|rotation|position|bound|grid|overflow|mask|image|gradient|scale|width|height|weight|decoration|stack|sizing|distribution|justify|wrap|flex|shadow|blur)/i;

  for (const [key, value] of Object.entries(node)) {
    if (key === "guid" || key === "parentIndex" || key === "phase" || key === "type" || key === "name") continue;
    if (!relevantKey.test(key)) continue;
    const serialized = serializableValue(value);
    if (serialized !== undefined) output[key] = serialized;
  }

  if (node.textData?.characters && !output.textData) {
    output.textData = { characters: String(node.textData.characters).slice(0, 1400) };
  }
  return output;
}

function scoreName(name: string, prompt: string): number {
  const terms = prompt.toLowerCase().match(/[a-z0-9]{3,}/g) || [];
  const haystack = name.toLowerCase();
  return terms.reduce((score, term) => score + (haystack.includes(term) ? 1 : 0), 0);
}

function buildDesignTree(doc: any, prompt: string): {
  pageNames: string[];
  frames: LocalFigDesignContext["frames"];
  tree: string;
} {
  const nodes = Array.isArray(doc.nodes) ? doc.nodes : [];
  const byId = new Map<string, any>();
  for (const node of nodes) {
    const id = getNodeId(node);
    if (id) byId.set(id, node);
  }

  const childrenMap: Map<string, any[]> = doc.childrenMap instanceof Map ? doc.childrenMap : new Map();
  const childIds = new Set<string>();
  for (const children of childrenMap.values()) {
    for (const child of children) {
      const id = getNodeId(child);
      if (id) childIds.add(id);
    }
  }

  const pageNodes = nodes.filter((node: any) => /^(?:CANVAS|PAGE)$/i.test(String(node.type || "")));
  const roots = pageNodes.length > 0
    ? pageNodes
    : nodes.filter((node: any) => {
        const id = getNodeId(node);
        return Boolean(id && !childIds.has(id));
      });

  const parentById = new Map<string, string>();
  for (const [parentId, children] of childrenMap.entries()) {
    for (const child of children) {
      const childId = getNodeId(child);
      if (childId) parentById.set(childId, parentId);
    }
  }

  const pathFor = (node: any): string => {
    const parts: string[] = [];
    let current: any = node;
    let guard = 0;
    while (current && guard < 32) {
      const label = String(current.name || current.type || "").trim();
      if (label) parts.unshift(label);
      const parentId = parentById.get(getNodeId(current) || "");
      current = parentId ? byId.get(parentId) : null;
      guard += 1;
    }
    return parts.join(" / ");
  };

  const frames: LocalFigDesignContext["frames"] = nodes
    .filter((node: any) => /^(?:FRAME|COMPONENT|INSTANCE)$/i.test(String(node.type || "")))
    .slice(0, 500)
    .map((node: any) => ({
      id: getNodeId(node) || "unknown",
      name: String(node.name || node.type || "Unnamed frame"),
      page: pathFor(node).split(" / ").slice(0, -1).join(" / "),
      width: Number.isFinite(Number(node.size?.x)) ? Number(node.size.x) : null,
      height: Number.isFinite(Number(node.size?.y)) ? Number(node.size.y) : null,
    }));

  const pageNames = roots.map((node: any) => String(node.name || node.type || "Unnamed page"));
  const rankedRoots = [...roots].sort((a: any, b: any) =>
    scoreName(String(b.name || ""), prompt) - scoreName(String(a.name || ""), prompt)
  );

  let includedNodes = 0;
  const serializeTree = (node: any, depth: number): any => {
    if (includedNodes >= MAX_TREE_NODES) return { truncated: true, reason: "Node limit reached" };
    includedNodes += 1;
    const id = getNodeId(node);
    const entry: Record<string, any> = {
      id,
      type: String(node.type || "UNKNOWN"),
      name: String(node.name || ""),
    };
    if (node.size) entry.size = serializableValue(node.size);
    const properties = nodeProperties(node);
    if (Object.keys(properties).length > 0) entry.properties = properties;

    if (depth < 18 && id) {
      const children = childrenMap.get(id) || [];
      if (children.length > 0) entry.children = children.map((child) => serializeTree(child, depth + 1));
    } else if (depth >= 18) {
      const childCount = id ? (childrenMap.get(id) || []).length : 0;
      if (childCount > 0) entry.childrenTruncated = childCount;
    }
    return entry;
  };

  const pageTrees = rankedRoots.map((root: any) => serializeTree(root, 0));
  let tree = JSON.stringify({
    note: "This is the native .fig node hierarchy. It is structured source data, not a screenshot or rendered page image.",
    pages: pageTrees,
    frameIndex: frames,
  });

  if (tree.length > MAX_TREE_CHARACTERS) {
    tree = tree.slice(0, MAX_TREE_CHARACTERS) + "…[design tree clipped at the context safety limit]";
  }

  return { pageNames, frames, tree };
}

async function extractImages(
  doc: any,
  includeData: boolean,
  copyIntoThemeAssets: boolean
): Promise<LocalFigImageAsset[]> {
  if (!(doc.images instanceof Map)) return [];
  await mkdir(CACHE_DIRECTORY, { recursive: true });

  const output: LocalFigImageAsset[] = [];
  let inlineBytes = 0;

  for (const [originalName, rawBytes] of doc.images.entries()) {
    if (!(rawBytes instanceof Uint8Array) || rawBytes.byteLength === 0) continue;
    const mimeType = getImageMime(rawBytes, originalName);
    if (!mimeType) continue;

    const digest = createHash("sha256").update(rawBytes).digest("hex");
    const extension = mimeType === "image/jpeg" ? ".jpg" : mimeType === "image/png" ? ".png" : mimeType === "image/gif" ? ".gif" : ".webp";
    const fileName = digest.slice(0, 16) + "_" + safeName(path.basename(originalName, path.extname(originalName))) + extension;
    const outputPath = path.join(CACHE_DIRECTORY, fileName);
    await writeFile(outputPath, rawBytes);

    const asset: LocalFigImageAsset = {
      name: originalName,
      mimeType,
      bytes: rawBytes.byteLength,
      sha256: digest,
    };

    if (copyIntoThemeAssets && rawBytes.byteLength <= 20 * 1024 * 1024 && output.length < 120) {
      const themeFileName = "figma-" + digest.slice(0, 20) + extension;
      const themePath = path.join(THEME_ASSET_DIRECTORY, themeFileName);
      await mkdir(THEME_ASSET_DIRECTORY, { recursive: true });
      try {
        await stat(themePath);
      } catch {
        await writeFile(themePath, rawBytes, { flag: "wx" });
      }
      asset.themeAssetPath = "assets/" + themeFileName;
    }

    if (
      includeData &&
      output.length < MAX_INLINE_IMAGE_COUNT &&
      inlineBytes + rawBytes.byteLength <= MAX_INLINE_IMAGE_BYTES
    ) {
      asset.data = Buffer.from(rawBytes).toString("base64");
      inlineBytes += rawBytes.byteLength;
    }

    output.push(asset);
  }

  return output;
}

export async function readLocalFigDesignContext(
  prompt = "",
  options: { includeImageData?: boolean; copyIntoThemeAssets?: boolean } = {}
): Promise<LocalFigDesignContext> {
  const filePath = await resolveFigFile();
  const metadata = await stat(filePath);
  if (!metadata.isFile()) throw new LocalFigDesignError("The selected Figma path is not a regular file.", 400);
  if (metadata.size <= 0 || metadata.size > MAX_FILE_BYTES) {
    throw new LocalFigDesignError("The .fig file must be between 1 byte and 80 MB.", 413);
  }

  const bytes = await readFile(filePath);
  if (bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
    throw new LocalFigDesignError("The selected file is not a supported zipped .fig archive. Keep the original Figma .fig export unchanged.", 415);
  }

  let doc: any;
  try {
    doc = parseFig(new Uint8Array(bytes));
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown parser error";
    throw new LocalFigDesignError("OpenFig could not parse " + path.basename(filePath) + ": " + detail, 422);
  }

  if (!Array.isArray(doc.nodes) || doc.nodes.length === 0) {
    throw new LocalFigDesignError("The .fig file parsed but contained no design nodes.", 422);
  }

  const { pageNames, frames, tree } = buildDesignTree(doc, prompt);
  const images = await extractImages(
    doc,
    Boolean(options.includeImageData),
    Boolean(options.copyIntoThemeAssets)
  );

  return {
    fileName: path.basename(filePath),
    filePath: path.relative(process.cwd(), filePath).replace(/\\/g, "/"),
    formatVersion: Number.isFinite(Number(doc.header?.version)) ? Number(doc.header.version) : null,
    nodeCount: doc.nodes.length,
    pageCount: pageNames.length,
    pageNames,
    frameCount: frames.length,
    imageCount: images.length,
    frames,
    tree,
    images,
  };
}

export async function getLocalFigDesignStatus(): Promise<Omit<LocalFigDesignContext, "tree" | "images"> & {
  images: Array<Omit<LocalFigImageAsset, "data">>;
}> {
  const context = await readLocalFigDesignContext("", { includeImageData: false });
  return {
    fileName: context.fileName,
    filePath: context.filePath,
    formatVersion: context.formatVersion,
    nodeCount: context.nodeCount,
    pageCount: context.pageCount,
    pageNames: context.pageNames,
    frameCount: context.frameCount,
    imageCount: context.imageCount,
    frames: context.frames.slice(0, 150),
    images: context.images.map(({ name, mimeType, bytes, sha256, themeAssetPath }) => ({
      name, mimeType, bytes, sha256, ...(themeAssetPath ? { themeAssetPath } : {}),
    })),
  };
}
