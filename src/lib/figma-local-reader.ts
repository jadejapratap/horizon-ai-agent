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
  const relevantKey = /(?:size|transform|paint|fill|stroke|effect|opacity|visible|corner|text|font|line|letter|align|layout|axis|spacing|padding|clip|constraint|component|symbol|override|shape|radius|character|style|blend|rotation|position|bound|grid|overflow|mask|image|gradient|scale|width|height|weight|decoration|stack|sizing|distribution|justify|wrap|flex|shadow|blur)/i;

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


export type LocalFigInspectionNodeSummary = {
  id: string;
  name: string;
  type: string;
  width: number | null;
  height: number | null;
  childCount: number;
};

export type LocalFigComponentTextSummary = {
  id: string;
  name: string;
  overrideKey: string | null;
  baseCharacters: string;
  effectiveCharacters: string;
  overrideApplied: boolean;
  matchedOverridePath: string[] | null;
  fontFamily: string | null;
  fontStyle: string | null;
  fontSize: number | null;
  lineHeight: number | null;
  fillPaints: any[];
};

export type LocalFigComponentResolution = {
  status: "resolved" | "missing-master" | "external-library" | "no-reference";
  instanceNodeId: string;
  instanceName: string;
  referenceField: string | null;
  referenceId: string | null;
  referenceCandidates: Array<{ field: string; id: string }>;
  componentKey: string | null;
  master: {
    id: string;
    name: string;
    type: string;
    width: number | null;
    height: number | null;
    parentPath: string[];
    childCount: number;
  } | null;
  masterTextNodes: LocalFigComponentTextSummary[];
  textOverrides: Array<{ path: string[]; characters: string | null }>;
  componentProperties: Record<string, any> | null;
  masterHierarchy: any | null;
  masterHierarchyNodeCount: number;
  masterHierarchyTruncated: boolean;
  diagnostic: string;
};

export type LocalFigInspectionContext = {
  fileName: string;
  filePath: string;
  formatVersion: number | null;
  nodeCount: number;
  pageCount: number;
  pageNames: string[];
  pages: Array<{ id: string; name: string; childCount: number }>;
  selectedPage: { id: string; name: string; childCount: number };
  focusedNode: (LocalFigInspectionNodeSummary & { parentPath: string[] }) | null;
  focusedChildren: LocalFigInspectionNodeSummary[];
  topLevelChildren: LocalFigInspectionNodeSummary[];
  topLevelFrames: LocalFigInspectionNodeSummary[];
  designPageTree: string;
  treeTruncated: boolean;
};

function isLocalFigPageNode(node: any): boolean {
  return /^(?:CANVAS|PAGE)$/i.test(String(node?.type || ""));
}

function isLocalFigFrameNode(node: any): boolean {
  return /^(?:FRAME|COMPONENT|INSTANCE|COMPONENT_SET|SECTION)$/i.test(String(node?.type || ""));
}

function getLocalFigNodeSize(node: any): { width: number | null; height: number | null } {
  const width = Number(node?.size?.x);
  const height = Number(node?.size?.y);
  return {
    width: Number.isFinite(width) ? width : null,
    height: Number.isFinite(height) ? height : null,
  };
}

function getLocalFigGuidId(value: any): string | null {
  if (typeof value === "string") {
    const normalized = value.trim();
    return /^\d+:\d+$/.test(normalized) ? normalized : null;
  }
  if (!value || typeof value !== "object") return null;

  const sessionID = Number(value.sessionID ?? value.sessionId);
  const localID = Number(value.localID ?? value.localId);
  if (Number.isFinite(sessionID) && Number.isFinite(localID)) {
    return String(sessionID) + ":" + String(localID);
  }

  if (value.guid && value.guid !== value) return getLocalFigGuidId(value.guid);
  if (value.id && value.id !== value) return getLocalFigGuidId(value.id);
  return null;
}

function componentOverridePathMatches(nodePathKeys: string[], overridePath: string[]): boolean {
  if (nodePathKeys.length === 0 || overridePath.length === 0) return false;
  if (nodePathKeys[nodePathKeys.length - 1] !== overridePath[overridePath.length - 1]) return false;

  let pathIndex = nodePathKeys.length - 2;
  for (let overrideIndex = overridePath.length - 2; overrideIndex >= 0; overrideIndex -= 1) {
    while (pathIndex >= 0 && nodePathKeys[pathIndex] !== overridePath[overrideIndex]) {
      pathIndex -= 1;
    }
    if (pathIndex < 0) return false;
    pathIndex -= 1;
  }
  return true;
}

function resolveLocalFigComponentReference(
  doc: any,
  instanceNode: any,
  byId: Map<string, any>,
  parentById: Map<string, string>,
  childrenFor: (node: any) => any[]
): LocalFigComponentResolution {
  const instanceId = getNodeId(instanceNode) || "unknown";
  const symbolData = instanceNode?.symbolData && typeof instanceNode.symbolData === "object"
    ? instanceNode.symbolData
    : {};
  const rawOverrides = Array.isArray(symbolData.symbolOverrides)
    ? symbolData.symbolOverrides
    : Array.isArray(instanceNode?.symbolOverrides)
      ? instanceNode.symbolOverrides
      : [];
  const componentKeyValue = symbolData.componentKey ?? symbolData.symbolKey ?? instanceNode?.componentKey ?? instanceNode?.symbolKey;
  const componentKey = typeof componentKeyValue === "string" && componentKeyValue.trim()
    ? componentKeyValue.trim()
    : null;

  const referenceCandidates: Array<{ field: string; id: string }> = [];
  const candidateFields: Array<[string, any]> = [
    ["symbolData.symbolID", symbolData.symbolID],
    ["symbolData.componentId", symbolData.componentId],
    ["symbolData.mainComponentId", symbolData.mainComponentId],
    ["componentId", instanceNode?.componentId],
    ["mainComponentId", instanceNode?.mainComponentId],
    ["mainComponent", instanceNode?.mainComponent],
    ["symbolID", instanceNode?.symbolID],
  ];
  for (const [field, value] of candidateFields) {
    const id = getLocalFigGuidId(value);
    if (id && !referenceCandidates.some((candidate) => candidate.id === id && candidate.field === field)) {
      referenceCandidates.push({ field, id });
    }
  }

  const matchedReference = referenceCandidates.find((candidate) => byId.has(candidate.id)) || referenceCandidates[0] || null;
  const referenceId = matchedReference?.id ?? null;
  const referenceField = matchedReference?.field ?? null;
  const masterNode = referenceId
    ? (doc?.nodeMap instanceof Map ? doc.nodeMap.get(referenceId) : undefined) || byId.get(referenceId) || null
    : null;

  const summarizeOverride = (override: any) => {
    const path = Array.isArray(override?.guidPath?.guids)
      ? override.guidPath.guids.map(getLocalFigGuidId).filter((id: string | null): id is string => Boolean(id))
      : [];
    const characters = typeof override?.textData?.characters === "string"
      ? override.textData.characters
      : null;
    return { path, characters };
  };
  const overrideSummaries = rawOverrides
    .map(summarizeOverride)
    .filter((override: { path: string[]; characters: string | null }) => override.path.length > 0 || override.characters !== null);

  let status: LocalFigComponentResolution["status"] = "no-reference";
  let diagnostic = "The INSTANCE node contains no recognized local component reference or external component key.";
  if (masterNode) {
    status = "resolved";
    diagnostic = "A native component reference resolved to an exact node ID in the parsed local .fig document.";
  } else if (referenceId) {
    status = "missing-master";
    diagnostic = "The INSTANCE contains a local component reference, but no parsed node with that exact ID was found.";
  } else if (componentKey) {
    status = "external-library";
    diagnostic = "The INSTANCE exposes a component key but no local master-node ID. The referenced library component is outside this .fig document.";
  }

  const parentPathFor = (nodeId: string): string[] => {
    const ancestors: string[] = [];
    let currentId = parentById.get(nodeId) || "";
    let guard = 0;
    while (currentId && guard < 100) {
      const ancestor = byId.get(currentId);
      if (!ancestor) break;
      ancestors.unshift(String(ancestor.name || ancestor.type || "Unnamed node"));
      currentId = parentById.get(currentId) || "";
      guard += 1;
    }
    return ancestors;
  };

  const masterTextNodes: LocalFigComponentTextSummary[] = [];
  let hierarchyNodeCount = 0;
  let hierarchyTruncated = false;
  const maxHierarchyNodes = 500;
  const maxHierarchyDepth = 24;

  const serializeMasterNode = (node: any, parentOverrideKeys: string[], depth: number): any => {
    if (hierarchyNodeCount >= maxHierarchyNodes) {
      hierarchyTruncated = true;
      return { id: getNodeId(node) || "unknown", type: String(node.type || "UNKNOWN"), truncated: true };
    }
    hierarchyNodeCount += 1;

    const id = getNodeId(node) || "unknown";
    const type = String(node.type || "UNKNOWN");
    const ownOverrideKey = getLocalFigGuidId(node.overrideKey);
    const overridePathKeys = ownOverrideKey
      ? [...parentOverrideKeys, ownOverrideKey]
      : parentOverrideKeys;
    const entry: Record<string, any> = {
      id,
      name: String(node.name || ""),
      type,
      width: getLocalFigNodeSize(node).width,
      height: getLocalFigNodeSize(node).height,
    };
    if (ownOverrideKey) entry.overrideKey = ownOverrideKey;

    if (/^TEXT$/i.test(type)) {
      const baseCharacters = typeof node.textData?.characters === "string"
        ? node.textData.characters
        : typeof node.derivedTextData?.characters === "string"
          ? node.derivedTextData.characters
          : "";
      const matchedOverrides = overrideSummaries.filter((override: { path: string[]; characters: string | null }) =>
        override.characters !== null && componentOverridePathMatches(overridePathKeys, override.path)
      );
      const appliedOverride = matchedOverrides.length > 0 ? matchedOverrides[matchedOverrides.length - 1] : null;
      const fontName = node.fontName && typeof node.fontName === "object" ? node.fontName : {};
      const textSummary: LocalFigComponentTextSummary = {
        id,
        name: String(node.name || "Text"),
        overrideKey: ownOverrideKey,
        baseCharacters,
        effectiveCharacters: appliedOverride?.characters ?? baseCharacters,
        overrideApplied: Boolean(appliedOverride),
        matchedOverridePath: appliedOverride?.path ?? null,
        fontFamily: typeof fontName.family === "string" ? fontName.family : null,
        fontStyle: typeof fontName.style === "string" ? fontName.style : null,
        fontSize: Number.isFinite(Number(node.fontSize)) ? Number(node.fontSize) : null,
        lineHeight: Number.isFinite(Number(node.lineHeight)) ? Number(node.lineHeight) : null,
        fillPaints: Array.isArray(node.fillPaints) ? serializableValue(node.fillPaints) : [],
      };
      masterTextNodes.push(textSummary);
      entry.text = {
        baseCharacters: textSummary.baseCharacters,
        effectiveCharacters: textSummary.effectiveCharacters,
        overrideApplied: textSummary.overrideApplied,
        fontFamily: textSummary.fontFamily,
        fontStyle: textSummary.fontStyle,
        fontSize: textSummary.fontSize,
        lineHeight: textSummary.lineHeight,
      };
    }

    const childNodes = childrenFor(node);
    entry.childCount = childNodes.length;
    if (childNodes.length > 0) {
      if (depth >= maxHierarchyDepth || hierarchyNodeCount >= maxHierarchyNodes) {
        hierarchyTruncated = true;
        entry.childrenTruncated = childNodes.length;
      } else {
        entry.children = childNodes.map((child) => serializeMasterNode(child, overridePathKeys, depth + 1));
      }
    }
    return entry;
  };

  let masterSummary: LocalFigComponentResolution["master"] = null;
  let masterHierarchy: any = null;
  if (masterNode) {
    const masterId = getNodeId(masterNode) || referenceId || "unknown";
    const masterSize = getLocalFigNodeSize(masterNode);
    masterSummary = {
      id: masterId,
      name: String(masterNode.name || masterNode.type || "Unnamed component"),
      type: String(masterNode.type || "UNKNOWN"),
      width: masterSize.width,
      height: masterSize.height,
      parentPath: parentPathFor(masterId),
      childCount: childrenFor(masterNode).length,
    };
    masterHierarchy = serializeMasterNode(masterNode, [], 0);
    if (!/^(?:COMPONENT|COMPONENT_SET|SYMBOL|SYMBOL_MASTER)$/i.test(masterSummary.type)) {
      diagnostic += " The referenced node exists, but its type is " + masterSummary.type + " rather than a recognized component-master type.";
    }
  }

  const componentProperties = symbolData.componentProperties ?? instanceNode?.componentProperties ?? instanceNode?.componentPropertyValues ?? null;
  let componentPropertySummary: Record<string, any> | null = null;
  if (componentProperties && typeof componentProperties === "object") {
    componentPropertySummary = {};
    for (const [key, value] of Object.entries(componentProperties).slice(0, 50)) {
      componentPropertySummary[key] =
        typeof value === "string" || typeof value === "number" || typeof value === "boolean" || value === null
          ? value
          : serializableValue(value);
    }
  }

  return {
    status,
    instanceNodeId: instanceId,
    instanceName: String(instanceNode.name || "Instance"),
    referenceField,
    referenceId,
    referenceCandidates,
    componentKey,
    master: masterSummary,
    masterTextNodes,
    textOverrides: overrideSummaries,
    componentProperties: componentPropertySummary,
    masterHierarchy,
    masterHierarchyNodeCount: hierarchyNodeCount,
    masterHierarchyTruncated: hierarchyTruncated,
    diagnostic,
  };
}

/**
 * Read-only Figma inspection entry point.
 *
 * Unlike readLocalFigDesignContext(), this function does not extract/copy image
 * assets into the Shopify theme. It returns the selected page's native node
 * hierarchy so inspection prompts never need to enter the theme-edit pipeline.
 */
export async function readLocalFigInspectionContext(
  prompt = ""
): Promise<LocalFigInspectionContext> {
  const filePath = await resolveFigFile();
  const metadata = await stat(filePath);
  if (!metadata.isFile()) {
    throw new LocalFigDesignError("The selected Figma path is not a regular file.", 400);
  }
  if (metadata.size <= 0 || metadata.size > MAX_FILE_BYTES) {
    throw new LocalFigDesignError("The .fig file must be between 1 byte and 80 MB.", 413);
  }

  const bytes = await readFile(filePath);
  if (bytes[0] !== 0x50 || bytes[1] !== 0x4b) {
    throw new LocalFigDesignError(
      "The selected file is not a supported zipped .fig archive. Keep the original Figma .fig export unchanged.",
      415
    );
  }

  let doc: any;
  try {
    doc = parseFig(new Uint8Array(bytes));
  } catch (error) {
    const detail = error instanceof Error ? error.message : "unknown parser error";
    throw new LocalFigDesignError(
      "OpenFig could not parse " + path.basename(filePath) + ": " + detail,
      422
    );
  }

  if (!Array.isArray(doc.nodes) || doc.nodes.length === 0) {
    throw new LocalFigDesignError("The .fig file parsed but contained no design nodes.", 422);
  }

  const nodes: any[] = doc.nodes;
  const byId = new Map<string, any>();
  for (const node of nodes) {
    const id = getNodeId(node);
    if (id) byId.set(id, node);
  }

  const childrenMap: Map<string, any[]> =
    doc.childrenMap instanceof Map ? doc.childrenMap : new Map();
  const parentById = new Map<string, string>();
  for (const [parentId, children] of childrenMap.entries()) {
    for (const child of children) {
      const childId = getNodeId(child);
      if (childId) parentById.set(childId, String(parentId));
    }
  }

  const childrenFor = (node: any): any[] => {
    const id = getNodeId(node);
    if (!id) return [];
    const indexed = childrenMap.get(id);
    if (Array.isArray(indexed)) return indexed;
    return nodes.filter((candidate) => parentById.get(getNodeId(candidate) || "") === id);
  };

  const pageNodes = nodes.filter(isLocalFigPageNode);
  if (pageNodes.length === 0) {
    throw new LocalFigDesignError("The .fig file contains nodes but no Figma page/canvas nodes.", 422);
  }

  const pages = pageNodes.map((page) => {
    const id = getNodeId(page) || "unknown";
    return {
      id,
      name: String(page.name || "Unnamed page"),
      childCount: childrenFor(page).length,
    };
  });

  const explicitPageMatch = prompt.match(/\b(?:page|under|within)\s+["']([^"']{1,100})["']/i);
  const requestedPageName = explicitPageMatch?.[1]?.trim().toLowerCase();
  const selectedPageNode =
    (requestedPageName
      ? pageNodes.find((page) => String(page.name || "").trim().toLowerCase() === requestedPageName)
      : undefined) ||
    pageNodes.find((page) => String(page.name || "").trim().toLowerCase() === "design") ||
    pageNodes[0];

  const selectedPageId = getNodeId(selectedPageNode) || "unknown";
  const directChildren = childrenFor(selectedPageNode);

  const summarizeNode = (node: any): LocalFigInspectionNodeSummary => {
    const size = getLocalFigNodeSize(node);
    return {
      id: getNodeId(node) || "unknown",
      name: String(node.name || node.type || "Unnamed node"),
      type: String(node.type || "UNKNOWN"),
      width: size.width,
      height: size.height,
      childCount: childrenFor(node).length,
    };
  };

  const topLevelChildren = directChildren.map(summarizeNode);
  const topLevelFrames = directChildren.filter(isLocalFigFrameNode).map(summarizeNode);

  // A focused node request lets the agent inspect one frame subtree at a time.
  // This avoids truncating an entire multi-section Figma page before the target
  // homepage/frame has been understood.
  const requestedNodeMatch = prompt.match(/\b(?:node(?:\s*id)?|frame(?:\s*id)?|nodeId)\s*[:#]?\s*["']?(\d+:\d+)/i);
  const requestedNodeId = requestedNodeMatch?.[1] || null;
  const requestedNode = requestedNodeId ? byId.get(requestedNodeId) : null;

  if (requestedNodeId && !requestedNode) {
    throw new LocalFigDesignError("Figma node " + requestedNodeId + " was not found in " + path.basename(filePath) + ".", 404);
  }

  const isWithinSelectedPage = (node: any): boolean => {
    let currentId = getNodeId(node);
    let guard = 0;
    while (currentId && guard < 100) {
      if (currentId === selectedPageId) return true;
      currentId = parentById.get(currentId) || "";
      guard += 1;
    }
    return false;
  };

  if (requestedNode && !isWithinSelectedPage(requestedNode)) {
    throw new LocalFigDesignError(
      "Figma node " + requestedNodeId + " is not inside the selected page '" + String(selectedPageNode.name || "Unnamed page") + "'.",
      400
    );
  }

  const focusedNode = requestedNode
    ? {
        ...summarizeNode(requestedNode),
        parentPath: (() => {
          const ancestors: string[] = [];
          let currentId = parentById.get(requestedNodeId || "") || "";
          let guard = 0;
          while (currentId && guard < 100) {
            const ancestor = byId.get(currentId);
            if (!ancestor) break;
            ancestors.unshift(String(ancestor.name || ancestor.type || "Unnamed node"));
            if (currentId === selectedPageId) break;
            currentId = parentById.get(currentId) || "";
            guard += 1;
          }
          return ancestors;
        })(),
      }
    : null;

  const focusedComponentResolution = requestedNode && /^INSTANCE$/i.test(String(requestedNode.type || ""))
    ? resolveLocalFigComponentReference(doc, requestedNode, byId, parentById, childrenFor)
    : null;

  // Keep the focused node's immediate child inventory separate from its
  // potentially large serialized subtree. These exact names/IDs/dimensions
  // remain available even when the deeper hierarchy hits the model context cap.
  const focusedChildren: LocalFigInspectionNodeSummary[] = requestedNode
    ? childrenFor(requestedNode).map(summarizeNode)
    : [];

  const inspectionRoot = requestedNode || selectedPageNode;
  const inspectionRootChildren = requestedNode ? [requestedNode] : directChildren;

  let remainingNodes = 6200;
  const maxDepth = 22;
  const serializeHierarchy = (node: any, depth: number): any => {
    if (remainingNodes <= 0) {
      return {
        id: getNodeId(node) || "unknown",
        name: String(node.name || node.type || "Unnamed node"),
        type: String(node.type || "UNKNOWN"),
        truncated: true,
      };
    }
    remainingNodes -= 1;

    const size = getLocalFigNodeSize(node);
    const id = getNodeId(node);
    const childNodes = childrenFor(node);
    const serialized: Record<string, any> = {
      id: id || "unknown",
      name: String(node.name || ""),
      type: String(node.type || "UNKNOWN"),
      width: size.width,
      height: size.height,
      childCount: childNodes.length,
    };

    const properties = nodeProperties(node);
    if (Object.keys(properties).length > 0) {
      serialized.properties = properties;
    }

    if (childNodes.length > 0) {
      if (depth >= maxDepth || remainingNodes <= 0) {
        serialized.childrenTruncated = childNodes.length;
      } else {
        serialized.children = childNodes.map((child) => serializeHierarchy(child, depth + 1));
      }
    }
    return serialized;
  };

  const hierarchy = {
    source: "figma/design.fig",
    note: "Native Figma node hierarchy parsed locally. This is structured .fig data, not a screenshot or rendered image.",
    selectedPage: {
      id: selectedPageId,
      name: String(selectedPageNode.name || "Unnamed page"),
      childCount: directChildren.length,
    },
    ...(focusedNode
      ? {
          focusedNode,
          focusedChildren,
          focusedSubtree: serializeHierarchy(inspectionRoot, 0),
        }
      : {
          topLevelChildren: inspectionRootChildren.map((node) => serializeHierarchy(node, 0)),
        }),
  };

  const completeTree = JSON.stringify(hierarchy, null, 2);
  const treeTruncated = completeTree.length > 72000 || remainingNodes <= 0;
  const designPageTree = treeTruncated
    ? completeTree.slice(0, 72000) + "\n[Native node hierarchy truncated for model-context limits.]"
    : completeTree;

  return {
    fileName: path.basename(filePath),
    filePath: path.relative(process.cwd(), filePath).replace(/\\/g, "/"),
    formatVersion: Number.isFinite(Number(doc.header?.version)) ? Number(doc.header.version) : null,
    nodeCount: nodes.length,
    pageCount: pages.length,
    pageNames: pages.map((page) => page.name),
    pages,
    selectedPage: {
      id: selectedPageId,
      name: String(selectedPageNode.name || "Unnamed page"),
      childCount: directChildren.length,
    },
    focusedNode,
    focusedChildren,
    focusedComponentResolution,
    topLevelChildren,
    topLevelFrames,
    designPageTree,
    treeTruncated,
  };
}
