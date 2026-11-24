import { Buffer } from "node:buffer";

export type FigmaDesignContext = {
  fileName: string;
  fileKey: string;
  nodeId: string;
  nodeName: string;
  summary: string;
  imageData: string;
  imageMimeType: string;
};

export class FigmaConnectionError extends Error {
  status: number;

  constructor(message: string, status = 502) {
    super(message);
    this.name = "FigmaConnectionError";
    this.status = status;
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
      throw new FigmaConnectionError("Figma API rate limit reached. Wait briefly and try connecting again.", 429);
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

export async function verifyFigmaConnection(value: string): Promise<{ fileName: string; fileKey: string }> {
  const { fileKey } = parseFigmaDesignUrl(value);
  const file = await figmaJson<{ name?: string; document?: FigmaNode }>("/files/" + fileKey + "?depth=2");
  if (!file.document) {
    throw new FigmaConnectionError("Figma returned file metadata but no readable design document.", 502);
  }
  return { fileName: file.name || "Figma design", fileKey };
}

export async function loadFigmaDesignContext(value: string): Promise<FigmaDesignContext> {
  const { fileKey, nodeId: requestedNodeId } = parseFigmaDesignUrl(value);
  const file = await figmaJson<{ name?: string; document?: FigmaNode }>("/files/" + fileKey + "?depth=2");
  if (!file.document) {
    throw new FigmaConnectionError("Figma returned no design document for this file.", 502);
  }

  let targetNodeId = requestedNodeId;
  if (!targetNodeId) {
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

  return {
    fileName: file.name || "Figma design",
    fileKey,
    nodeId: targetNodeId,
    nodeName: root.name || "Homepage frame",
    summary: summarizeNodeTree(root),
    imageData: imageBuffer.toString("base64"),
    imageMimeType: imageResponseBody.headers.get("content-type")?.split(";")[0] || "image/png",
  };
}
