import { NextResponse } from "next/server";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const THEME_DIR = path.join(
  process.cwd(),
  "themes",
  "horizon"
);

const ALLOWED_EXTENSIONS = [
  ".liquid",
  ".json",
  ".js",
  ".css",
];

const STOP_WORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "want",
  "need",
  "how",
  "what",
  "where",
  "which",
  "when",
  "does",
  "this",
  "that",
  "from",
  "into",
  "have",
  "has",
  "are",
  "is",
  "my",
  "your",
  "actual",
  "please",
  "change",
  "explain",
  "analyze",
  "tell",
  "me",
  "can",
  "you",
  "to",
  "on",
  "of",
  "it",
]);

async function getThemeFiles(
  dir: string
): Promise<string[]> {
  const entries = await readdir(dir, {
    withFileTypes: true,
  });

  const files = await Promise.all(
    entries.map(async (entry) => {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        return getThemeFiles(fullPath);
      }

      if (
        entry.isFile() &&
        ALLOWED_EXTENSIONS.includes(
          path.extname(entry.name).toLowerCase()
        )
      ) {
        return [fullPath];
      }

      return [];
    })
  );

  return files.flat();
}

// Expand related Shopify theme keywords
function expandKeywords(query: string): string[] {
  const lowerQuery = query.toLowerCase();

  const words = lowerQuery
    .replace(/[^a-z0-9_\s-]/g, " ")
    .split(/\s+/)
    .filter(
      (word) =>
        word.length > 2 &&
        !STOP_WORDS.has(word)
    );

  const expanded = new Set(words);

  // Collection page intent
  if (
    lowerQuery.includes("collection") ||
    lowerQuery.includes("product listing")
  ) {
    [
      "collection",
      "main-collection",
      "collection-list",
      "collection-grid",
      "product-grid",
      "template-collection",
    ].forEach((word) => expanded.add(word));
  }

  // Product card intent
  if (
    lowerQuery.includes("product card") ||
    lowerQuery.includes("product cards")
  ) {
    [
      "product-card",
      "product_card",
      "product-card-styles",
      "card-gallery",
      "card-image",
      "product-grid",
      "_product-card",
    ].forEach((word) => expanded.add(word));
  }

  // Product image intent
  if (
    lowerQuery.includes("image") ||
    lowerQuery.includes("photo")
  ) {
    [
      "image",
      "image-ratio",
      "aspect_ratio",
      "image_width",
      "image_height",
      "media",
      "gallery",
      "object-fit",
    ].forEach((word) => expanded.add(word));
  }

  // Size / dimensions intent
  if (
    lowerQuery.includes("size") ||
    lowerQuery.includes("dimension") ||
    lowerQuery.includes("larger") ||
    lowerQuery.includes("increase")
  ) {
    [
      "size",
      "width",
      "height",
      "aspect_ratio",
      "card_size",
      "product_card_size",
      "mobile_product_card_size",
      "columns",
    ].forEach((word) => expanded.add(word));
  }

  // Header / logo intent
  if (lowerQuery.includes("logo")) {
    [
      "logo",
      "logo_height",
      "logo_height_mobile",
      "logo_width",
    ].forEach((word) => expanded.add(word));
  }

  if (lowerQuery.includes("header")) {
    [
      "header",
      "header-logo",
      "header_logo",
    ].forEach((word) => expanded.add(word));
  }

  return [...expanded];
}

// Calculate search score
function calculateFileScore(
  query: string,
  keywords: string[],
  relativePath: string,
  fileName: string,
  content: string
): number {
  let score = 0;

  const lowerQuery = query.toLowerCase();
  const lowerContent = content.toLowerCase();

  const isCollectionImageRequest =
    lowerQuery.includes("collection") &&
    (
      lowerQuery.includes("product card") ||
      lowerQuery.includes("image")
    );

  // Filename and content matches
  for (const keyword of keywords) {
    if (fileName.includes(keyword)) {
      score += 15;
    }

    if (relativePath.includes(keyword)) {
      score += 8;
    }

    const matches =
      lowerContent.split(keyword).length - 1;

    score += Math.min(matches, 3);
  }

  // Collection product card priorities
  if (isCollectionImageRequest) {
    const priorities: Record<string, number> = {
      "sections/main-collection.liquid": 100,
      "blocks/_product-card.liquid": 95,
      "snippets/product-card.liquid": 90,
      "snippets/card-gallery.liquid": 85,
      "snippets/product-grid.liquid": 80,
      "snippets/product-card-styles.liquid": 75,
      "templates/collection.json": 70,
      "config/settings_schema.json": 60,
      "assets/base.css": 55,
    };

    score += priorities[relativePath] ?? 0;
  }

  // Logo priorities
  if (lowerQuery.includes("logo")) {
    if (
      relativePath === "blocks/_header-logo.liquid"
    ) {
      score += 50;
    }

    if (
      relativePath === "config/settings_schema.json"
    ) {
      score += 40;
    }
  }

  return score;
}

export async function POST(req: Request) {
  try {
    const { query } = await req.json();

    if (!query || typeof query !== "string") {
      return NextResponse.json(
        {
          success: false,
          message: "Search query required",
        },
        { status: 400 }
      );
    }

    const files = (
      await getThemeFiles(THEME_DIR)
    ).filter((file) => {
      const relativePath = path.relative(
        THEME_DIR,
        file
      );

      return !relativePath.startsWith(
        "locales" + path.sep
      );
    });

    const keywords = expandKeywords(query);

    const results: {
      file: string;
      score: number;
    }[] = [];

    for (const fullPath of files) {
      const content = await readFile(
        fullPath,
        "utf-8"
      );

      const relativePath = path
        .relative(THEME_DIR, fullPath)
        .replace(/\\/g, "/")
        .toLowerCase();

      const fileName = path
        .basename(relativePath)
        .replace(/_/g, "-");

      const score = calculateFileScore(
        query,
        keywords,
        relativePath,
        fileName,
        content
      );

      if (score > 0) {
        results.push({
          file: relativePath,
          score,
        });
      }
    }

    results.sort((a, b) => b.score - a.score);

    console.log("\n========== THEME SEARCH ==========");
    console.log("Query:", query);
    console.log("Keywords:", keywords);
    console.table(results.slice(0, 10));

    return NextResponse.json({
      success: true,
      query,
      keywords,
      totalFiles: files.length,
      results: results.slice(0, 10),
    });
  } catch (error) {
    console.error("Theme Search Error:", error);

    return NextResponse.json(
      {
        success: false,
        message: "Theme search failed",
      },
      { status: 500 }
    );
  }
}