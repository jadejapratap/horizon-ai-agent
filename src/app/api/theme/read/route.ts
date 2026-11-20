import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import path from "node:path";

const THEME_DIR = path.join(
  process.cwd(),
  "themes",
  "horizon"
);

export async function POST(req: Request) {
  try {
    const { file } = await req.json();

    if (!file || typeof file !== "string") {
      return NextResponse.json(
        { success: false, message: "File path is required" },
        { status: 400 }
      );
    }

    const normalizedPath = file.replace(/\\/g, "/");

    const allowedExtensions = [
      ".liquid",
      ".json",
      ".js",
      ".css",
      ".md",
      ".svg",
    ];

    if (
      path.posix.isAbsolute(normalizedPath) ||
      normalizedPath.split("/").includes("..") ||
      !allowedExtensions.includes(path.posix.extname(normalizedPath))
    ) {
      return NextResponse.json(
        { success: false, message: "Invalid theme file path" },
        { status: 400 }
      );
    }

    const fullPath = path.resolve(THEME_DIR, normalizedPath);

    if (!fullPath.startsWith(THEME_DIR + path.sep)) {
      return NextResponse.json(
        { success: false, message: "Access denied" },
        { status: 403 }
      );
    }

    const content = await readFile(fullPath, "utf-8");

    return NextResponse.json({
      success: true,
      file: normalizedPath,
      content,
    });

  } catch (error) {
    console.error("Theme File Reader Error:", error);

    return NextResponse.json(
      {
        success: false,
        message: "Unable to read theme file.",
      },
      { status: 500 }
    );
  }
}