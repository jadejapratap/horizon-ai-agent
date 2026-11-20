import { NextResponse } from "next/server";
import { readdir } from "node:fs/promises";
import path from "node:path";



const THEME_DIR = path.join(
  process.cwd(),
  "themes",
  "horizon"
);

async function getThemeFiles(
  directory: string,
  baseDirectory: string = THEME_DIR
): Promise<string[]> {
  const entries = await readdir(directory, {
    withFileTypes: true,
  });

  const files: string[] = [];

  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;

    const fullPath = path.join(directory, entry.name);

    if (entry.isDirectory()) {
      files.push(
        ...(await getThemeFiles(fullPath, baseDirectory))
      );
    } else {
      files.push(
        path.relative(baseDirectory, fullPath).replace(/\\/g, "/")
      );
    }
  }

  return files;
}

export async function GET() {
  try {
    const files = await getThemeFiles(THEME_DIR);

    return NextResponse.json({
      success: true,
      totalFiles: files.length,
      files,
    });
  } catch (error) {
    console.error("Theme Reader Error:", error);

    return NextResponse.json(
      {
        success: false,
        message: "Unable to read Horizon theme files.",
      },
      { status: 500 }
    );
  }
}