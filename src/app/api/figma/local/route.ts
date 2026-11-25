import { NextResponse } from "next/server";
import { getLocalFigDesignStatus, LocalFigDesignError } from "@/lib/figma-local-reader";

export const runtime = "nodejs";

export async function GET() {
  try {
    const design = await getLocalFigDesignStatus();
    return NextResponse.json({
      success: true,
      available: true,
      source: "local-file",
      design,
    });
  } catch (error) {
    const status = error instanceof LocalFigDesignError ? error.status : 500;
    const message = error instanceof Error ? error.message : "The local .fig file could not be read.";
    return NextResponse.json({
      success: false,
      available: false,
      source: "local-file",
      error: message,
    }, { status });
  }
}
