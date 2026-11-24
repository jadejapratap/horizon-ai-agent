import { NextRequest, NextResponse } from "next/server";
import { FigmaConnectionError, verifyFigmaConnection } from "@/lib/figma-connection";

export const runtime = "nodejs";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => null);
    const url = body && typeof body.url === "string" ? body.url : "";
    if (!url) {
      return NextResponse.json({ success: false, error: "A Figma Design URL is required." }, { status: 400 });
    }

    const result = await verifyFigmaConnection(url);
    return NextResponse.json({ success: true, ...result });
  } catch (error) {
    const status = error instanceof FigmaConnectionError ? error.status : 502;
    const message = error instanceof Error ? error.message : "Figma connection check failed.";
    console.error("HORIZON FIGMA CONNECTION CHECK FAILED:", message);
    return NextResponse.json({ success: false, error: message }, { status });
  }
}
