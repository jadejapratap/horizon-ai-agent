import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  outputFileTracingIncludes: {
    "/api/agent": [
      "./themes/horizon/**/*.liquid",
      "./themes/horizon/**/*.json",
      "./themes/horizon/**/*.js",
      "./themes/horizon/**/*.css",
      "./shopify-toolkit/skills/shopify/**/*.md",
    ],
  },
  outputFileTracingExcludes: {
    "/api/agent": ["./themes/horizon/locales/**/*"],
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
