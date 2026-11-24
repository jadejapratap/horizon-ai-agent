import { readFile } from "node:fs/promises";
import path from "node:path";

export type ShopifyAgentProfile = {
  id: string;
  name: string;
  instructions: string;
};

export type ShopifySkillContext = {
  agentProfile: ShopifyAgentProfile;
  version: string;
  topics: string[];
  references: string[];
  context: string;
  unavailableFiles: string[];
};

type TopicRule = {
  topic: string;
  file: string;
  matches: RegExp[];
};

const SKILL_ROOT = path.resolve(
  process.cwd(),
  "shopify-toolkit",
  "skills",
  "shopify"
);

const MAX_CONTEXT_CHARS = 30000;
const MAX_REFERENCE_CHARS = 10500;

const TOPIC_RULES: TopicRule[] = [
  { topic: "liquid", file: "liquid.md", matches: [/\bliquid\b/i, /\btheme\b/i, /\bsections?\b/i, /\bsnippets?\b/i, /\btemplates?\b/i] },
  { topic: "admin", file: "admin.md", matches: [/admin(?:istrator)?\s+(?:graphql|api)/i, /\bproducts?\b/i, /\bvariants?\b/i, /\borders?\b/i, /\binventory\b/i, /\bcollections?\b/i, /\bfulfillment\b/i, /\brefunds?\b/i, /\bdrafts?\s+orders?\b/i, /\bshipping\b/i, /\blocations?\b/i, /\bstore settings?\b/i] },
  { topic: "custom-data", file: "custom-data.md", matches: [/metafields?/i, /metaobjects?/i, /custom data/i] },
  { topic: "customer", file: "customer.md", matches: [/customer account api/i, /customer accounts?/i, /new customer accounts?/i] },
  { topic: "functions", file: "functions.md", matches: [/shopify functions?/i, /\bdiscount function\b/i, /cart transform/i, /delivery customization/i, /payment customization/i, /fulfillment constraint/i] },
  { topic: "flow", file: "flow.md", matches: [/shopify flow/i, /\bworkflow automation\b/i, /\.flow\b/i] },
  { topic: "hydrogen", file: "hydrogen.md", matches: [/\bhydrogen\b/i] },
  { topic: "onboarding-dev", file: "onboarding-dev.md", matches: [/build (?:a |an )?shopify app/i, /create (?:a |an )?shopify app/i, /shopify app development/i, /scaffold(?:ing)? (?:a |an )?(?:shopify )?(?:app|theme)/i, /shopify development setup/i] },
  { topic: "onboarding-merchant", file: "onboarding-merchant.md", matches: [/create (?:a |an )?shopify store/i, /start (?:a |an )?shopify store/i, /open (?:a |an )?shopify store/i, /mock\.shop/i, /merchant onboarding/i] },
  { topic: "partner", file: "partner.md", matches: [/shopify partner api/i, /partner dashboard/i] },
  { topic: "payments-apps", file: "payments-apps.md", matches: [/payments? app api/i, /payment provider integration/i] },
  { topic: "polaris-app-home", file: "polaris-app-home.md", matches: [/\bpolaris\b/i, /admin app home/i, /embedded app interface/i] },
  { topic: "polaris-admin-extensions", file: "polaris-admin-extensions.md", matches: [/admin ui extension/i, /admin extension/i, /app home extension/i] },
  { topic: "polaris-checkout-extensions", file: "polaris-checkout-extensions.md", matches: [/checkout ui extension/i, /checkout extension/i] },
  { topic: "polaris-customer-account-extensions", file: "polaris-customer-account-extensions.md", matches: [/customer account extension/i, /order status extension/i] },
  { topic: "pos-ui", file: "pos-ui.md", matches: [/\bshopify pos\b/i, /point.of.sale/i, /pos ui/i] },
  { topic: "shopifyql", file: "shopifyql.md", matches: [/\bshopifyql\b/i, /sales analytics/i, /revenue report/i, /orders by channel/i] },
  { topic: "storefront-graphql", file: "storefront-graphql.md", matches: [/storefront graphql/i, /storefront api/i, /headless storefront/i, /cart api/i] },
  { topic: "use-shopify-cli", file: "use-shopify-cli.md", matches: [/shopify cli/i, /shopify store execute/i, /shopify store auth/i, /theme check/i] },
  { topic: "app-pricing", file: "app-pricing.md", matches: [/app pricing/i, /subscription plans? for (?:the )?app/i, /app billing/i, /usage based pricing/i] },
  { topic: "app-store-review", file: "app-store-review.md", matches: [/app store review/i, /app submission compliance/i, /app store requirements/i] },
];

const AGENT_PROFILES: Record<string, ShopifyAgentProfile> = {
  knowledge: {
    id: "shopify-knowledge",
    name: "Shopify Knowledge Agent",
    instructions: "Resolve Shopify platform questions using official documentation, versioned schemas, and the selected Shopify Toolkit references. Distinguish supported behavior from assumptions.",
  },
  theme: {
    id: "shopify-theme-development",
    name: "Shopify Theme Development Agent",
    instructions: "Specialize in Shopify Liquid, JSON templates, theme architecture, responsive theme work, accessibility, and theme QA. Trace real source relationships before recommending changes.",
  },
  products: {
    id: "shopify-product-management",
    name: "Shopify Product Management Agent",
    instructions: "Specialize in product, variant, media, pricing, SKU, barcode, inventory, collections, tags, and product metafield workflows. Verify API scope and mutation support before proposing live changes.",
  },
  orders: {
    id: "shopify-order-management",
    name: "Shopify Order Management Agent",
    instructions: "Specialize in order search, draft orders, fulfillment, tracking, cancellation, and refunds. Treat refunds and cancellation as sensitive operations requiring explicit approval.",
  },
  customers: {
    id: "shopify-customer-management",
    name: "Shopify Customer Management Agent",
    instructions: "Specialize in customer records, addresses, tags, metafields, and B2B company workflows. Verify current API support and required scopes before suggesting execution.",
  },
  store: {
    id: "shopify-store-configuration",
    name: "Shopify Store Configuration Agent",
    instructions: "Specialize in store settings, locations, inventory configuration, shipping/delivery, metafield definitions, and navigation. Separate API-supported operations from unsupported settings.",
  },
  qa: {
    id: "shopify-qa-testing",
    name: "Shopify QA and Testing Agent",
    instructions: "Specialize in schema validation, GraphQL validation, Liquid/theme checks, regression plans, accessibility, and runtime QA. Report validation only when a tool actually returned a successful result.",
  },
};

export function resolveShopifyAgentProfile(prompt: string): ShopifyAgentProfile {
  // Prefer the implementation domain when a theme-development request also
  // contains test/verification language; reserve QA for cross-domain QA tasks.
  if (/\b(?:liquid|theme|section|snippet|template|figma|srs|responsive page|theme editor|add to cart|cart drawer)\b/i.test(prompt)) return AGENT_PROFILES.theme;
  if (/\b(?:test|testing|validate|validation|qa|quality assurance|regression|audit|lint)\b/i.test(prompt)) return AGENT_PROFILES.qa;
  if (/\b(?:refund|refunds|cancel|cancellation|fulfillment|tracking|draft order|orders?)\b/i.test(prompt)) return AGENT_PROFILES.orders;
  if (/\b(?:product|products|variant|variants|sku|barcode|collection|pricing|import products)\b/i.test(prompt)) return AGENT_PROFILES.products;
  if (/\b(?:customer|customers|company|companies|b2b|address book)\b/i.test(prompt)) return AGENT_PROFILES.customers;
  if (/\b(?:store settings?|locations?|shipping|delivery|navigation|menu|inventory settings|metafield definitions)\b/i.test(prompt)) return AGENT_PROFILES.store;
  return AGENT_PROFILES.knowledge;
}

function selectTopics(prompt: string): string[] {
  const selected = new Set<string>();
  const isThemeDevelopmentRequest =
    /\b(?:liquid|theme|sections?|snippets?|templates?|add to cart|cart drawer|theme editor)\b/i.test(prompt);
  const explicitlyNeedsAdminContext =
    /\b(?:admin\s+(?:graphql|api)|graphql|metafields?|metaobjects?|api version|api scopes?)\b/i.test(prompt);

  for (const rule of TOPIC_RULES) {
    // Product/variant mentions in a theme task describe storefront source,
    // not necessarily Admin API work. Avoid loading unrelated Admin guidance.
    if (
      rule.topic === "admin" &&
      isThemeDevelopmentRequest &&
      !explicitlyNeedsAdminContext
    ) {
      continue;
    }

    if (rule.matches.some((pattern) => pattern.test(prompt))) {
      selected.add(rule.topic);
    }
  }

  if (/metafields?|metaobjects?/i.test(prompt)) selected.add("admin");
  if (
    /\b(?:product|variant|order|inventory|collection|fulfillment|refund|customer)\b/i.test(prompt) &&
    (!isThemeDevelopmentRequest || explicitlyNeedsAdminContext)
  ) {
    selected.add("admin");
  }
  if (/checkout.{0,30}(?:extension|ui)|(?:extension|ui).{0,30}checkout/i.test(prompt)) selected.add("polaris-checkout-extensions");
  if (/\b(?:customer account|order status)\b.{0,40}extension/i.test(prompt)) selected.add("polaris-customer-account-extensions");

  // The existing Horizon agent is theme-first, so use the Liquid skill as
  // the safe default when the request does not identify another Shopify API.
  if (selected.size === 0) selected.add("liquid");

  return [...selected];
}

function extractVersion(skillText: string): string {
  const match = skillText.match(/^\s+version:\s*["']?([^"'\r\n]+)["']?/m);
  return match?.[1]?.trim() || "unknown";
}

function extractRoutingRows(skillText: string, topics: string[]): string[] {
  return skillText
    .split(/\r?\n/)
    .filter((line) => topics.some((topic) => line.startsWith(`| \`${topic}\` |`)));
}

function appendBounded(parts: string[], value: string, budget: { remaining: number }): void {
  if (budget.remaining <= 0 || !value) return;
  const clipped = value.length > budget.remaining
    ? value.slice(0, budget.remaining) + "\n[Skill context truncated at the request budget.]"
    : value;
  parts.push(clipped);
  budget.remaining -= clipped.length;
}

export async function loadShopifySkillContext(prompt: string): Promise<ShopifySkillContext> {
  const topics = selectTopics(prompt);
  const unavailableFiles: string[] = [];
  const skillPath = path.join(SKILL_ROOT, "SKILL.md");

  let skillText = "";
  try {
    skillText = await readFile(skillPath, "utf8");
  } catch {
    unavailableFiles.push("skills/shopify/SKILL.md");
  }

  const version = skillText ? extractVersion(skillText) : "unknown";
  const references = [
    ...TOPIC_RULES
      .filter((rule) => topics.includes(rule.topic))
      .map((rule) => rule.file),
    ...(topics.includes("onboarding-merchant") ? ["onboarding-merchant.part2.md"] : []),
    ...(topics.includes("polaris-admin-extensions") ? ["polaris-admin-extensions.part2.md"] : []),
  ];
  const parts: string[] = [];
  const budget = { remaining: MAX_CONTEXT_CHARS };

  appendBounded(parts, [
    "OFFICIAL SHOPIFY AI TOOLKIT — CONSOLIDATED SHOPIFY SKILL",
    `Skill version: ${version}`,
    `Selected topics: ${topics.join(", ")}`,
    "",
    "Use the selected topic instructions and references as the source of Shopify-specific guidance.",
    "The original skill describes shell helpers for coding-agent hosts. This Gemini server does not assume a Bash or IDE-plugin runtime. Use dynamically discovered Shopify Dev MCP tools for documentation search, schema lookup, and supported validation. Do not claim that a search or validation ran unless the MCP call actually succeeded.",
    "The Shopify Dev MCP server is a developer documentation/schema/validation service. It does not authenticate to a merchant store and does not grant Admin API access.",
  ].join("\n"), budget);

  if (skillText) {
    const intro = skillText.slice(0, Math.min(2600, skillText.length));
    appendBounded(parts, "\n\n--- OFFICIAL SKILL OVERVIEW ---\n" + intro, budget);
    const rows = extractRoutingRows(skillText, topics);
    appendBounded(parts, "\n\n--- SELECTED OFFICIAL TOPIC ROUTING ---\n" + rows.join("\n"), budget);
  }

  const loadedReferences: string[] = [];
  for (const file of references) {
    if (budget.remaining <= 0) break;
    const relative = `references/${file}`;
    try {
      const content = await readFile(path.join(SKILL_ROOT, relative), "utf8");
      const clipped = content.length > MAX_REFERENCE_CHARS
        ? content.slice(0, MAX_REFERENCE_CHARS) + "\n[Reference truncated for the Gemini context budget.]"
        : content;
      appendBounded(parts, `\n\n--- SHOPIFY TOOLKIT ${relative} ---\n${clipped}`, budget);
      loadedReferences.push(relative);
    } catch {
      unavailableFiles.push(`skills/shopify/${relative}`);
    }
  }

  return {
    agentProfile: resolveShopifyAgentProfile(prompt),
    version,
    topics,
    references: loadedReferences,
    context: parts.join("\n"),
    unavailableFiles,
  };
}
