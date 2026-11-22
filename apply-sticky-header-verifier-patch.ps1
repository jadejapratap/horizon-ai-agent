$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$routePath = Join-Path (Get-Location) 'src\app\api\agent\route.ts'
if (-not (Test-Path -LiteralPath $routePath)) {
  throw "Could not find route file: $routePath. Run this script from the project root."
}

$content = [System.IO.File]::ReadAllText($routePath)

if ($content.Contains('function supportsLiquidSectionSchemaSettingDefinitionClaim(') -or
    $content.Contains('function supportsStickyCssRuleClaim(') -or
    $content.Contains('function supportsStickyHeaderObserverClaim(')) {
  throw 'At least one sticky-header verifier already exists. No changes were made; inspect the route before applying anything again.'
}

function Get-ExactCount([string]$Text, [string]$Needle) {
  return [regex]::Matches($Text, [regex]::Escape($Needle)).Count
}

$helperAnchor = '// Product-card currency settings are simple truthy Liquid guards rather than'
$jsonIntegrationAnchor = '          supportsJsonSettingDefinitionClaim(claim.claim, evidenceItems) ||'
$currencySpecializedAnchor = '          (/\buse_currency\b/.test(claimTextLower) &&'
$plannerAnchor = '      .filter((request) => !settingDefinitionFiles.has(request.file))'

$expected = @(
  @{ Name = 'helper insertion point'; Needle = $helperAnchor; Count = 1 },
  @{ Name = 'JSON verifier integration points'; Needle = $jsonIntegrationAnchor; Count = 2 },
  @{ Name = 'specialized-evidence insertion point'; Needle = $currencySpecializedAnchor; Count = 1 },
  @{ Name = 'planner filter insertion point'; Needle = $plannerAnchor; Count = 1 }
)
foreach ($item in $expected) {
  $actual = Get-ExactCount -Text $content -Needle $item.Needle
  if ($actual -ne $item.Count) {
    throw "Expected $($item.Count) occurrence(s) for $($item.Name), found $actual. No changes were made."
  }
}

$helpers = @'
// Section-level Shopify settings are JSON objects embedded inside a Liquid
// `{% schema %}` block. Verify the exact parsed object by its setting id and
// only the properties/values that the claim actually asserts. These settings
// must be attributed to their owning Liquid section file.
function supportsLiquidSectionSchemaSettingDefinitionClaim(
  claim: string,
  evidenceItems: string[],
  file: string
): boolean {
  if (
    !/^sections\/[a-z0-9_./-]+\.liquid$/i.test(normalizePath(file)) ||
    !/enable_sticky_header/i.test(claim)
  ) return false;

  const normalizedClaim = claim.toLowerCase();

  return evidenceItems.some((evidence) => {
    let parsed: unknown;
    const trimmedEvidence = evidence.trim();
    try {
      parsed = JSON.parse(trimmedEvidence);
    } catch {
      // Accept a complete property fragment only when wrapping it in braces
      // produces valid JSON. Do not repair malformed values or nested data.
      try {
        parsed = JSON.parse(`{${trimmedEvidence.replace(/,\s*$/, "")}}`);
      } catch {
        return false;
      }
    }

    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return false;
    const definition = parsed as Record<string, unknown>;
    if (definition.id !== "enable_sticky_header") return false;

    if (/\btype\b/i.test(claim)) {
      if (typeof definition.type !== "string" ||
          !normalizedClaim.includes(definition.type.toLowerCase())) return false;
    }
    if (/\blabel\b/i.test(claim)) {
      if (typeof definition.label !== "string" ||
          !normalizedClaim.includes(definition.label.toLowerCase())) return false;
    }
    if (/\bdefault\b/i.test(claim)) {
      if (typeof definition.default !== "string" ||
          !normalizedClaim.includes(definition.default.toLowerCase())) return false;
    }
    if (/\boptions?\b/i.test(claim)) {
      if (!Array.isArray(definition.options) || definition.options.length === 0) return false;
      const optionValues = definition.options.map((option) => {
        if (!option || typeof option !== "object" || Array.isArray(option)) return null;
        const value = (option as Record<string, unknown>).value;
        return typeof value === "string" ? value.toLowerCase() : null;
      });
      if (optionValues.some((value) => value === null) ||
          !optionValues.every((value) => normalizedClaim.includes(value!))) return false;
    }
    return true;
  });
}

// Sticky CSS claims require the named selector and the claimed declarations
// to occur together in one quoted rule. Generic mentions of `sticky` or `top`
// cannot establish which selector owns those declarations.
function supportsStickyCssRuleClaim(
  claim: string,
  evidenceItems: string[]
): boolean {
  const claimWithoutThemePaths = claim.replace(
    /\b(?:assets|blocks|config|layout|sections|snippets|templates)\/[A-Za-z0-9_./-]+\.(?:liquid|json|js|css)\b/gi,
    " "
  );
  const selectorMatch = claimWithoutThemePaths.match(/\.(?:[A-Za-z_-][A-Za-z0-9_-]*)/);
  if (!selectorMatch || !/\b(?:position|sticky|top|offset|z-index)\b/i.test(claim)) return false;

  const selector = selectorMatch[0];
  const selectorPattern = new RegExp(
    `${escapeRegExp(selector)}\\s*\\{([^{}]*)\\}`,
    "i"
  );
  const requiresStickyPosition = /\bsticky\s+position\b|\bposition\s*:\s*sticky\b/i.test(claim);
  const requiresTop = /\btop\b|\boffset\b/i.test(claim);
  const requiresZIndex = /\bz-index\b/i.test(claim);

  return evidenceItems.some((evidence) => {
    const rule = selectorPattern.exec(evidence);
    if (!rule) return false;
    const declarations = rule[1];
    if (requiresStickyPosition && !/(?:^|;)\s*position\s*:\s*sticky\b/i.test(declarations)) return false;
    if (requiresTop && !/(?:^|;)\s*top\s*:\s*[^;]+/i.test(declarations)) return false;
    if (requiresZIndex && !/(?:^|;)\s*z-index\s*:\s*[^;]+/i.test(declarations)) return false;
    return true;
  });
}

// A claim that IntersectionObserver is installed by #observeStickyPosition
// needs both the constructor in cited evidence and the constructor inside
// that same method in the contiguous source segment sent to the model.
function supportsStickyHeaderObserverClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
): boolean {
  if (!/#observeStickyPosition/i.test(claim) || !/\bIntersectionObserver\b/i.test(claim)) return false;
  if (!evidenceItems.some((evidence) => /new\s+IntersectionObserver\s*\(/i.test(evidence))) return false;

  const methodPattern = /#observeStickyPosition\s*=\s*\([^)]*\)\s*=>\s*\{/;
  return sourceSegments.some((segment) => {
    const method = methodPattern.exec(segment);
    if (!method) return false;
    const bodyStart = method.index + method[0].length;
    const afterMethodStart = segment.slice(bodyStart);
    const nextPrivateMethod = afterMethodStart.search(/\n\s*#[A-Za-z_$][A-Za-z0-9_$]*\s*=/);
    const methodBody = nextPrivateMethod >= 0
      ? afterMethodStart.slice(0, nextPrivateMethod)
      : afterMethodStart.slice(0, 12000);
    return /new\s+IntersectionObserver\s*\(/i.test(methodBody);
  });
}

'@

$content = $content.Replace($helperAnchor, $helpers + $helperAnchor)

$verifierCalls = @'
          supportsJsonSettingDefinitionClaim(claim.claim, evidenceItems) ||
          supportsLiquidSectionSchemaSettingDefinitionClaim(claim.claim, evidenceItems, file) ||
          supportsStickyCssRuleClaim(claim.claim, evidenceItems) ||
          supportsStickyHeaderObserverClaim(claim.claim, evidenceItems, sourceSegments) ||
'@
$verifierBlock = ($verifierCalls -replace '^\r?\n', '').TrimEnd()
$content = $content.Replace($jsonIntegrationAnchor, $verifierBlock)

$specializedChecks = @'
          (/enable_sticky_header/i.test(claimTextLower) && /\b(?:schema|setting|select|options?)\b/.test(claimTextLower)) ||
          (/\.sticky-content\b/i.test(claimTextLower) && /\b(?:position|sticky|top|offset|z-index)\b/.test(claimTextLower)) ||
          (/#observestickyposition/i.test(claimTextLower) && /\bintersectionobserver\b/.test(claimTextLower)) ||
'@
$specializedBlock = ($specializedChecks -replace '^\r?\n', '').TrimEnd()
$content = $content.Replace($currencySpecializedAnchor, $specializedBlock + "`r`n" + $currencySpecializedAnchor)

$plannerFilter = @'
      .filter((request) => !settingDefinitionFiles.has(request.file))
      // The sticky-header toggle is defined in sections/header.liquid's
      // `{% schema %}` block, not in the global theme settings schema.
      // Avoid spending the limited on-demand budget on that unrelated file.
      .filter((request) => !(
        availableCoverageDomains.includes("sticky-header") &&
        normalizePath(request.file).toLowerCase() === "config/settings_schema.json"
      ))
'@
$plannerBlock = ($plannerFilter -replace '^\r?\n', '').TrimEnd()
$content = $content.Replace($plannerAnchor, $plannerBlock)

$backupPath = "$routePath.before-sticky-header-verifier.bak"
if (Test-Path -LiteralPath $backupPath) {
  throw "Backup already exists at $backupPath. No file was written; inspect the route and backup first."
}

# Final sanity checks before touching the user's route.
foreach ($required in @(
  'function supportsLiquidSectionSchemaSettingDefinitionClaim(',
  'function supportsStickyCssRuleClaim(',
  'function supportsStickyHeaderObserverClaim('
)) {
  if (-not $content.Contains($required)) { throw "Patch validation failed: missing $required" }
}
if ((Get-ExactCount -Text $content -Needle 'supportsLiquidSectionSchemaSettingDefinitionClaim(claim.claim, evidenceItems, file)') -ne 2) {
  throw 'Patch validation failed: expected both verifier integration points. No file was written.'
}
if (-not $content.Contains('normalizePath(request.file).toLowerCase() === "config/settings_schema.json"')) {
  throw 'Patch validation failed: planner filter was not inserted. No file was written.'
}

Copy-Item -LiteralPath $routePath -Destination $backupPath
[System.IO.File]::WriteAllText($routePath, $content, [System.Text.UTF8Encoding]::new($false))

Write-Host 'Sticky-header verifier patch applied.' -ForegroundColor Green
Write-Host "Backup: $backupPath"
Write-Host 'The existing sticky CSS/JS target-range builders were not changed.'
Write-Host 'Next: run the sticky-header test prompt and share the terminal result.'
