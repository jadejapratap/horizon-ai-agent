$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$routePath = Join-Path (Get-Location) 'src\app\api\agent\route.ts'
if (-not (Test-Path -LiteralPath $routePath)) {
  throw "Could not find route file: $routePath. Run this script from the project root."
}

$originalContent = [System.IO.File]::ReadAllText($routePath)
$newline = if ($originalContent.Contains("`r`n")) { "`r`n" } else { "`n" }
$content = $originalContent -replace "`r`n", "`n"

function Get-ExactCount([string]$Text, [string]$Needle) {
  return [regex]::Matches($Text, [regex]::Escape($Needle)).Count
}

# Ensure this is the expected post-verifier-patch route, and avoid silently
# applying the same range fix twice.
$requiredAnchors = @(
  @{ Name = 'CSS verifier'; Needle = 'function supportsStickyCssRuleClaim('; Count = 1 },
  @{ Name = 'sticky JS range builder'; Needle = 'function buildStickyScriptTargetRequests('; Count = 1 },
  @{ Name = 'sticky JS caller'; Needle = 'const referencedScripts = buildStickyScriptTargetRequests('; Count = 1 },
  @{ Name = 'filtered JS manifest input'; Needle = 'manifestFiles.filter((file) => !selectedFiles.has(file.file))'; Count = 1 },
  @{ Name = 'CSS verifier integration'; Needle = 'supportsStickyCssRuleClaim(claim.claim, evidenceItems)'; Count = 2 }
)
foreach ($item in $requiredAnchors) {
  $actual = Get-ExactCount -Text $content -Needle $item.Needle
  if ($actual -ne $item.Count) {
    throw "Expected $($item.Count) occurrence(s) for $($item.Name), found $actual. No changes were made."
  }
}
if ($content.Contains('reservedRequests: PlannerChunkRequest[] = []') -or
    $content.Contains('STICKY HEADER RANGE FIX: include planner-reserved JS ranges')) {
  throw 'The sticky-header range fix appears to be already applied. No changes were made.'
}

# CSS declarations may be quoted as separate snippets by the model. Validate
# each quoted declaration against the complete exact selector rule in a
# contiguous source segment that was actually sent to Gemini.
$cssFunctionPattern = '(?s)function supportsStickyCssRuleClaim\(.*?\n\}\n\n// A claim that IntersectionObserver'
$cssMatch = [regex]::Match($content, $cssFunctionPattern)
if (-not $cssMatch.Success) {
  throw 'Could not locate the complete existing supportsStickyCssRuleClaim function. No changes were made.'
}
$oldCssFunction = $cssMatch.Value -replace '\n\n// A claim that IntersectionObserver$', ''
$newCssFunction = @'
function supportsStickyCssRuleClaim(
  claim: string,
  evidenceItems: string[],
  sourceSegments: string[]
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
  const requiresStickyPosition = /\bposition\b.{0,32}\bsticky\b|\bposition\s*:\s*sticky\b/i.test(claim);
  const requiresTop = /\btop\b|\boffset\b/i.test(claim);
  const requiresZIndex = /\bz-index\b/i.test(claim);
  const normalizeCssEvidence = (value: string) =>
    value.replace(/\s+/g, " ").trim().toLowerCase();

  // sourceSegments contain the exact contiguous ranges sent to Gemini. Use
  // them to prove selector ownership and the complete rule; then require the
  // model's quoted declaration to be an exact fragment of that same rule.
  return sourceSegments.some((segment) => {
    const rule = selectorPattern.exec(segment);
    if (!rule) return false;

    const declarations = rule[1];
    const normalizedClaim = normalizeCssEvidence(claim);
    const getDeclarationValue = (property: string) => {
      const match = new RegExp(`(?:^|;)\\s*${property}\\s*:\\s*([^;]+)`, "i").exec(declarations);
      return match?.[1]?.trim() ?? null;
    };

    if (requiresStickyPosition && !/(?:^|;)\s*position\s*:\s*sticky\b/i.test(declarations)) return false;

    if (requiresTop) {
      const topValue = getDeclarationValue("top");
      if (!topValue || !normalizedClaim.includes(normalizeCssEvidence(topValue))) return false;
    }

    if (requiresZIndex) {
      const zIndexValue = getDeclarationValue("z-index");
      if (!zIndexValue || !normalizedClaim.includes(normalizeCssEvidence(zIndexValue))) return false;
    }

    const normalizedRule = normalizeCssEvidence(rule[0]);
    return evidenceItems.some((evidence) => {
      const normalizedEvidence = normalizeCssEvidence(evidence);
      return normalizedEvidence.length > 0 && normalizedRule.includes(normalizedEvidence);
    });
  });
}

// A claim that IntersectionObserver
'@
$newCssFunction = $newCssFunction -replace '\n\n// A claim that IntersectionObserver\n$', "`n`n// A claim that IntersectionObserver"
$content = $content.Substring(0, $cssMatch.Index) + $newCssFunction + $content.Substring($cssMatch.Index + $cssMatch.Length)

# Both atomic-proof and evidence-support paths must call the CSS verifier with
# the same contiguous source segments used by the other specialized checks.
$content = $content.Replace(
  'supportsStickyCssRuleClaim(claim.claim, evidenceItems)',
  'supportsStickyCssRuleClaim(claim.claim, evidenceItems, sourceSegments)'
)

# Make the sticky JS range builder aware of chunks already reserved by Gemini's
# planner. This prevents duplicate overlap while still allowing later ranges
# in the same assets/header.js file to be selected.
$signatureOld = @'
  referencedAssets: Set<string>,
  maxRequests = 2
): PlannerChunkRequest[] {
'@
$signatureNew = @'
  referencedAssets: Set<string>,
  maxRequests = 2,
  reservedRequests: PlannerChunkRequest[] = []
): PlannerChunkRequest[] {
'@
if ((Get-ExactCount -Text $content -Needle $signatureOld) -ne 1) {
  throw 'Could not uniquely locate the sticky JS builder signature. No changes were made.'
}
$content = $content.Replace($signatureOld, $signatureNew)

$scriptStart = $content.IndexOf('function buildStickyScriptTargetRequests(')
$scriptEndMarker = 'console.log("STICKY SCRIPT TARGET RANGE CANDIDATES:"'
$scriptEnd = $content.IndexOf($scriptEndMarker, $scriptStart)
if ($scriptStart -lt 0 -or $scriptEnd -lt 0) {
  throw 'Could not isolate the sticky JS builder body. No changes were made.'
}
$scriptBlock = $content.Substring($scriptStart, $scriptEnd - $scriptStart)
$knownRangesOld = '    const knownRanges = mergeRanges(manifest.retrievedRanges ?? []);'
$knownRangesNew = @'
    const plannerReservedRanges = reservedRequests
      .filter((request) => normalizePath(request.file).toLowerCase() === normalizePath(manifest.file).toLowerCase())
      .map((request) => {
        const startOffset = Math.max(0, Math.min(source.length, request.start_offset));
        const endOffset = Math.max(startOffset, Math.min(source.length, request.start_offset + request.max_chars));
        return { startOffset, endOffset };
      })
      .filter((range) => range.endOffset > range.startOffset);
    const knownRanges = mergeRanges([
      ...(manifest.retrievedRanges ?? []),
      ...plannerReservedRanges,
    ]);
'@
if ((Get-ExactCount -Text $scriptBlock -Needle $knownRangesOld) -ne 1) {
  throw 'Could not uniquely locate knownRanges inside the sticky JS builder. No changes were made.'
}
$scriptBlock = $scriptBlock.Replace($knownRangesOld, $knownRangesNew.TrimEnd())
$content = $content.Substring(0, $scriptStart) + $scriptBlock + $content.Substring($scriptEnd)

# Do not exclude a planner-selected JS file. Its selected offsets are now
# passed to the builder as reserved ranges, so the builder seeks the next gap.
$content = $content.Replace(
  '      manifestFiles.filter((file) => !selectedFiles.has(file.file)),',
  '      manifestFiles,'
)
$callerOld = '      Math.min(2, Math.max(0, requestLimit - selected.length))'
$callerNew = @'
      Math.min(2, Math.max(0, requestLimit - selected.length)),
      plannerRequests
'@
if ((Get-ExactCount -Text $content -Needle $callerOld) -ne 1) {
  throw 'Could not uniquely locate the sticky JS caller maxRequests argument. No changes were made.'
}
$content = $content.Replace($callerOld, $callerNew.TrimEnd())

# Add a one-time marker next to the caller so subsequent runs fail safely.
$commentAnchor = '    const referencedScripts = buildStickyScriptTargetRequests('
$content = $content.Replace(
  $commentAnchor,
  '    // STICKY HEADER RANGE FIX: include planner-reserved JS ranges and seek the next uncovered source gap.' + "`n" + $commentAnchor
)

# Validate all expected edits before creating a backup or writing the route.
if ((Get-ExactCount -Text $content -Needle 'supportsStickyCssRuleClaim(claim.claim, evidenceItems, sourceSegments)') -ne 2) {
  throw 'Final validation failed: expected two CSS verifier calls with sourceSegments. No file was written.'
}
if (-not $content.Contains('reservedRequests: PlannerChunkRequest[] = []')) {
  throw 'Final validation failed: reserved JS range parameter is missing. No file was written.'
}
if (-not $content.Contains('...plannerReservedRanges')) {
  throw 'Final validation failed: planner ranges are not merged into the JS known ranges. No file was written.'
}
if (-not $content.Contains('      manifestFiles,')) {
  throw 'Final validation failed: sticky JS builder is not receiving all manifest files. No file was written.'
}
if (-not [regex]::IsMatch($content, 'plannerRequests\s*\)\s*;')) {
  throw 'Final validation failed: planner requests are not passed to the sticky JS builder. No file was written.'
}

$backupPath = "$routePath.before-sticky-header-range-fix.bak"
if (Test-Path -LiteralPath $backupPath) {
  throw "Backup already exists at $backupPath. No file was written; inspect the route and backup first."
}

Copy-Item -LiteralPath $routePath -Destination $backupPath
$outputContent = if ($newline -eq "`r`n") { $content -replace "(?<!`r)`n", "`r`n" } else { $content }
[System.IO.File]::WriteAllText($routePath, $outputContent, [System.Text.UTF8Encoding]::new($false))

Write-Host 'Sticky-header CSS evidence and JS range fix applied.' -ForegroundColor Green
Write-Host "Backup: $backupPath"
Write-Host 'CSS verification now checks each cited fragment against the complete selector rule in a contiguous sent source segment.'
Write-Host 'Sticky JS targeting now examines planner-selected files and skips already reserved ranges to retrieve a later uncovered range.'
Write-Host 'No theme source files or existing CSS/JS targeting patterns were changed.'
