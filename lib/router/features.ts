/**
 * Prompt features for the router: the TypeScript port of `evals/cascade/features.py`.
 *
 * The router is trained in Python and served here, so the two implementations must
 * agree bit for bit. Both interpret the same feature spec (embedded in
 * `data/router.json`) under the same rules:
 *
 * - CRLF and CR become LF before anything is counted.
 * - Lower-casing touches only A-Z. `toLowerCase()` would also fold characters such
 *   as the Kelvin sign, which Python's ASCII-only matching does not.
 * - Length is counted in code points (iterating a string yields code points),
 *   matching Python's `len`, not UTF-16 units.
 * - Patterns run without the `u` flag, so `\b` and classes are ASCII, as in Python
 *   with `re.ASCII`. The spec avoids `.` and negated classes, which would count an
 *   astral character twice here.
 * - A feature value is `n / (n + k)` or `n / (chars + 1)`: integer counts, one
 *   addition, one division, all exact in IEEE 754. No `log`, no `exp`.
 *
 * `tests/router-parity.test.ts` checks this file against vectors exported by Python.
 */

export interface FeatureSpec {
  name: string;
  label: string;
  kind: "chars" | "count" | "lines";
  source?: "raw" | "lower";
  patterns?: string[];
  pattern?: string;
  transform: "sat" | "ratio";
  k?: number;
}

export function normalize(text: string): string {
  return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

export function asciiLower(text: string): string {
  return text.replace(/[A-Z]/g, (letter) => String.fromCharCode(letter.charCodeAt(0) + 32));
}

function codePoints(text: string): number {
  // Spreading a string iterates by code point, so an astral character counts once.
  return [...text].length;
}

const globalCache = new Map<string, RegExp>();
const lineCache = new Map<string, RegExp>();

function countMatches(patterns: string[], text: string): number {
  let total = 0;
  for (const pattern of patterns) {
    let compiled = globalCache.get(pattern);
    if (!compiled) {
      compiled = new RegExp(pattern, "g");
      globalCache.set(pattern, compiled);
    }
    // String.prototype.match with a global regex ignores lastIndex and returns every match.
    total += text.match(compiled)?.length ?? 0;
  }
  return total;
}

function countLines(pattern: string, raw: string): number {
  let compiled = lineCache.get(pattern);
  if (!compiled) {
    compiled = new RegExp(`^(?:${pattern})`);
    lineCache.set(pattern, compiled);
  }
  let total = 0;
  for (const line of raw.split("\n")) {
    if (compiled.test(line.replace(/^[ \t]+/, ""))) total++;
  }
  return total;
}

/** One value per feature, each in [0, 1). */
export function extractFeatures(spec: FeatureSpec[], prompt: string): number[] {
  const raw = normalize(prompt);
  const lower = asciiLower(raw);
  const chars = codePoints(raw);
  return spec.map((feature) => {
    let n: number;
    if (feature.kind === "chars") n = chars;
    else if (feature.kind === "count") {
      n = countMatches(feature.patterns ?? [], feature.source === "lower" ? lower : raw);
    } else n = countLines(feature.pattern ?? "", raw);
    return feature.transform === "sat" ? n / (n + (feature.k ?? 1)) : n / (chars + 1);
  });
}
