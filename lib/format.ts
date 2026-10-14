/** Number formatting shared by the dashboard, the explorer, and the playground. */

const whole = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

export function usd(amount: number): string {
  if (amount === 0) return "$0";
  const abs = Math.abs(amount);
  if (abs < 0.001) return `$${amount.toFixed(5)}`;
  if (abs < 0.1) return `$${amount.toFixed(4)}`;
  if (abs < 10) return `$${amount.toFixed(2)}`;
  if (abs < 1000) return `$${amount.toFixed(1)}`;
  return `$${whole.format(amount)}`;
}

export function pct(fraction: number, digits = 1): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function duration(ms: number | null): string {
  if (ms === null) return "n/a";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)} s`;
  return `${Math.round(ms / 1000)} s`;
}

export function count(value: number): string {
  return whole.format(value);
}

const MODEL_NAMES: Record<string, string> = { haiku: "Haiku", sonnet: "Sonnet", opus: "Opus" };

/** "opus-high" becomes "Opus · high". */
export function armLabel(key: string): string {
  const [model, effort] = key.split("-");
  return `${MODEL_NAMES[model] ?? model} · ${effort}`;
}

/** "opus-high" becomes "Opus, high effort": the plain form for headlines, the chart, and anywhere a stranger reads first. */
export function armName(key: string): string {
  const [model, effort] = key.split("-");
  return `${MODEL_NAMES[model] ?? model}, ${effort} effort`;
}

/** "opus-high" becomes "O·high", for tight table headers. */
export function armCode(key: string): string {
  const [model, effort] = key.split("-");
  return `${model.charAt(0).toUpperCase()}·${effort === "medium" ? "med" : effort}`;
}

export function armParts(key: string): { model: string; effort: string } {
  const [model, effort] = key.split("-");
  return { model, effort };
}
