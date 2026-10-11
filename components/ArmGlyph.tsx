/**
 * One mark per arm, used in every chart, table, and legend so an arm looks the
 * same everywhere. Shape is the model tier (circle, square, triangle, cheapest
 * to most capable); fill is the effort level (hollow, half, solid). The encoding
 * is monochrome on purpose: the single signal colour is kept for the router.
 */
const EFFORT_FILL: Record<string, number> = { low: 0, medium: 0.45, high: 1, xhigh: 1, max: 1 };

interface GlyphProps {
  model: string;
  effort: string;
  size?: number;
  /** Dominated arms are drawn in the muted colour. */
  muted?: boolean;
  className?: string;
}

export function glyphPath(model: string, r: number): string {
  if (model === "sonnet") return `M${-r},${-r}H${r}V${r}H${-r}Z`;
  if (model === "opus") {
    const h = r * 1.15;
    return `M0,${-h}L${r * 1.2},${h * 0.85}H${-r * 1.2}Z`;
  }
  return `M${-r},0A${r},${r} 0 1,0 ${r},0A${r},${r} 0 1,0 ${-r},0Z`;
}

/** The glyph as SVG children, centred on the origin, for use inside a chart. */
export function GlyphMark({ model, effort, r = 5, muted = false }: GlyphProps & { r?: number }) {
  const ink = muted ? "var(--muted)" : "var(--fg)";
  const fill = EFFORT_FILL[effort] ?? 1;
  const d = glyphPath(model, r);
  return (
    <>
      <path d={d} fill="var(--bg)" />
      <path d={d} fill={ink} fillOpacity={fill} stroke={ink} strokeWidth={1.5} strokeLinejoin="round" />
      {(effort === "xhigh" || effort === "max") && (
        <circle r={r + 3.5} fill="none" stroke={ink} strokeWidth={1} strokeDasharray={effort === "max" ? undefined : "2 2"} />
      )}
    </>
  );
}

export function ArmGlyph({ model, effort, size = 12, muted = false, className = "" }: GlyphProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="-8 -8 16 16"
      aria-hidden
      className={`inline-block shrink-0 align-[-1px] ${className}`}
    >
      <GlyphMark model={model} effort={effort} r={5} muted={muted} />
    </svg>
  );
}
