"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArmGlyph, GlyphMark } from "@/components/ArmGlyph";
import { armName, duration, pct, usd } from "@/lib/format";

export interface ChartArm {
  key: string;
  model: string;
  effort: string;
  /** Dollars per 1,000 requests at list price. */
  cost: number;
  quality: number;
  ci: [number, number];
  frontier: boolean;
  p50: number | null;
}

export interface RouterPoint {
  threshold: number;
  cost: number;
  quality: number;
}

interface Props {
  arms: ChartArm[];
  /** The router's cross-validated cost and quality at each threshold, cheapest first. */
  curve: RouterPoint[];
  selected: number;
  synthetic: boolean;
}

const X_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500];

function tickLabel(value: number): string {
  return value < 1 ? `$${value.toFixed(2)}` : `$${value}`;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}
const overlaps = (a: Box, b: Box) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** Quality-versus-cost scatter with the Pareto frontier and the router's operating curve. */
export function ParetoChart({ arms, curve, selected, synthetic }: Props) {
  const frame = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(720);
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const node = frame.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      setWidth(Math.max(280, Math.round(entry.contentRect.width)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const compact = width < 520;
  const height = compact ? 300 : Math.min(520, Math.round(width * 0.72));
  const margin = { top: 26, right: compact ? 10 : 18, bottom: 40, left: compact ? 38 : 46 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const layout = useMemo(() => {
    const costs = [...arms.map((a) => a.cost), ...curve.map((p) => p.cost)];
    const lo = Math.log10(Math.min(...costs) / 1.8);
    const hi = Math.log10(Math.max(...costs) * 1.6);
    const floor = Math.max(
      0,
      Math.floor((Math.min(...arms.map((a) => a.ci[0]), ...curve.map((p) => p.quality)) - 0.02) * 10) / 10,
    );
    // Rounded to hundredths: Node and the browser can disagree on the last bit of log10,
    // and the server-rendered SVG must match the client's exactly or hydration fails.
    const r2 = (v: number) => Math.round(v * 100) / 100;
    const x = (cost: number) => r2(margin.left + ((Math.log10(cost) - lo) / (hi - lo)) * innerW);
    const y = (quality: number) => r2(margin.top + (1 - (quality - floor) / (1 - floor)) * innerH);
    const yTicks: number[] = [];
    for (let t = floor; t <= 1.0001; t += 0.1) yTicks.push(Math.round(t * 10) / 10);
    const xTicks = X_TICKS.filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi);

    // Label placement. Labels are plain words ("Sonnet, medium effort"), which are long, and
    // the expensive arms crowd into one corner. So each label tries spots next to its point
    // first, then further out along eight directions, keeping the first that is inside the
    // plot and hits no marker, no router dot, and no other label. A label placed away from
    // its point gets a thin leader line. Crowded points are placed first.
    const charW = compact ? 5.5 : 6.1;
    const labelH = compact ? 11 : 12;
    const points = arms.map((arm) => ({ arm, px: x(arm.cost), py: y(arm.quality) }));
    const markers: Box[] = points.map((p) => ({ x: p.px - 8, y: p.py - 8, w: 16, h: 16 }));
    const taken: Box[] = [
      ...markers,
      ...curve.map((p) => ({ x: x(p.cost) - 4, y: y(p.quality) - 4, w: 8, h: 8 })),
    ];
    const inside = (b: Box) =>
      b.x >= margin.left && b.x + b.w <= width - 2 && b.y >= margin.top - 10 && b.y + b.h <= height - margin.bottom;
    const near = (p: (typeof points)[number]) =>
      points.filter((q) => q !== p && Math.hypot(q.px - p.px, q.py - p.py) < 60).length;
    const order = [...points.keys()].sort((a, b) => near(points[b]) - near(points[a]) || a - b);
    const labels: { text: string; x: number; y: number; leader: [number, number, number, number] | null }[] =
      new Array(points.length);
    for (const i of order) {
      const p = points[i];
      const text = compact ? armName(p.arm.key).replace(" effort", "") : armName(p.arm.key);
      const w = text.length * charW + 4;
      const h = labelH;
      const options: Box[] = [];
      for (const d of [0, 10, 22, 36, 52, 70]) {
        const k = d * 0.75;
        const right: Box = { x: p.px + 10 + d, y: p.py - h / 2, w, h };
        const left: Box = { x: p.px - 10 - d - w, y: p.py - h / 2, w, h };
        const belowRight: Box = { x: p.px + 7 + k, y: p.py + 5 + k, w, h };
        const aboveLeft: Box = { x: p.px - 7 - k - w, y: p.py - 5 - h - k, w, h };
        const belowLeft: Box = { x: p.px - 7 - k - w, y: p.py + 5 + k, w, h };
        const aboveRight: Box = { x: p.px + 7 + k, y: p.py - 5 - h - k, w, h };
        const below: Box = { x: p.px - w / 2, y: p.py + 10 + d, w, h };
        const above: Box = { x: p.px - w / 2, y: p.py - 10 - h - d, w, h };
        // The frontier climbs to the upper right, so below-right of a frontier point is usually clear.
        options.push(
          ...(p.arm.frontier
            ? [belowRight, right, aboveLeft, left, below, above, belowLeft, aboveRight]
            : [right, belowRight, left, below, above, belowLeft, aboveLeft, aboveRight]),
        );
      }
      const chosen =
        options.find((b) => inside(b) && !taken.some((t) => t !== markers[i] && overlaps(b, t))) ?? options[0];
      taken.push(chosen);
      // A leader runs from the marker's edge to the nearest point of a label that is not touching it.
      const cx = Math.min(Math.max(p.px, chosen.x), chosen.x + chosen.w);
      const cy = Math.min(Math.max(p.py, chosen.y), chosen.y + chosen.h);
      const gap = Math.hypot(cx - p.px, cy - p.py);
      const leader: [number, number, number, number] | null =
        gap > 16 ? [p.px + ((cx - p.px) / gap) * 8, p.py + ((cy - p.py) / gap) * 8, cx, cy] : null;
      labels[i] = { text, x: chosen.x + 2, y: chosen.y + h - 2.5, leader };
    }
    return { x, y, xTicks, yTicks, points, labels, taken };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arms, curve, width, height]);

  const { x, y, xTicks, yTicks, points, labels, taken } = layout;
  const frontier = points.filter((p) => p.arm.frontier).sort((a, b) => a.arm.cost - b.arm.cost);
  const frontierPath = frontier.map((p, i) => `${i ? "L" : "M"}${p.px.toFixed(1)},${p.py.toFixed(1)}`).join("");
  const curvePath = curve
    .map((p, i) => `${i ? "L" : "M"}${x(p.cost).toFixed(1)},${y(p.quality).toFixed(1)}`)
    .join("");
  const operating = curve[selected];
  const ox = x(operating.cost);
  const oy = y(operating.quality);

  // The router's marker moves with the threshold, so its label looks for a free corner each time.
  const routerLabel = (() => {
    const w = 46;
    const h = 12;
    const corners: Box[] = [
      { x: ox - 12 - w, y: oy - 20, w, h },
      { x: ox + 12, y: oy + 8, w, h },
      { x: ox + 12, y: oy - 20, w, h },
      { x: ox - 12 - w, y: oy + 8, w, h },
      { x: ox - w / 2, y: oy - 27, w, h },
      { x: ox - w / 2, y: oy + 15, w, h },
    ];
    const inside = (b: Box) =>
      b.x >= margin.left && b.x + b.w <= width - 2 && b.y >= 0 && b.y + b.h <= height - margin.bottom;
    const free = corners.find((b) => inside(b) && !taken.some((t) => overlaps(b, t))) ?? corners[0];
    return { dx: free.x - ox, dy: free.y - oy + 9.5 };
  })();

  const tip = (() => {
    if (active === "router") {
      return {
        px: ox,
        py: oy,
        title: `Cascade at ${pct(operating.threshold, 0)} sure`,
        rows: [
          ["tasks passed", pct(operating.quality)],
          ["per 1,000 requests", usd(operating.cost)],
        ],
      };
    }
    const point = points.find((p) => p.arm.key === active);
    if (!point) return null;
    const a = point.arm;
    return {
      px: point.px,
      py: point.py,
      title: armName(a.key),
      rows: [
        ["tasks passed", pct(a.quality)],
        ["likely range", `${pct(a.ci[0], 0)} to ${pct(a.ci[1], 0)}`],
        ["per 1,000 requests", usd(a.cost)],
        ["typical wait", duration(a.p50)],
        ["best value", a.frontier ? "yes" : "no, beaten"],
      ],
    };
  })();

  return (
    <figure className="m-0">
      <div ref={frame} className="relative w-full" onMouseLeave={() => setActive(null)}>
        <svg
          width={width}
          height={height}
          viewBox={`0 0 ${width} ${height}`}
          className="block h-auto max-w-full font-mono"
          role="group"
          aria-label="Share of tasks passed against cost per 1,000 requests for every Claude model and effort setting, with Cascade's curve"
        >
          {yTicks.map((t) => (
            <g key={`y${t}`}>
              <line x1={margin.left} x2={width - margin.right} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
              <text x={margin.left - 6} y={y(t) + 3.5} textAnchor="end" fontSize="10" fill="var(--muted)">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {xTicks.map((t) => {
            const major = /^1(0*)$/.test(String(t)) || /^0\.0*1$/.test(String(t));
            return (
              <g key={`x${t}`}>
                <line
                  x1={x(t)}
                  x2={x(t)}
                  y1={margin.top}
                  y2={height - margin.bottom}
                  stroke="var(--grid)"
                  strokeDasharray={major ? undefined : "2 4"}
                />
                {(major || !compact) && (
                  <text x={x(t)} y={height - margin.bottom + 14} textAnchor="middle" fontSize="10" fill="var(--muted)">
                    {tickLabel(t)}
                  </text>
                )}
              </g>
            );
          })}
          <line
            x1={margin.left}
            x2={width - margin.right}
            y1={height - margin.bottom}
            y2={height - margin.bottom}
            stroke="var(--border)"
          />
          <text x={margin.left + innerW / 2} y={height - 6} textAnchor="middle" fontSize="10" fill="var(--muted)">
            cost per 1,000 requests (log scale) →
          </text>
          <text x={margin.left - (compact ? 30 : 38)} y={11} fontSize="10" fill="var(--muted)">
            ↑ tasks passed
          </text>

          {/* 95% intervals: how far each arm's pass rate could move with a different task sample. */}
          {points.map((p) => (
            <line
              key={`ci-${p.arm.key}`}
              x1={p.px}
              x2={p.px}
              y1={y(p.arm.ci[0])}
              y2={y(p.arm.ci[1])}
              stroke="var(--muted)"
              strokeOpacity={0.35}
            />
          ))}

          <path d={frontierPath} fill="none" stroke="var(--fg)" strokeWidth={1.25} />
          <path d={curvePath} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="1 4" strokeLinecap="round" />
          {curve.map((p) => (
            <circle key={p.threshold} cx={x(p.cost)} cy={y(p.quality)} r={1.6} fill="var(--accent)" />
          ))}

          {points.map((p) => (
            <g
              key={p.arm.key}
              transform={`translate(${p.px.toFixed(1)},${p.py.toFixed(1)})`}
              tabIndex={0}
              role="img"
              aria-label={`${armName(p.arm.key)}: ${pct(p.arm.quality)} of tasks passed at ${usd(p.arm.cost)} per 1,000 requests${p.arm.frontier ? ", on the best-value line" : ", beaten by a cheaper or better option"}`}
              onMouseEnter={() => setActive(p.arm.key)}
              onFocus={() => setActive(p.arm.key)}
              onBlur={() => setActive(null)}
              className="cursor-default outline-none [&:focus-visible>circle]:opacity-100"
            >
              <circle r={11} fill="none" stroke="var(--accent)" strokeWidth={2} className="opacity-0" />
              <circle r={11} fill="transparent" />
              <GlyphMark model={p.arm.model} effort={p.arm.effort} r={compact ? 4.5 : 5.5} muted={!p.arm.frontier} />
            </g>
          ))}

          <g
            className="glide outline-none [&:focus-visible>circle:first-child]:stroke-[3]"
            style={{ transform: `translate(${ox.toFixed(1)}px, ${oy.toFixed(1)}px)` }}
            tabIndex={0}
            role="img"
            aria-label={`Cascade at ${pct(operating.threshold, 0)} sure: ${pct(operating.quality)} of tasks passed at ${usd(operating.cost)} per 1,000 requests`}
            onMouseEnter={() => setActive("router")}
            onFocus={() => setActive("router")}
            onBlur={() => setActive(null)}
          >
            <circle r={9} fill="var(--bg)" stroke="var(--accent)" strokeWidth={2} />
            <circle r={3.5} fill="var(--accent)" />
            <text
              x={routerLabel.dx}
              y={routerLabel.dy}
              fontSize="10"
              fontWeight={600}
              fill="var(--accent)"
              stroke="var(--bg)"
              strokeWidth={3}
              paintOrder="stroke"
            >
              Cascade
            </text>
          </g>

          <g aria-hidden pointerEvents="none">
            {points.map((p, i) =>
              labels[i].leader ? (
                <line
                  key={`leader-${p.arm.key}`}
                  x1={labels[i].leader[0]}
                  y1={labels[i].leader[1]}
                  x2={labels[i].leader[2]}
                  y2={labels[i].leader[3]}
                  stroke="var(--muted)"
                  strokeOpacity={0.6}
                  strokeWidth={0.75}
                />
              ) : null,
            )}
            {points.map((p, i) => (
              <text
                key={p.arm.key}
                x={labels[i].x}
                y={labels[i].y}
                fontSize={compact ? 9 : 10}
                fill={p.arm.frontier ? "var(--fg)" : "var(--muted)"}
                stroke="var(--bg)"
                strokeWidth={3}
                paintOrder="stroke"
              >
                {labels[i].text}
              </text>
            ))}
          </g>
        </svg>

        {tip && (
          <div
            role="tooltip"
            className="pointer-events-none absolute z-10 w-52 border border-border bg-surface p-2 font-mono text-[11px] shadow-lg"
            style={{
              left: Math.min(Math.max(tip.px - 104, 0), width - 208),
              top: tip.py > height / 2 ? tip.py - 96 : tip.py + 18,
            }}
          >
            <p className="mb-1 font-semibold">{tip.title}</p>
            {tip.rows.map(([name, value]) => (
              <p key={name} className="flex justify-between gap-2 text-muted">
                <span>{name}</span>
                <span className="tabular text-fg">{value}</span>
              </p>
            ))}
            {synthetic && <p className="mt-1 text-accent">simulated</p>}
          </div>
        )}
      </div>

      <figcaption className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-muted">
        <span className="inline-flex items-center gap-1">
          <ArmGlyph model="haiku" effort="high" /> Haiku
          <ArmGlyph model="sonnet" effort="high" className="ml-1" /> Sonnet
          <ArmGlyph model="opus" effort="high" className="ml-1" /> Opus
          <span className="ml-0.5">(smallest to largest)</span>
        </span>
        <span className="inline-flex items-center gap-1">
          <ArmGlyph model="haiku" effort="low" />
          <ArmGlyph model="haiku" effort="medium" />
          <ArmGlyph model="haiku" effort="high" /> low, medium, high effort
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-px w-5 bg-fg" /> best value: nothing beats it on both
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block w-5 border-t-2 border-dotted border-accent" /> Cascade at each
          slider setting
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-3 w-px bg-muted opacity-60" /> likely range
        </span>
      </figcaption>
    </figure>
  );
}
