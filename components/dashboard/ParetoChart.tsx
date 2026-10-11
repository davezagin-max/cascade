"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArmGlyph, GlyphMark } from "@/components/ArmGlyph";
import { armCode, armLabel, duration, pct, usd } from "@/lib/format";

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
  const margin = { top: 14, right: compact ? 10 : 18, bottom: 40, left: compact ? 38 : 46 };
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
    const x = (cost: number) => margin.left + ((Math.log10(cost) - lo) / (hi - lo)) * innerW;
    const y = (quality: number) => margin.top + (1 - (quality - floor) / (1 - floor)) * innerH;
    const yTicks: number[] = [];
    for (let t = floor; t <= 1.0001; t += 0.1) yTicks.push(Math.round(t * 10) / 10);
    const xTicks = X_TICKS.filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi);

    // Greedy label placement: try right, left, above, below; keep the first that hits nothing.
    const points = arms.map((arm) => ({ arm, px: x(arm.cost), py: y(arm.quality) }));
    const taken: Box[] = points.map((p) => ({ x: p.px - 7, y: p.py - 7, w: 14, h: 14 }));
    const labels = points.map((p) => {
      const text = armCode(p.arm.key);
      const w = text.length * 6.1 + 2;
      const right: Box = { x: p.px + 10, y: p.py - 6, w, h: 12 };
      const left: Box = { x: p.px - 10 - w, y: p.py - 6, w, h: 12 };
      const above: Box = { x: p.px - w / 2, y: p.py - 22, w, h: 12 };
      const below: Box = { x: p.px - w / 2, y: p.py + 10, w, h: 12 };
      // The frontier climbs to the upper right, so below-right of a frontier point is clear.
      const belowRight: Box = { x: p.px + 8, y: p.py + 4, w, h: 12 };
      const options: Box[] = p.arm.frontier
        ? [belowRight, right, left, below, above]
        : [right, left, below, above];
      const inside = (b: Box) => b.x >= margin.left && b.x + b.w <= width - 2;
      const chosen =
        options.find((b) => inside(b) && !taken.some((t) => t !== taken[points.indexOf(p)] && overlaps(b, t))) ??
        options[0];
      taken.push(chosen);
      return { text, x: chosen.x, y: chosen.y + 9.5 };
    });
    return { x, y, xTicks, yTicks, points, labels };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [arms, curve, width, height]);

  const { x, y, xTicks, yTicks, points, labels } = layout;
  const frontier = points.filter((p) => p.arm.frontier).sort((a, b) => a.arm.cost - b.arm.cost);
  const frontierPath = frontier.map((p, i) => `${i ? "L" : "M"}${p.px.toFixed(1)},${p.py.toFixed(1)}`).join("");
  const curvePath = curve
    .map((p, i) => `${i ? "L" : "M"}${x(p.cost).toFixed(1)},${y(p.quality).toFixed(1)}`)
    .join("");
  const operating = curve[selected];
  const ox = x(operating.cost);
  const oy = y(operating.quality);

  const tip = (() => {
    if (active === "router") {
      return {
        px: ox,
        py: oy,
        title: `Router at ${operating.threshold.toFixed(2)}`,
        rows: [
          ["pass rate", pct(operating.quality)],
          ["per 1k requests", usd(operating.cost)],
        ],
      };
    }
    const point = points.find((p) => p.arm.key === active);
    if (!point) return null;
    const a = point.arm;
    return {
      px: point.px,
      py: point.py,
      title: armLabel(a.key),
      rows: [
        ["pass rate", `${pct(a.quality)} (${pct(a.ci[0], 0)} to ${pct(a.ci[1], 0)})`],
        ["per 1k requests", usd(a.cost)],
        ["p50 latency", duration(a.p50)],
        ["frontier", a.frontier ? "yes" : "dominated"],
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
          aria-label="Pass rate against cost per 1,000 requests for every model and effort level, with the router's operating curve"
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
            cost per 1,000 requests, log scale
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
              aria-label={`${armLabel(p.arm.key)}: ${pct(p.arm.quality)} pass rate at ${usd(p.arm.cost)} per 1,000 requests${p.arm.frontier ? ", on the frontier" : ", dominated"}`}
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
            aria-label={`Router at threshold ${operating.threshold.toFixed(2)}: ${pct(operating.quality)} pass rate at ${usd(operating.cost)} per 1,000 requests`}
            onMouseEnter={() => setActive("router")}
            onFocus={() => setActive("router")}
            onBlur={() => setActive(null)}
          >
            <circle r={9} fill="var(--bg)" stroke="var(--accent)" strokeWidth={2} />
            <circle r={3.5} fill="var(--accent)" />
            <text
              x={-13}
              y={-8}
              textAnchor="end"
              fontSize="10"
              fontWeight={600}
              fill="var(--accent)"
              stroke="var(--bg)"
              strokeWidth={3}
              paintOrder="stroke"
            >
              router
            </text>
          </g>

          <g aria-hidden pointerEvents="none">
            {points.map((p, i) => (
              <text
                key={p.arm.key}
                x={labels[i].x}
                y={labels[i].y}
                fontSize="10"
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
        </span>
        <span className="inline-flex items-center gap-1">
          <ArmGlyph model="sonnet" effort="high" /> Sonnet
        </span>
        <span className="inline-flex items-center gap-1">
          <ArmGlyph model="opus" effort="high" /> Opus
        </span>
        <span className="inline-flex items-center gap-1">
          <ArmGlyph model="haiku" effort="low" />
          <ArmGlyph model="haiku" effort="medium" />
          <ArmGlyph model="haiku" effort="high" /> effort low, medium, high
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-px w-5 bg-fg" /> Pareto frontier
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block w-5 border-t-2 border-dotted border-accent" /> router at each
          threshold
        </span>
      </figcaption>
    </figure>
  );
}
