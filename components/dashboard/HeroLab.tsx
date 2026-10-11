"use client";

import Link from "next/link";
import { useState } from "react";
import { ArmGlyph } from "@/components/ArmGlyph";
import { SimTag } from "@/components/SimTag";
import { armLabel, armParts, pct, usd } from "@/lib/format";
import { ParetoChart, type ChartArm } from "./ParetoChart";

export interface SweepPoint {
  threshold: number;
  quality: number;
  cost_per_request: number;
  cost_saved: number;
  quality_retained: number;
  mix: Record<string, number>;
}

interface Props {
  synthetic: boolean;
  arms: ChartArm[];
  /** Arm keys, cheapest first. */
  armOrder: string[];
  sweep: SweepPoint[];
  defaultThreshold: number;
  baselineKey: string;
  baseline: { quality: number; cost_per_request: number };
  nTasks: number;
  folds: number;
}

/**
 * The first screen: the router headline, the control that moves it, and the chart
 * it moves on. Everything shown is precomputed by cross-validation in Python; the
 * control only selects which threshold's numbers to show.
 */
export function HeroLab({
  synthetic,
  arms,
  armOrder,
  sweep,
  defaultThreshold,
  baselineKey,
  baseline,
  nTasks,
  folds,
}: Props) {
  const initial = Math.max(
    0,
    sweep.findIndex((point) => point.threshold === defaultThreshold),
  );
  const [index, setIndex] = useState(initial);
  const point = sweep[index];
  const total = Object.values(point.mix).reduce((sum, n) => sum + n, 0);
  const step = (delta: number) => setIndex((i) => Math.min(sweep.length - 1, Math.max(0, i + delta)));

  return (
    <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <div className="flex min-w-0 flex-col gap-5">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
            Router versus always {armLabel(baselineKey)}
          </p>
          <h1 aria-live="polite" className="mt-2 text-[1.7rem] font-semibold leading-[1.15] tracking-tight sm:text-4xl">
            The router saves{" "}
            <span className="tabular whitespace-nowrap text-accent">{pct(point.cost_saved)}</span> of cost
            and keeps <span className="tabular whitespace-nowrap text-accent">{pct(point.quality_retained)}</span>{" "}
            of quality. <SimTag show={synthetic} className="align-[0.35em]" />
          </h1>
          <p className="mt-3 max-w-xl text-sm text-muted">
            Each request goes to the cheapest Claude model and effort level predicted to get it right. Measured by{" "}
            {folds}-fold cross-validation on {nTasks} eval tasks: every task is routed by a model that never saw it.
          </p>
        </div>

        <div className="border border-border bg-surface p-3">
          <div className="flex items-baseline justify-between gap-3">
            <label htmlFor="threshold" className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
              Confidence threshold
            </label>
            <output htmlFor="threshold" className="tabular font-mono text-sm font-semibold">
              {point.threshold.toFixed(2)}
            </output>
          </div>
          <div className="mt-1 flex items-center gap-2">
            <button
              type="button"
              onClick={() => step(-1)}
              disabled={index === 0}
              aria-label="Lower the threshold: cheaper, less careful"
              className="size-8 shrink-0 border border-border font-mono text-base leading-none hover:bg-bg disabled:opacity-40"
            >
              −
            </button>
            <input
              id="threshold"
              type="range"
              className="lab-range min-w-0"
              min={0}
              max={sweep.length - 1}
              step={1}
              value={index}
              onChange={(event) => setIndex(Number(event.target.value))}
              aria-valuetext={`${point.threshold.toFixed(2)}: saves ${pct(point.cost_saved)}, keeps ${pct(point.quality_retained)} of quality`}
            />
            <button
              type="button"
              onClick={() => step(1)}
              disabled={index === sweep.length - 1}
              aria-label="Raise the threshold: more careful, more expensive"
              className="size-8 shrink-0 border border-border font-mono text-base leading-none hover:bg-bg disabled:opacity-40"
            >
              +
            </button>
          </div>
          <p className="mt-1 text-xs text-muted">
            Send each request to the cheapest arm predicted to pass at least{" "}
            <span className="tabular font-mono text-fg">{pct(point.threshold, 0)}</span> of the time. Move it and watch
            the router slide along its curve.
          </p>

          <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-border pt-3 font-mono text-xs">
            <div>
              <dt className="text-muted">cost per 1k requests</dt>
              <dd className="tabular mt-0.5 text-sm">
                <span className="font-semibold">{usd(point.cost_per_request * 1000)}</span>{" "}
                <span className="text-muted">vs {usd(baseline.cost_per_request * 1000)}</span>
              </dd>
            </div>
            <div>
              <dt className="text-muted">pass rate</dt>
              <dd className="tabular mt-0.5 text-sm">
                <span className="font-semibold">{pct(point.quality)}</span>{" "}
                <span className="text-muted">vs {pct(baseline.quality)}</span>
              </dd>
            </div>
          </dl>

          <div className="mt-3">
            <p className="font-mono text-[11px] text-muted">where the {total} tasks go</p>
            <div className="mt-1 flex h-3 w-full overflow-hidden border border-border" role="img" aria-label={armOrder
              .filter((key) => point.mix[key])
              .map((key) => `${armLabel(key)}: ${point.mix[key]}`)
              .join(", ")}>
              {armOrder.map((key, position) => (
                <div
                  key={key}
                  title={`${armLabel(key)}: ${point.mix[key] ?? 0}`}
                  className="glide h-full border-r border-bg last:border-r-0"
                  style={{
                    width: `${((point.mix[key] ?? 0) / total) * 100}%`,
                    background: "var(--fg)",
                    opacity: 0.14 + (0.86 * position) / Math.max(1, armOrder.length - 1),
                  }}
                />
              ))}
            </div>
            <ul className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 font-mono text-[11px] text-muted">
              {armOrder
                .filter((key) => point.mix[key])
                .map((key) => (
                  <li key={key} className="inline-flex items-center gap-1 whitespace-nowrap">
                    <ArmGlyph {...armParts(key)} size={10} />
                    {armLabel(key)} <span className="tabular text-fg">{point.mix[key]}</span>
                  </li>
                ))}
            </ul>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 text-sm">
          <Link
            href="/playground"
            className="bg-accent px-3 py-2 font-medium text-accent-fg hover:opacity-90"
          >
            Try the router on a prompt
          </Link>
          <Link href="/tasks" className="border border-border px-3 py-2 hover:bg-surface">
            Browse the {nTasks} tasks
          </Link>
        </div>
      </div>

      <div className="min-w-0">
        <ParetoChart
          arms={arms}
          curve={sweep.map((p) => ({ threshold: p.threshold, cost: p.cost_per_request * 1000, quality: p.quality }))}
          selected={index}
          synthetic={synthetic}
        />
      </div>
    </div>
  );
}
