"use client";

import Link from "next/link";
import { useState } from "react";
import { ArmGlyph } from "@/components/ArmGlyph";
import { SimTag } from "@/components/SimTag";
import { armName, armParts, pct, usd } from "@/lib/format";
import { ParetoChart, type ChartArm } from "./ParetoChart";

export interface SweepPoint {
  threshold: number;
  quality: number;
  cost_per_request: number;
  cost_saved: number;
  quality_retained: number;
  mix: Record<string, number>;
  /** The strongest single arm that costs no more than the router does at this threshold. */
  best_fixed: { arm: string; quality: number; cost_per_request: number } | null;
}

/** "2.6 points ahead of", "4.8 points behind", or "level with": the router against one setting. */
function versus(delta: number): string {
  const points = Math.abs(delta * 100).toFixed(1);
  if (points === "0.0") return "level with";
  return `${points} points ${delta > 0 ? "ahead of" : "behind"}`;
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
  /** Tasks the router was evaluated on. */
  nTasks: number;
  /** Tasks in the suite, which is more when a task had to be left out of the evaluation. */
  suiteTasks: number;
  folds: number;
}

/** How Cascade works, in three steps a stranger can follow. */
const STEPS = ["Read the prompt", "Predict which models would get it right", "Send it to the cheapest of those"];

/** The preset the primary button plays: a cheap win, which is the whole point. */
export const FIRST_PRESET = "support-ticket";

/**
 * The first screen: what Cascade is in one sentence, one button that starts the demo,
 * the router's result with the slider that moves it, and the chart it moves on.
 * Everything shown is precomputed by cross-validation in Python; the slider only
 * selects which threshold's numbers to show.
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
  suiteTasks,
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
  const bar = pct(point.threshold, 0);

  return (
    <div className="grid gap-x-12 gap-y-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
      <div className="flex min-w-0 flex-col gap-6">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">
            Cascade · a model router for Claude
          </p>
          <h1 className="mt-2 text-[1.9rem] font-semibold leading-[1.08] tracking-tight sm:text-[2.6rem]">
            Not every request needs the most expensive model.
          </h1>
          <p className="mt-3 max-w-xl text-[15px] leading-relaxed text-muted sm:text-base">
            <span className="text-fg">
              Cascade picks the cheapest Claude model and effort setting that can handle each request,
            </span>{" "}
            then tests that choice on {suiteTasks} graded tasks.
          </p>
        </div>

        <ol aria-label="How it works" className="flex gap-3 text-[13px] leading-snug">
          {STEPS.map((text, i) => (
            <li key={text} className="flex flex-1 basis-0 gap-1.5 border-t border-border pt-2">
              {i > 0 && (
                <span aria-hidden className="text-accent">
                  →
                </span>
              )}
              <span>{text}</span>
            </li>
          ))}
        </ol>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <Link
            href={`/playground?try=${FIRST_PRESET}`}
            className="bg-accent px-4 py-2.5 text-[15px] font-medium text-accent-fg hover:opacity-90"
          >
            Try it on a sample prompt
          </Link>
          <Link
            href="/tasks"
            className="text-sm underline decoration-border underline-offset-4 hover:decoration-fg"
          >
            Browse the {suiteTasks} test tasks
          </Link>
        </div>

        <section aria-labelledby="result-title" className="border border-border bg-surface">
          <div className="p-4">
            <h2
              id="result-title"
              className="flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted"
            >
              The result <SimTag show={synthetic} />
            </h2>
            <p className="mt-1 text-[13px] text-muted">
              Compared with always using the top setting, {armName(baselineKey)}:
            </p>
            <div aria-live="polite" data-testid="headline" className="mt-2 grid grid-cols-2 gap-4">
              <p>
                <span className="tabular block text-[2rem] font-semibold leading-none tracking-tight text-accent sm:text-4xl">
                  {pct(Math.abs(point.cost_saved))}
                </span>
                <span className="mt-1 block text-sm">{point.cost_saved >= 0 ? "lower cost" : "higher cost"}</span>
              </p>
              <p>
                <span className="tabular block text-[2rem] font-semibold leading-none tracking-tight text-accent sm:text-4xl">
                  {pct(point.quality_retained)}
                </span>
                <span className="mt-1 block text-sm">of the quality kept</span>
              </p>
            </div>
            {/* The harder baseline: not "always the top setting", but the best single setting you could afford instead. */}
            <dl className="mt-3 border-t border-grid pt-2" data-testid="best-fixed">
              <dt className="text-xs text-muted">Tougher comparison: the best single setting that costs no more</dt>
              <dd className="mt-0.5 text-[13px] leading-snug">
                {point.best_fixed ? (
                  <>
                    Cascade is{" "}
                    <span className="tabular font-semibold">{versus(point.quality - point.best_fixed.quality)}</span>{" "}
                    always using{" "}
                    <span className="whitespace-nowrap">
                      <ArmGlyph {...armParts(point.best_fixed.arm)} size={10} /> {armName(point.best_fixed.arm)}
                    </span>{" "}
                    <span className="tabular text-muted">
                      ({pct(point.best_fixed.quality)} of tasks passed, {usd(point.best_fixed.cost_per_request * 1000)}{" "}
                      per 1,000 requests)
                    </span>
                    .
                  </>
                ) : (
                  "No single setting is this cheap."
                )}
              </dd>
            </dl>
          </div>

          <div className="border-t border-border p-4">
            <div className="flex items-baseline justify-between gap-3">
              <label htmlFor="threshold" className="text-sm font-medium">
                How sure must it be before it picks a cheaper model?
              </label>
              <output htmlFor="threshold" className="tabular font-mono text-sm font-semibold">
                {bar}
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
                aria-valuetext={`${bar} sure: ${pct(Math.abs(point.cost_saved))} ${point.cost_saved >= 0 ? "lower" : "higher"} cost, ${pct(point.quality_retained)} of the quality kept`}
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
            <p aria-hidden className="mt-0.5 flex justify-between gap-3 px-10 font-mono text-[10px] text-muted">
              <span>← spend less</span>
              <span className="text-right">get more right →</span>
            </p>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              Cascade only picks a cheaper model when it predicts that model gets the task right at least{" "}
              <span className="tabular font-mono text-fg">{bar}</span> of the time. Slide it and watch the orange ring
              move on the chart.
            </p>

            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-grid pt-3 text-xs">
              <div>
                <dt className="text-muted">Cost per 1,000 requests</dt>
                <dd className="tabular mt-0.5 font-mono text-sm">
                  <span className="font-semibold">{usd(point.cost_per_request * 1000)}</span>{" "}
                  <span className="text-muted">vs {usd(baseline.cost_per_request * 1000)}</span>
                </dd>
              </div>
              <div>
                <dt className="text-muted">Tasks passed</dt>
                <dd className="tabular mt-0.5 font-mono text-sm">
                  <span className="font-semibold">{pct(point.quality)}</span>{" "}
                  <span className="text-muted">vs {pct(baseline.quality)}</span>
                </dd>
              </div>
            </dl>
            <p className="mt-3 text-[11px] leading-snug text-muted">
              {synthetic ? "Computed on simulated results" : "Measured"} by {folds}-fold cross-validation on {nTasks}{" "}
              tasks: every task is routed by a version of the router that never saw it.
            </p>
          </div>
        </section>
      </div>

      <div className="min-w-0">
        <p className="flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
          What you are seeing <SimTag show={synthetic} />
        </p>
        <p className="mb-4 mt-1 max-w-2xl text-sm leading-relaxed text-muted">
          Claude comes in three sizes (Haiku, Sonnet, Opus), and each can think briefly or at length: its effort
          setting. Each shape is one of those nine options. <span className="text-fg">Further right costs more;
          higher passes more of the {nTasks} test tasks.</span> The orange ring is Cascade, choosing per request.
        </p>
        <ParetoChart
          arms={arms}
          curve={sweep.map((p) => ({ threshold: p.threshold, cost: p.cost_per_request * 1000, quality: p.quality }))}
          selected={index}
          synthetic={synthetic}
        />

        <div className="mt-6">
          <p className="text-xs text-muted">
            Where Cascade sent the {total} tasks, at <span className="tabular font-mono text-fg">{bar}</span> sure
          </p>
          <div
            className="mt-1 flex h-3 w-full overflow-hidden border border-border"
            role="img"
            aria-label={armOrder
              .filter((key) => point.mix[key])
              .map((key) => `${armName(key)}: ${point.mix[key]}`)
              .join(", ")}
          >
            {armOrder.map((key, position) => (
              <div
                key={key}
                title={`${armName(key)}: ${point.mix[key] ?? 0}`}
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
                  {armName(key)} <span className="tabular text-fg">{point.mix[key]}</span>
                </li>
              ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
