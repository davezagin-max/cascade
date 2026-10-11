"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArmGlyph } from "@/components/ArmGlyph";
import { CostMeter } from "@/components/kit/CostMeter";
import { useKey } from "@/components/kit/KeyProvider";
import { SimTag } from "@/components/SimTag";
import type { ApiMeta } from "@/lib/api";
import { streamApi } from "@/lib/client";
import { armLabel as labelFromKey, armParts, pct, usd } from "@/lib/format";
import { scrollBehavior } from "@/lib/motion";
import type { PlaygroundEvent, PlaygroundOutput } from "@/lib/playground";
import {
  ROUTER,
  armLabel,
  explain,
  findArm,
  probability,
  route,
  topContributions,
  type Decision,
} from "@/lib/router/model";
import { AnswerText } from "./AnswerText";

export interface Preset {
  demoId: string;
  label: string;
  prompt: string;
}

interface Props {
  presets: Preset[];
  /** Largest prompt the route accepts. */
  maxChars: number;
}

type Run =
  | { state: "idle" }
  | { state: "running"; text: string; meta: Pick<ApiMeta, "mode" | "placeholder" | "recordedAt"> | null }
  | { state: "done"; data: PlaygroundOutput; meta: ApiMeta }
  | { state: "stopped"; text: string }
  | { state: "error"; kind: string; message: string; meta: ApiMeta | null };

const short = (model: string) => model.replace("claude-", "").replace("-5-5", "");

/** Where an answer came from, in words that match what actually produced it. */
function provenance(meta: Pick<ApiMeta, "mode" | "placeholder" | "recordedAt">, arm: string | null): string {
  if (meta.mode === "live") return arm ? `Live answer from ${labelFromKey(arm)}` : "Live answer";
  if (meta.placeholder) return "Sample answer, written by hand. Not a model run.";
  const when = meta.recordedAt ? ` on ${meta.recordedAt.slice(0, 10)}` : "";
  return arm ? `Recorded run${when}, answered by ${labelFromKey(arm)}` : `Recorded run${when}`;
}

export function Playground({ presets, maxChars }: Props) {
  const { apiKey, isLive, openKeyModal } = useKey();
  const [prompt, setPrompt] = useState("");
  const [run, setRun] = useState<Run>({ state: "idle" });
  const controller = useRef<AbortController | null>(null);
  const runId = useRef(0);
  const responseRef = useRef<HTMLDivElement>(null);
  const synthetic = ROUTER.trained_on.synthetic;

  // Abandoning the page mid-run must stop the request, and with it the model call.
  useEffect(() => () => controller.current?.abort(), []);

  const trimmed = prompt.trim();
  const decision = useMemo(() => (trimmed ? route(ROUTER, trimmed) : null), [trimmed]);
  const preset = presets.find((item) => item.prompt === trimmed) ?? null;
  const busy = run.state === "running";
  const canRun = Boolean(trimmed) && !busy && (isLive || preset !== null);

  async function start(text: string, demoId: string | undefined) {
    controller.current?.abort();
    const abort = new AbortController();
    controller.current = abort;
    const id = ++runId.current;
    const current = () => runId.current === id && !abort.signal.aborted;
    setRun({ state: "running", text: "", meta: null });

    const response = await streamApi<PlaygroundOutput, PlaygroundEvent>(
      "/api/route-and-run",
      demoId ? { demoId, prompt: text } : { prompt: text },
      apiKey,
      {
        onMeta: (meta) => {
          if (current()) setRun((r) => (r.state === "running" ? { ...r, meta } : r));
        },
        onEvent: (event) => {
          if (event.type !== "text" || !current()) return;
          setRun((r) => (r.state === "running" ? { ...r, text: r.text + event.delta } : r));
        },
      },
      abort.signal,
    );
    if (!current()) return; // superseded or stopped: that path already set the state
    if (response.ok) setRun({ state: "done", data: response.data, meta: response.meta });
    else
      setRun({
        state: "error",
        kind: response.error.kind,
        message: response.error.message,
        meta: response.meta ?? null,
      });
  }

  function stop() {
    controller.current?.abort();
    setRun((r) => (r.state === "running" ? { state: "stopped", text: r.text } : r));
  }

  function choose(item: Preset) {
    setPrompt(item.prompt);
    void start(item.prompt, item.demoId);
    // On a phone the answer is below the fold, so a preset tap would appear to do nothing.
    requestAnimationFrame(() => {
      const node = responseRef.current;
      if (!node) return;
      const { top, bottom } = node.getBoundingClientRect();
      if (top >= 0 && bottom <= window.innerHeight) return;
      const stacked = window.matchMedia("(max-width: 1023px)").matches;
      node.scrollIntoView({ behavior: scrollBehavior(), block: stacked ? "start" : "nearest" });
    });
  }

  return (
    <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
      <div className="flex min-w-0 flex-col gap-4">
        <div>
          <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
            Presets: no key needed
          </p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {presets.map((item) => {
              const arm = route(ROUTER, item.prompt).arm;
              return (
                <button
                  key={item.demoId}
                  type="button"
                  onClick={() => choose(item)}
                  disabled={busy}
                  aria-pressed={preset?.demoId === item.demoId}
                  className={`inline-flex items-center gap-1.5 border px-2 py-1.5 text-left text-sm disabled:opacity-50 ${
                    preset?.demoId === item.demoId ? "border-fg bg-surface" : "border-border hover:bg-surface"
                  }`}
                >
                  <ArmGlyph model={short(arm.model)} effort={arm.effort} size={11} />
                  {item.label}
                </button>
              );
            })}
          </div>
        </div>

        <form
          className="flex flex-col gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            if (canRun) void start(trimmed, preset?.demoId);
          }}
        >
          <label htmlFor="prompt" className="font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
            Prompt
          </label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            maxLength={maxChars}
            rows={6}
            spellCheck={false}
            placeholder="Type any prompt. The routing decision updates as you type and needs no key."
            aria-describedby="prompt-help"
            className="min-w-0 resize-y border border-border bg-surface p-2.5 font-mono text-[13px] leading-relaxed outline-none focus-visible:border-fg"
          />
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
            {busy ? (
              <button type="button" onClick={stop} className="border border-fg px-3 py-2 text-sm font-medium hover:bg-surface">
                Stop
              </button>
            ) : (
              <button
                type="submit"
                disabled={!canRun}
                className="bg-accent px-3 py-2 text-sm font-medium text-accent-fg hover:opacity-90 disabled:opacity-40"
              >
                {decision ? `Run on ${armLabel(decision.arm)}` : "Run"}
              </button>
            )}
            <p id="prompt-help" className="min-w-0 flex-1 basis-56 text-xs text-muted">
              {isLive ? (
                "Runs on your key: the router picks the arm, that model answers, and you see what it cost."
              ) : trimmed && !preset ? (
                <>
                  Routing is live with no key. To get this prompt answered,{" "}
                  <button type="button" onClick={openKeyModal} className="text-fg underline underline-offset-2">
                    add your Anthropic key
                  </button>
                  , or pick a preset.
                </>
              ) : (
                "No key needed for presets. Routing is computed in your browser; only the answer needs a model."
              )}
            </p>
          </div>
        </form>

        {/* The scroll margin clears the sticky banner when a preset scrolls the answer into view. */}
        <div ref={responseRef} className="scroll-mt-24 sm:scroll-mt-12">
          <Response run={run} decision={decision} synthetic={synthetic} onAddKey={openKeyModal} />
        </div>
      </div>

      <DecisionPanel decision={decision} synthetic={synthetic} />
    </div>
  );
}

function Response({
  run,
  decision,
  synthetic,
  onAddKey,
}: {
  run: Run;
  decision: Decision | null;
  synthetic: boolean;
  onAddKey: () => void;
}) {
  const box = "min-h-40 border border-border";
  const bar = "flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-border bg-surface px-3 py-1.5 font-mono text-[11px] text-muted";

  if (run.state === "idle") {
    return (
      <div className={`${box} flex items-center justify-center p-6 text-center text-sm text-muted`} aria-live="polite">
        Pick a preset to watch a request get routed and answered.
      </div>
    );
  }
  if (run.state === "error") {
    const needsKey = run.kind === "demo_miss" || run.kind === "auth";
    return (
      <div className={`${box} p-3`} role="alert">
        <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-danger">
          {run.kind === "refusal" ? "Declined" : run.kind === "demo_miss" ? "Needs a key" : "Could not run"}
        </p>
        <p className="mt-1 text-sm">{run.message}</p>
        {needsKey && (
          <button type="button" onClick={onAddKey} className="mt-3 border border-border px-3 py-1.5 text-sm hover:bg-surface">
            {run.kind === "auth" ? "Check your key" : "Add your key"}
          </button>
        )}
        {run.meta && <CostMeter meta={run.meta} className="mt-3" />}
      </div>
    );
  }
  if (run.state === "stopped") {
    return (
      <div className={box} aria-live="polite">
        <p className={bar}>Stopped. The request was cancelled, so no further tokens are billed.</p>
        <AnswerText text={run.text || "(nothing had arrived yet)"} />
      </div>
    );
  }
  if (run.state === "running") {
    return (
      <div className={box} aria-live="polite" aria-busy>
        <p className={bar}>
          {run.meta ? provenance(run.meta, run.meta.mode === "live" ? (decision?.arm.key ?? null) : null) : "Routing…"}
        </p>
        <AnswerText text={run.text} streaming />
      </div>
    );
  }

  const { data, meta } = run;
  const stale =
    meta.mode === "demo" && !meta.placeholder && decision !== null && data.arm.key !== decision.arm.key;
  const comparison = data.comparison;
  return (
    <div className={box} aria-live="polite" data-testid="response">
      <p className={bar}>
        <span className="text-fg">{provenance(meta, data.arm.key)}</span>
        {data.stopReason === "max_tokens" && <span className="text-danger">cut off at the token limit</span>}
      </p>
      <AnswerText text={data.answer} />
      <div className="space-y-1 border-t border-border px-3 py-2">
        <CostMeter meta={meta} />
        {comparison && (
          <p className="font-mono text-xs text-muted">
            About <span className="text-fg">{usd(comparison.estimatedUsd)}</span> on{" "}
            {labelFromKey(comparison.baselineKey)}, so this saved roughly{" "}
            <span className="text-fg">{pct(1 - comparison.actualUsd / comparison.estimatedUsd, 0)}</span>. Estimate:
            same input tokens, output scaled x{comparison.outputRatio.toFixed(1)} by the eval&apos;s mean token ratio
            {synthetic ? ", which is simulated" : ""}.
          </p>
        )}
        {stale && (
          <p className="font-mono text-xs text-muted">
            This was recorded with an earlier router, which chose {labelFromKey(data.arm.key)}. The current router
            picks {armLabel(decision.arm)}.
          </p>
        )}
      </div>
    </div>
  );
}

function DecisionPanel({ decision, synthetic }: { decision: Decision | null; synthetic: boolean }) {
  const baseline = findArm(ROUTER, ROUTER.baseline_arm);
  if (!decision) {
    return (
      <aside aria-label="Router decision" className="min-w-0 border border-dashed border-border p-4 text-sm text-muted">
        <p className="font-mono text-[11px] uppercase tracking-[0.14em]">Router decision</p>
        <p className="mt-2">
          The router scores a prompt from {ROUTER.features.length} cheap text features and picks the cheapest arm
          predicted to pass at least {pct(ROUTER.threshold, 0)} of the time. It appears here as soon as there is a
          prompt.
        </p>
      </aside>
    );
  }
  const chosen = decision.arm;
  const chosenP = probability(chosen.bias + decision.score);
  const signals = topContributions(decision, 6);
  const peak = Math.max(...signals.map((item) => Math.abs(item.contribution)), 0.001);

  return (
    <aside aria-label="Router decision" aria-live="polite" className="min-w-0 border border-border" data-testid="decision">
      <div className="border-b border-border bg-surface px-3 py-2">
        <p className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-muted">
          Router decision
          <span className="normal-case tracking-normal">computed locally, no model call</span>
        </p>
        <p className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-1">
          <ArmGlyph model={short(chosen.model)} effort={chosen.effort} size={18} />
          <span className="text-xl font-semibold tracking-tight">{armLabel(chosen)}</span>
          <span className="font-mono text-xs text-muted">{chosen.model}</span>
        </p>
        <p className="tabular mt-0.5 font-mono text-xs text-muted">
          predicted pass <span className="text-fg">{pct(chosenP, 0)}</span>, threshold{" "}
          {pct(ROUTER.threshold, 0)}
          {decision.fallback && <span className="text-accent"> · fallback: nothing clears the threshold</span>}
        </p>
      </div>

      <div className="space-y-4 p-3">
        <div className="space-y-1.5 text-sm">
          {explain(ROUTER, decision).map((line) => (
            <p key={line}>{line}</p>
          ))}
        </div>

        <div>
          <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">
            Every arm, cheapest first
          </p>
          <ul className="font-mono text-[11px]">
            {ROUTER.arms.map((arm, i) => {
              const p = probability(decision.logits[i]);
              const picked = arm.key === chosen.key;
              return (
                <li
                  key={arm.key}
                  className={`grid grid-cols-[7.4rem_minmax(0,1fr)_2.1rem_3.9rem] items-center gap-x-2 border-t border-grid py-[3px] ${
                    picked ? "font-semibold" : "text-muted"
                  }`}
                >
                  <span className="whitespace-nowrap">
                    <ArmGlyph {...armParts(arm.key)} size={10} muted={!picked} /> {armLabel(arm)}
                  </span>
                  <span className="relative h-2.5 bg-grid" aria-hidden>
                    <span
                      className="glide absolute inset-y-0 left-0"
                      style={{ width: `${p * 100}%`, background: picked ? "var(--accent)" : "var(--muted)" }}
                    />
                    <span className="absolute -inset-y-0.5 w-px bg-fg" style={{ left: `${ROUTER.threshold * 100}%` }} />
                  </span>
                  <span className="tabular text-right">{pct(p, 0)}</span>
                  <span className="tabular text-right">{usd(arm.cost_per_request)}</span>
                </li>
              );
            })}
          </ul>
          <p className="mt-1 font-mono text-[10px] text-muted">
            bar: predicted pass probability · line: threshold · right: mean eval cost per request{" "}
            <SimTag show={synthetic} />
          </p>
        </div>

        {signals.length > 0 && (
          <div>
            <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">Strongest signals</p>
            <ul className="font-mono text-[11px]">
              {signals.map((item) => {
                const share = (Math.abs(item.contribution) / peak) * 50;
                return (
                  <li
                    key={item.name}
                    className="grid grid-cols-[minmax(0,10rem)_minmax(0,1fr)_2.75rem] items-center gap-x-2 border-t border-grid py-[3px]"
                  >
                    <span className="truncate" title={`${item.label}: ${item.z >= 0 ? "above" : "below"} the suite average`}>
                      {item.label}
                    </span>
                    <span className="relative h-2.5" aria-hidden>
                      <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
                      <span
                        className="absolute inset-y-0.5 bg-fg"
                        style={item.contribution >= 0 ? { left: "50%", width: `${share}%` } : { right: "50%", width: `${share}%` }}
                      />
                    </span>
                    <span className="tabular text-right">
                      {item.contribution >= 0 ? "+" : ""}
                      {item.contribution.toFixed(2)}
                    </span>
                  </li>
                );
              })}
            </ul>
            <p className="mt-1 flex justify-between font-mono text-[10px] text-muted">
              <span>left: toward a stronger arm</span>
              <span>right: toward a cheaper arm</span>
            </p>
          </div>
        )}

        {baseline && baseline.key !== chosen.key && (
          <p className="border-t border-border pt-2 font-mono text-xs text-muted">
            On the eval suite {armLabel(chosen)} averages{" "}
            <span className="text-fg">{usd(chosen.cost_per_request)}</span> per request against{" "}
            <span className="text-fg">{usd(baseline.cost_per_request)}</span> for {armLabel(baseline)}.{" "}
            <SimTag show={synthetic} />
          </p>
        )}
      </div>
    </aside>
  );
}
