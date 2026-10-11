"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArmGlyph } from "@/components/ArmGlyph";
import { SimTag } from "@/components/SimTag";
import { armCode, armLabel, armParts, count, duration, usd } from "@/lib/format";
import { scrollBehavior } from "@/lib/motion";
import type { TaskDetail, TaskIndexEntry } from "@/lib/run-schema";

interface Props {
  tasks: TaskIndexEntry[];
  /** Arm keys in display order: tier, then effort. */
  arms: string[];
  categories: { key: string; label: string; n_tasks: number }[];
  synthetic: boolean;
}

const GRADER_NAMES: Record<string, string> = {
  exact: "exact match",
  json: "JSON schema + fields",
  code: "hidden unit tests",
  judge: "Opus judge, rubric",
};

/** A square shaded by how many trials passed. */
function PassCell({ rate }: { rate: number | undefined }) {
  if (rate === undefined) return <span className="block size-3.5 sm:size-4" />;
  return (
    <span
      className="block size-3.5 border border-fg/60 sm:size-4"
      style={{ background: `color-mix(in srgb, var(--fg) ${Math.round(rate * 100)}%, transparent)` }}
    />
  );
}

type Loaded = { state: "loading" } | { state: "error" } | { state: "ready"; detail: TaskDetail };

export function TaskExplorer({ tasks, arms, categories, synthetic }: Props) {
  const [category, setCategory] = useState<string>("all");
  const [open, setOpen] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, Loaded>>({});
  const requested = useRef(new Set<string>());

  const visible = useMemo(
    () => tasks.filter((task) => category === "all" || task.category === category),
    [tasks, category],
  );

  function load(id: string) {
    if (requested.current.has(id)) return;
    requested.current.add(id);
    setDetails((all) => ({ ...all, [id]: { state: "loading" } }));
    fetch(`/run/tasks/${id}.json`)
      .then((res) => {
        if (!res.ok) throw new Error(String(res.status));
        return res.json() as Promise<TaskDetail>;
      })
      .then((detail) => setDetails((all) => ({ ...all, [id]: { state: "ready", detail } })))
      .catch(() => {
        requested.current.delete(id); // allow a retry
        setDetails((all) => ({ ...all, [id]: { state: "error" } }));
      });
  }

  function toggle(id: string) {
    setOpen((current) => (current === id ? null : id));
    load(id);
  }

  const chip = (key: string, label: string, n: number) => (
    <button
      key={key}
      type="button"
      onClick={() => setCategory(key)}
      aria-pressed={category === key}
      className={`border px-2 py-1 font-mono text-xs ${
        category === key ? "border-fg bg-fg text-bg" : "border-border hover:bg-surface"
      }`}
    >
      {label} <span className="tabular opacity-70">{n}</span>
    </button>
  );

  return (
    <div>
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by task type">
        {chip("all", "All", tasks.length)}
        {categories.map((c) => chip(c.key, c.label, c.n_tasks))}
      </div>

      <div className="mt-4 border border-border">
        <div className="flex items-end gap-2 border-b border-border bg-surface px-2 py-1.5 font-mono text-[10px] text-muted">
          <span className="min-w-0 flex-1">task</span>
          <span className="flex gap-px sm:gap-0.5" aria-hidden>
            {arms.map((key) => (
              <span key={key} title={armLabel(key)} className="flex w-3.5 justify-center sm:w-4">
                <ArmGlyph {...armParts(key)} size={11} />
              </span>
            ))}
          </span>
          <span className="hidden w-16 text-right sm:block">router</span>
        </div>
        <ul>
          {visible.map((task) => {
            const expanded = open === task.id;
            const pick = task.router_pick;
            return (
              <li key={task.id} className="border-b border-grid last:border-b-0">
                <button
                  type="button"
                  onClick={() => toggle(task.id)}
                  aria-expanded={expanded}
                  aria-controls={`task-${task.id}`}
                  className={`flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-surface ${expanded ? "bg-surface" : ""}`}
                >
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm">{task.title}</span>
                    <span className="block truncate font-mono text-[10px] text-muted">
                      {task.id} · difficulty {task.difficulty} · {GRADER_NAMES[task.grader]}
                    </span>
                  </span>
                  <span
                    className="flex gap-px sm:gap-0.5"
                    role="img"
                    aria-label={arms
                      .map((key) => `${armLabel(key)} ${Math.round((task.cells[key]?.pass_rate ?? 0) * 100)}%`)
                      .join(", ")}
                  >
                    {arms.map((key) => (
                      <PassCell key={key} rate={task.cells[key]?.pass_rate} />
                    ))}
                  </span>
                  <span className="hidden w-16 justify-end font-mono text-[10px] text-muted sm:flex">
                    {pick ? (
                      <span className="inline-flex items-center gap-1" title={`Router chose ${armLabel(pick)}`}>
                        <ArmGlyph {...armParts(pick)} size={10} /> {armCode(pick)}
                      </span>
                    ) : (
                      "n/a"
                    )}
                  </span>
                </button>
                {expanded && (
                  <div id={`task-${task.id}`} className="border-t border-border bg-bg px-2 py-3 sm:px-3">
                    <Detail task={task} loaded={details[task.id]} arms={arms} synthetic={synthetic} onRetry={() => load(task.id)} />
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      </div>
      <p className="mt-2 font-mono text-[11px] text-muted">
        Each square is one arm: solid when every trial passed, empty when none did. Select a task to read its prompt,
        its grader, and every arm&apos;s answer.
      </p>
    </div>
  );
}

function Block({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.14em] text-muted">{label}</p>
      {children}
    </div>
  );
}

const pre = "max-h-72 overflow-auto whitespace-pre-wrap break-words border border-border bg-surface p-2 font-mono text-xs";

function GraderSpec({ grader }: { grader: TaskDetail["grader"] }) {
  const g = grader as Record<string, unknown>;
  if (g.type === "exact") {
    const labels = (g.labels as string[]) ?? [];
    return (
      <p className="font-mono text-xs">
        expected <span className="border border-border bg-surface px-1">{String(g.expected)}</span>
        {labels.length > 0 && <span className="text-muted"> from {labels.join(", ")}</span>}
      </p>
    );
  }
  if (g.type === "json") {
    return <pre className={pre}>{JSON.stringify(g.expected, null, 2)}</pre>;
  }
  if (g.type === "code") {
    return <pre className={pre}>{(g.tests as string[]).join("\n")}</pre>;
  }
  const rubric = (g.rubric as { name: string; description: string; weight: number }[]) ?? [];
  return (
    <ul className="space-y-1 text-xs">
      {rubric.map((criterion) => (
        <li key={criterion.name}>
          <span className="font-mono">{criterion.name}</span>{" "}
          <span className="font-mono text-muted">x{criterion.weight}</span>{" "}
          <span className="text-muted">{criterion.description}</span>
        </li>
      ))}
    </ul>
  );
}

/** Renders whatever the grader recorded about one answer. */
function GradeDetails({ details }: { details: Record<string, unknown> }) {
  const rows: React.ReactNode[] = [];
  if (Array.isArray(details.tests)) {
    const tests = details.tests as { ok: boolean; error?: string }[];
    rows.push(
      <p key="tests" className="flex flex-wrap gap-1">
        {tests.map((test, i) => (
          <span
            key={i}
            title={test.ok ? `test ${i + 1}: pass` : `test ${i + 1}: ${test.error ?? "fail"}`}
            className={`inline-block border px-1 ${test.ok ? "border-border text-muted" : "border-danger text-danger"}`}
          >
            {i + 1}
            {test.ok ? "" : " fail"}
          </span>
        ))}
      </p>,
    );
    const failed = tests.find((test) => !test.ok && test.error);
    if (failed) rows.push(<p key="why" className="text-muted">first failure: {failed.error}</p>);
  }
  if (details.fields && typeof details.fields === "object") {
    rows.push(
      <p key="fields" className="flex flex-wrap gap-1">
        {Object.entries(details.fields as Record<string, boolean>).map(([name, ok]) => (
          <span key={name} className={`border px-1 ${ok ? "border-border text-muted" : "border-danger text-danger"}`}>
            {name}
            {ok ? "" : " wrong"}
          </span>
        ))}
      </p>,
    );
  }
  if (details.schema_valid === false && Array.isArray(details.errors)) {
    rows.push(<p key="schema" className="text-danger">schema: {(details.errors as string[]).join("; ")}</p>);
  }
  if (typeof details.extracted === "string") {
    rows.push(
      <p key="exact" className="text-muted">
        read <span className="text-fg">{details.extracted || "(nothing)"}</span>, expected{" "}
        <span className="text-fg">{String(details.expected)}</span>
      </p>,
    );
  }
  if (details.criteria && typeof details.criteria === "object") {
    const rationales = (details.rationales ?? {}) as Record<string, string>;
    rows.push(
      <ul key="criteria" className="space-y-0.5">
        {Object.entries(details.criteria as Record<string, number>).map(([name, score]) => (
          <li key={name}>
            <span className={score === 2 ? "" : "text-danger"}>{name} {score}/2</span>{" "}
            <span className="font-sans text-muted">{rationales[name]}</span>
          </li>
        ))}
      </ul>,
    );
  }
  if (typeof details.load_error === "string") rows.push(<p key="load" className="text-danger">{details.load_error}</p>);
  if (typeof details.skipped === "string") rows.push(<p key="skip" className="text-danger">not graded: {details.skipped}</p>);
  if (typeof details.judge_error === "string") rows.push(<p key="judge" className="text-danger">{details.judge_error}</p>);
  if (rows.length === 0) return null;
  return <div className="space-y-1 font-mono text-[11px]">{rows}</div>;
}

function Detail({
  task,
  loaded,
  arms,
  synthetic,
  onRetry,
}: {
  task: TaskIndexEntry;
  loaded: Loaded | undefined;
  arms: string[];
  synthetic: boolean;
  onRetry: () => void;
}) {
  const [arm, setArm] = useState<string>(task.router_pick ?? arms[0]);
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    panel.current?.scrollIntoView({ block: "nearest", behavior: scrollBehavior() });
  }, []);

  if (!loaded || loaded.state === "loading") {
    return (
      <div ref={panel} className="font-mono text-xs text-muted" aria-live="polite">
        Loading task…
      </div>
    );
  }
  if (loaded.state === "error") {
    return (
      <div ref={panel} className="font-mono text-xs text-danger" role="alert">
        Could not load this task.{" "}
        <button type="button" onClick={onRetry} className="underline underline-offset-2">
          Try again
        </button>
      </div>
    );
  }
  const { detail } = loaded;
  const cells = detail.cells.filter((cell) => cell.arm === arm);

  return (
    <div ref={panel} className="grid gap-4 lg:grid-cols-2">
      <div className="min-w-0 space-y-3">
        <Block label="Prompt">
          <pre className={pre}>{detail.prompt}</pre>
        </Block>
        <Block label={`Grader: ${GRADER_NAMES[task.grader]}`}>
          <GraderSpec grader={detail.grader} />
        </Block>
        <details>
          <summary className="cursor-pointer font-mono text-[11px] text-muted">Reference answer and format contract</summary>
          <pre className={`${pre} mt-2`}>{detail.reference}</pre>
          <p className="mt-2 text-xs text-muted">{detail.system}</p>
        </details>
      </div>

      <div className="min-w-0 space-y-3">
        <Block label="Arm">
          <div className="flex flex-wrap gap-1" role="group" aria-label="Choose an arm">
            {arms.map((key) => {
              const passed = detail.cells.filter((cell) => cell.arm === key && cell.passed).length;
              const total = detail.cells.filter((cell) => cell.arm === key).length;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setArm(key)}
                  aria-pressed={arm === key}
                  title={armLabel(key)}
                  className={`inline-flex items-center gap-1 border px-1.5 py-1 font-mono text-[11px] ${
                    arm === key ? "border-fg bg-surface" : "border-border text-muted hover:text-fg"
                  }`}
                >
                  <ArmGlyph {...armParts(key)} size={10} />
                  {armCode(key)}
                  <span className="tabular">
                    {passed}/{total}
                  </span>
                </button>
              );
            })}
          </div>
          <p className="mt-1.5 font-mono text-[11px] text-muted">
            {task.router_pick && <>router chose {armLabel(task.router_pick)}</>}
            {task.oracle_arm && task.oracle_arm !== "none" && <> · cheapest that did best: {armLabel(task.oracle_arm)}</>}
            {task.oracle_arm === "none" && <> · no arm passed this task</>}
          </p>
        </Block>

        <Block label={`${armLabel(arm)}: ${cells.length} trial${cells.length === 1 ? "" : "s"}`}>
          <ul className="space-y-2">
            {cells.map((cell) => {
              const simulated = cell.details.simulated as { pass_probability: number; answer: string } | undefined;
              return (
                <li key={cell.trial} className="border border-border">
                  <p className="flex flex-wrap items-center gap-x-3 gap-y-0.5 border-b border-border bg-surface px-2 py-1 font-mono text-[11px]">
                    <span className={cell.passed ? "font-semibold" : "font-semibold text-danger"}>
                      {cell.passed === null ? "ungraded" : cell.passed ? "pass" : "fail"}
                    </span>
                    {cell.score !== null && <span className="tabular text-muted">score {cell.score.toFixed(2)}</span>}
                    <span className="tabular text-muted">
                      {count(cell.input_tokens)} in · {count(cell.output_tokens)} out · {usd(cell.cost_usd)}
                    </span>
                    <span className="text-muted">
                      {cell.mode}
                      {cell.latency_ms !== null && ` · ${duration(cell.latency_ms)}`}
                    </span>
                    {cell.status !== "ok" && <span className="text-danger">{cell.status}</span>}
                    <SimTag show={synthetic} />
                  </p>
                  <div className="space-y-2 p-2">
                    {simulated && (
                      <p className="text-[11px] text-muted">
                        No model wrote this. The simulator drew a {cell.passed ? "pass" : "fail"} (pass probability{" "}
                        <span className="font-mono">{simulated.pass_probability.toFixed(2)}</span>) and submitted the
                        task&apos;s {simulated.answer} answer to the real grader.
                      </p>
                    )}
                    <pre className={`${pre} max-h-48`}>{detail.answers[cell.answer] || "(empty)"}</pre>
                    <GradeDetails details={cell.details} />
                  </div>
                </li>
              );
            })}
          </ul>
        </Block>
      </div>
    </div>
  );
}
