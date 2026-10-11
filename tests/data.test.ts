import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import routerJson from "@/data/router.json";
import summaryJson from "@/data/summary.json";
import { RunSummary, TaskDetail } from "@/lib/run-schema";

const summary = RunSummary.parse(summaryJson);
const taskDir = path.join(process.cwd(), "public", "run", "tasks");

describe("exported run", () => {
  it("matches the schema the dashboard is written against", () => {
    expect(summary.arms.length).toBeGreaterThanOrEqual(2);
    expect(summary.tasks.length).toBe(summary.meta.n_tasks);
    expect(summary.meta.n_rows).toBe(summary.meta.batch_rows + summary.meta.sync_rows);
  });

  it("is labelled consistently as synthetic or real everywhere", () => {
    const { synthetic, generator } = summary.meta;
    expect(routerJson.trained_on.synthetic).toBe(synthetic);
    expect(routerJson.trained_on.run_id).toBe(summary.meta.run_id);
    if (summary.caching) expect(summary.caching.synthetic).toBe(synthetic);
    // A synthetic run must say what generated it; a real run must not carry a generator.
    if (synthetic) expect(generator?.note).toContain("Not a measurement");
    else expect(generator).toBeNull();
  });

  it("serves the same router the summary describes", () => {
    expect(routerJson.version).toBe(summary.router.version);
    expect(routerJson.threshold).toBe(summary.router.threshold);
    expect(routerJson.arms.map((arm) => arm.key)).toEqual(summary.router.confusion.predicted);
    expect(new Set(routerJson.arms.map((arm) => arm.key))).toEqual(new Set(summary.arms.map((arm) => arm.key)));
  });

  it("marks exactly the undominated arms as the frontier", () => {
    for (const arm of summary.arms) {
      const dominated = summary.arms.some(
        (other) =>
          other.key !== arm.key &&
          other.cost_per_request <= arm.cost_per_request &&
          other.quality >= arm.quality &&
          (other.cost_per_request < arm.cost_per_request || other.quality > arm.quality),
      );
      expect(arm.on_frontier, arm.key).toBe(!dominated);
    }
  });

  it("has a sweep that trades cost for quality in the right direction", () => {
    const { sweep, headline, threshold, baseline } = summary.router;
    const costs = sweep.map((point) => point.cost_per_request);
    expect(costs).toEqual([...costs].sort((a, b) => a - b));
    const selected = sweep.find((point) => point.threshold === threshold);
    expect(selected).toEqual({ threshold, ...headline });
    expect(headline.cost_saved).toBeCloseTo(1 - headline.cost_per_request / baseline.cost_per_request, 6);
    expect(headline.quality_retained).toBeCloseTo(headline.quality / baseline.quality, 6);
    for (const point of sweep) {
      expect(Object.values(point.mix).reduce((a, b) => a + b, 0)).toBe(summary.router.n_tasks);
    }
  });

  it("has a confusion matrix that accounts for every task", () => {
    const { matrix, exact, over_routed, under_routed, predicted, oracle } = summary.router.confusion;
    expect(matrix.length).toBe(oracle.length);
    expect(matrix.every((row) => row.length === predicted.length)).toBe(true);
    expect(matrix.flat().reduce((a, b) => a + b, 0)).toBe(summary.router.n_tasks);
    expect(exact + over_routed + under_routed).toBe(summary.router.n_tasks);
  });

  it("publishes one valid detail file per task", () => {
    const files = readdirSync(taskDir).filter((name) => name.endsWith(".json"));
    expect(new Set(files)).toEqual(new Set(summary.tasks.map((task) => `${task.id}.json`)));
    const perTask = summary.arms.length * summary.meta.trials;
    for (const file of files) {
      const detail = TaskDetail.parse(JSON.parse(readFileSync(path.join(taskDir, file), "utf8")));
      expect(detail.cells.length, file).toBe(perTask);
      for (const cell of detail.cells) {
        expect(detail.answers[cell.answer], file).toBeTypeOf("string");
        // Batch rows cannot be timed; sync rows always are.
        expect(cell.latency_ms === null, file).toBe(cell.mode === "batch");
        // In a synthetic run every cell says so, and says what it was drawn from.
        expect("simulated" in cell.details, file).toBe(summary.meta.synthetic);
      }
    }
  });
});
