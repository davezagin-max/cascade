import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { ZERO_USAGE } from "@/lib/config";
import type { Fixture } from "@/lib/demo";
import { Body, Output, PlaygroundEventSchema } from "@/lib/playground";
import { ROUTER, route } from "@/lib/router/model";

const read = (file: string) => JSON.parse(readFileSync(path.join(process.cwd(), "fixtures", file), "utf8"));
const store = read("route-and-run.json") as Record<string, Fixture>;
const presets = read("presets.json") as { route: string; label: string; body: z.infer<typeof Body> }[];

describe("playground fixtures", () => {
  it("has a fixture for every preset and a preset for every fixture", () => {
    expect(presets.length).toBeGreaterThanOrEqual(4);
    expect(presets.every((preset) => preset.route === "/api/route-and-run")).toBe(true);
    expect(presets.every((preset) => preset.label.length > 3)).toBe(true);
    expect(new Set(presets.map((preset) => preset.body.demoId))).toEqual(new Set(Object.keys(store)));
  });

  it.each(presets.map((preset) => [preset.body.demoId as string, preset] as const))(
    "%s is a valid output of the real pipeline",
    (demoId, preset) => {
      const body = Body.parse(preset.body);
      const fixture = store[demoId];
      const data = Output.parse(fixture.data);
      const events = (fixture.events ?? []).map(({ event }) => PlaygroundEventSchema.parse(event));

      // The stream is one decision, then the answer in pieces that add up to the final text.
      expect(events[0]).toEqual({ type: "decision", arm: data.arm, routerVersion: data.routerVersion });
      const text = events.flatMap((event) => (event.type === "text" ? [event.delta] : [])).join("");
      expect(text).toBe(data.answer);
      expect(events.filter((event) => event.type === "decision")).toHaveLength(1);
      const offsets = (fixture.events ?? []).map(({ at }) => at);
      expect(offsets).toEqual([...offsets].sort((a, b) => a - b));

      // The stored arm is a real arm, and the model that answered is that arm's model.
      const arm = ROUTER.arms.find((candidate) => candidate.key === data.arm.key);
      if (data.routerVersion === ROUTER.version) {
        // Same router as the one in the repo: the decision must be the one it makes today.
        expect(route(ROUTER, body.prompt).arm.key).toBe(data.arm.key);
        expect(arm).toMatchObject({ model: data.arm.model, effort: data.arm.effort });
      }

      // Honest bookkeeping, checked against the fixture's own flag.
      if (fixture.placeholder) {
        expect(fixture.recordedAt).toBeNull();
        expect(fixture.usage).toEqual(ZERO_USAGE);
        expect(fixture.costUsd).toBe(0);
        expect(data.comparison).toBeNull();
      } else {
        expect(fixture.recordedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
        expect(fixture.usage.output_tokens).toBeGreaterThan(0);
        expect(fixture.costUsd).toBeGreaterThan(0);
      }
    },
  );

  it("covers several routing outcomes while it matches the current router", () => {
    const current = Object.values(store).filter(
      (fixture) => Output.parse(fixture.data).routerVersion === ROUTER.version,
    );
    if (current.length < presets.length) return; // recorded under an older router: coverage is historical
    const arms = new Set(presets.map((preset) => route(ROUTER, preset.body.prompt).arm.key));
    expect(arms.size).toBeGreaterThanOrEqual(4);
    const costs = [...arms].map((key) => ROUTER.arms.findIndex((arm) => arm.key === key));
    expect(Math.min(...costs)).toBe(0); // the cheapest arm...
    expect(Math.max(...costs)).toBe(ROUTER.arms.length - 1); // ...and the most expensive are both shown
  });
});
