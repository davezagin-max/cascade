import presets from "@/fixtures/presets.json";
import fixtures from "@/fixtures/route-and-run.json";
import { streamRoute } from "@/lib/api";
import type { FixtureStore } from "@/lib/demo";
import { Body, runPlayground, type PlaygroundEvent, type PlaygroundOutput } from "@/lib/playground";

/**
 * POST /api/route-and-run
 *
 * The router decides locally, then the chosen model answers at the chosen effort.
 * Events: one `decision`, then `text` deltas. With no key, a preset's fixture is replayed.
 */
export const maxDuration = 300;

const presetPrompts = new Map(
  presets.filter((preset) => preset.route === "/api/route-and-run").map((preset) => [preset.body.demoId, preset.body.prompt]),
);

/**
 * A preset's `demoId` is the key its answer is replayed (and recorded) under, so it is only
 * accepted together with that preset's own prompt. Otherwise a request could pair a known id
 * with any text and be shown, or record, an answer to a different question.
 */
const RequestBody = Body.refine(
  (body) => body.demoId === undefined || !presetPrompts.has(body.demoId) || presetPrompts.get(body.demoId) === body.prompt,
);

export const POST = streamRoute({
  store: "route-and-run",
  fixtures: fixtures as FixtureStore<PlaygroundOutput, PlaygroundEvent>,
  schema: RequestBody,
  replaySpeed: 3,
  live: ({ body, client, req, emit }) =>
    runPlayground({ prompt: body.prompt, client, emit, signal: req.signal }),
});
