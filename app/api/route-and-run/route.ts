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

export const POST = streamRoute({
  store: "route-and-run",
  fixtures: fixtures as FixtureStore<PlaygroundOutput, PlaygroundEvent>,
  schema: Body,
  replaySpeed: 3,
  live: ({ body, client, req, emit }) =>
    runPlayground({ prompt: body.prompt, client, emit, signal: req.signal }),
});
