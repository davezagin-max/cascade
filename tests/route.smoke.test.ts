import { describe, expect, it } from "vitest";
import { POST } from "@/app/api/route-and-run/route";
import fixtures from "@/fixtures/route-and-run.json";
import presets from "@/fixtures/presets.json";
import type { StreamFrame } from "@/lib/api";
import type { Fixture } from "@/lib/demo";
import type { PlaygroundEvent, PlaygroundOutput } from "@/lib/playground";

type Frame = StreamFrame<PlaygroundOutput, PlaygroundEvent>;

const post = (body: unknown, headers: Record<string, string> = {}) =>
  new Request("http://localhost/api/route-and-run", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify(body),
  });

async function frames(res: Response): Promise<Frame[]> {
  const text = await res.text();
  return text
    .split("\n\n")
    .filter((chunk) => chunk.startsWith("data: "))
    .map((chunk) => JSON.parse(chunk.slice(6)) as Frame);
}

/**
 * The main path with no key: a visitor clicks a preset and watches the router's
 * decision arrive, then the answer, then the final result.
 */
describe("POST /api/route-and-run in demo mode", () => {
  const preset = presets[0];
  const fixture = (fixtures as Record<string, Fixture<PlaygroundOutput>>)[preset.body.demoId];

  it("replays a preset: decision first, then the answer, then the result", async () => {
    const res = await POST(post(preset.body));
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toContain("text/event-stream");
    const all = await frames(res);

    // Whether the banner says "sample" or "recorded" comes from the fixture, never from the UI.
    expect(all[0]).toEqual({
      type: "meta",
      meta: { mode: "demo", placeholder: fixture.placeholder, recordedAt: fixture.recordedAt },
    });
    const events = all.flatMap((frame) => (frame.type === "event" ? [frame.event] : []));
    expect(events[0]).toMatchObject({ type: "decision", arm: fixture.data.arm });
    const streamed = events.flatMap((event) => (event.type === "text" ? [event.delta] : [])).join("");
    expect(streamed).toBe(fixture.data.answer);

    const done = all.at(-1);
    expect(done?.type).toBe("done");
    if (done?.type !== "done") return;
    expect(done.data).toEqual(fixture.data);
    expect(done.meta).toMatchObject({
      mode: "demo",
      placeholder: fixture.placeholder,
      costUsd: fixture.costUsd,
      model: fixture.model,
    });
  }, 30_000);

  it("explains that a free-form prompt needs a key, without calling anything", async () => {
    const res = await POST(post({ prompt: "What is the capital of Australia?" }));
    expect(res.status).toBe(404);
    const body = await res.json();
    expect(body).toMatchObject({ ok: false, error: { kind: "demo_miss" } });
    expect(body.error.message).toContain("add your Anthropic key");
  });

  it("rejects malformed bodies and malformed keys", async () => {
    expect((await POST(post({ prompt: "" }))).status).toBe(400);
    expect((await POST(post({ prompt: "x".repeat(4001) }))).status).toBe(400);
    expect((await POST(post({ topic: "wrong field" }))).status).toBe(400);
    const badKey = await POST(post({ prompt: "hello" }, { "x-anthropic-key": "definitely-not-a-key" }));
    expect(badKey.status).toBe(401);
    expect(await badKey.text()).not.toContain("definitely-not-a-key");
  });
});
