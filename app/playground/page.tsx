import type { Metadata } from "next";
import { DemoBanner } from "@/components/kit/DemoBanner";
import { Playground, type Preset } from "@/components/playground/Playground";
import fixtures from "@/fixtures/route-and-run.json";
import presetList from "@/fixtures/presets.json";
import type { Fixture } from "@/lib/demo";
import { PROMPT_MAX_CHARS } from "@/lib/playground";
import { ROUTER } from "@/lib/router/model";
import { SYNTHETIC } from "@/lib/run";

export const metadata: Metadata = { title: "Playground" };

const store = fixtures as Record<string, Fixture>;

/** The banner says "sample" for as long as any preset is still a hand-written placeholder. */
const anyPlaceholder = Object.values(store).some((fixture) => fixture.placeholder);

interface PresetEntry {
  route: string;
  label?: string;
  body: { demoId: string; prompt: string };
}

/** Labels come from presets.json: recording a real run replaces the fixture, not the preset. */
const presets: Preset[] = (presetList as PresetEntry[])
  .filter((preset) => preset.route === "/api/route-and-run" && store[preset.body.demoId])
  .map((preset) => ({
    demoId: preset.body.demoId,
    label: preset.label ?? preset.body.demoId,
    prompt: preset.body.prompt,
  }));

/** Said in the demo line while the router behind every choice was trained on the simulator's run. */
const trained =
  ROUTER.trained_on.synthetic || SYNTHETIC ? ", and the router was trained on simulated test results" : "";

export default function PlaygroundPage() {
  return (
    <>
      <DemoBanner
        placeholder={anyPlaceholder}
        liveLabel="Live: the model Cascade picks answers with your key."
        demoLabels={{
          sample: `Demo: the answers here are pre-written examples${trained}. Add your Anthropic API key to run any prompt live.`,
          recorded: `Demo: the answers here are recorded runs of the chosen model${trained}. Add your Anthropic API key to run any prompt live.`,
        }}
      />
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pb-4 pt-6 sm:pt-8">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">Playground</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-3xl">
          Watch Cascade pick a model for a prompt
        </h1>
        <p className="mb-6 mt-2 max-w-3xl text-sm text-muted">
          Cascade reads the prompt, predicts how likely each Claude model and effort setting is to get it right, and
          picks the cheapest one that clears a {Math.round(ROUTER.threshold * 100)}% bar. That choice is simple
          arithmetic, shown here in full. Then the chosen model answers.
        </p>
        <Playground presets={presets} maxChars={PROMPT_MAX_CHARS} />
      </main>
    </>
  );
}
