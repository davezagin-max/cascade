import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SimTag } from "@/components/SimTag";
import { SyntheticBanner } from "@/components/SiteChrome";
import summaryJson from "@/data/summary.json";

vi.mock("next/link", () => ({
  default: ({ href, children, className }: { href: string; children: ReactNode; className?: string }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));
vi.mock("@/components/NavLinks", () => ({ NavLinks: () => null }));

const text = (html: string) => html.replace(/<[^>]+>/g, "").replace(/\s+/g, " ").trim();

async function renderHome(synthetic: boolean): Promise<string> {
  vi.resetModules();
  vi.doMock("@/lib/run", () => {
    const run = structuredClone(summaryJson) as typeof summaryJson;
    run.meta.synthetic = synthetic;
    if (!synthetic) {
      (run.meta as { generator: unknown }).generator = null;
      run.meta.run_id = "run-20261101-090000";
    } else if (!run.meta.generator) {
      // The committed export is a real run: give the forced-synthetic copy a simulator's metadata.
      (run.meta as { generator: unknown }).generator = {
        name: "cascade simulate",
        seed: 7,
        note: "Deterministic simulator. Not a measurement.",
      };
    }
    if (run.caching) run.caching.synthetic = synthetic;
    return { RUN: run, SYNTHETIC: synthetic };
  });
  const { default: Home } = await import("@/app/page");
  return renderToStaticMarkup(<Home />);
}

afterEach(() => {
  vi.doUnmock("@/lib/run");
});

/**
 * The dashboard must never present simulator output as a measurement, and must
 * stop saying "simulated" the moment a real run is loaded, with no code change.
 */
describe("synthetic data is labelled from the run's own flag", () => {
  it("shows the exact warning the banner is required to carry", () => {
    expect(text(renderToStaticMarkup(<SyntheticBanner />))).toBe(
      "Synthetic placeholder data. These are not measured results. Run uv run cascade run with an API key to replace them.",
    );
  });

  it("renders the tag only when asked to", () => {
    expect(renderToStaticMarkup(<SimTag show={false} />)).toBe("");
    expect(text(renderToStaticMarkup(<SimTag show />))).toBe("simulated");
  });

  it("tags the headline and every results section when the run is synthetic", async () => {
    const html = await renderHome(true);
    const tags = html.match(/>simulated</g) ?? [];
    // Headline, run facts, arms, categories, latency, router, and caching when the run has it.
    expect(tags.length).toBeGreaterThanOrEqual(summaryJson.caching ? 7 : 6);
    expect(text(html)).toContain("No real run exists yet");
    expect(text(html)).toContain("Not a measurement");
  });

  it("says nothing about simulation when the run is real", async () => {
    const html = await renderHome(false);
    expect(html.match(/>simulated</g)).toBeNull();
    const words = text(html).toLowerCase();
    expect(words).not.toContain("simulat");
    expect(words).not.toContain("synthetic");
    expect(words).not.toContain("no real run exists");
    expect(words).toContain("the router saves");
  });

  it("puts the harder comparison next to the headline, in both states", async () => {
    const fixed = summaryJson.router.headline.best_fixed;
    for (const synthetic of [true, false]) {
      const words = text(await renderHome(synthetic));
      // The easy win (versus always Opus) never stands alone: the best affordable arm is named too.
      expect(words).toContain("against the best single arm at this cost or less");
      if (fixed) {
        const delta = summaryJson.router.headline.quality - fixed.quality;
        const points = Math.abs(delta * 100).toFixed(1);
        const expected = points === "0.0" ? "level with" : `${points} points ${delta > 0 ? "ahead of" : "behind"}`;
        expect(words).toContain(`The router is ${expected} always using`);
      }
    }
  });
});
