#!/usr/bin/env node
/**
 * Fails if any page scrolls sideways on a phone, before or after interaction.
 *
 *   pnpm build && pnpm overflow
 *
 * Opens each page at 360 and 390 px wide, performs the page's main interaction,
 * and compares the document's scroll width with the viewport. Uses the same port
 * as `pnpm screenshots`.
 */
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const port = JSON.parse(readFileSync("screenshots.json", "utf8")).port ?? 3210;
const base = `http://localhost:${port}`;

const PAGES = [
  {
    path: "/",
    steps: async (page) => {
      await page.click('[aria-label^="Lower the threshold"]');
      await page.click('[aria-label^="Lower the threshold"]');
      await page.hover("svg [role=img]");
    },
  },
  {
    path: "/tasks",
    steps: async (page) => {
      await page.click("text=Semantic version precedence");
      await page.waitForSelector("text=Reference answer and format contract");
      await page.click("text=Reference answer and format contract");
    },
  },
  {
    path: "/playground",
    steps: async (page) => {
      await page.click("text=Write a linear-time merge");
      await page.waitForSelector("[data-testid=response]", { timeout: 20_000 });
      await page.fill("#prompt", `${"A very long unbroken token ".repeat(12)}${"x".repeat(240)}`);
    },
  },
];

const server = spawn("pnpm", ["exec", "next", "start", "-p", String(port)], {
  stdio: "ignore",
  detached: true,
});
const stop = () => {
  try {
    process.kill(-server.pid, "SIGTERM");
  } catch {}
};
process.on("exit", stop);

for (let i = 0; ; i++) {
  try {
    await fetch(base);
    break;
  } catch {
    if (i > 80) throw new Error("Server did not start. Run `pnpm build` first.");
    await new Promise((r) => setTimeout(r, 250));
  }
}

const measure = (page) =>
  page.evaluate(() => {
    const doc = document.documentElement;
    const wide = [...document.querySelectorAll("body *")]
      .filter((el) => el.getBoundingClientRect().right > doc.clientWidth + 1)
      .filter((el) => {
        // Content inside its own horizontal scroller is fine; the page itself must not scroll.
        for (let p = el.parentElement; p; p = p.parentElement) {
          const overflow = getComputedStyle(p).overflowX;
          if (overflow === "auto" || overflow === "scroll" || overflow === "hidden") return false;
        }
        return true;
      })
      .slice(0, 3)
      .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 60)}`);
    return { scroll: doc.scrollWidth, client: doc.clientWidth, wide };
  });

let failures = 0;
const browser = await chromium.launch();
try {
  for (const width of [360, 390]) {
    for (const spec of PAGES) {
      const context = await browser.newContext({ viewport: { width, height: 780 }, deviceScaleFactor: 2 });
      const page = await context.newPage();
      await page.goto(base + spec.path, { waitUntil: "networkidle" });
      for (const stage of ["on load", "after interacting"]) {
        if (stage === "after interacting") {
          await spec.steps(page);
          await page.waitForTimeout(300);
        }
        const result = await measure(page);
        const ok = result.scroll <= result.client;
        if (!ok) failures++;
        console.log(
          `${ok ? "ok  " : "FAIL"} ${width}px ${spec.path} ${stage}: scrollWidth ${result.scroll} vs ${result.client}` +
            (ok ? "" : `  offenders: ${result.wide.join(", ")}`),
        );
      }
      await context.close();
    }
  }
} finally {
  await browser.close();
  stop();
}
process.exit(failures ? 1 : 0);
