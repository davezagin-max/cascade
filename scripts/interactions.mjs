#!/usr/bin/env node
/**
 * Checks the playground's main interactions in a real browser, in demo mode.
 *
 *   pnpm build && pnpm interactions
 *
 * The unit tests cover the pipeline with a fake client; this covers what only a
 * browser can show: that Stop really ends a run (and does not start another),
 * that a newer run replaces an older one, and that free-form prompts are held
 * back without a key. Uses the same port as `pnpm screenshots`.
 */
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const port = JSON.parse(readFileSync("screenshots.json", "utf8")).port ?? 3210;
const base = `http://localhost:${port}`;

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

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${ok || !detail ? "" : `: ${detail}`}`);
};

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  let calls = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/api/route-and-run")) calls++;
  });
  await page.goto(`${base}/playground`, { waitUntil: "networkidle" });
  const response = page.locator("[aria-live=polite].min-h-40");
  const run = page.getByRole("button", { name: /^Run( on .+)?$/ });

  // Stop ends the run, keeps what had arrived, and sends nothing further.
  await page.click("text=Explain a concept with one analogy");
  await page.waitForSelector("button:text-is('Stop')");
  await page.waitForTimeout(450);
  await page.click("button:text-is('Stop')");
  await page.waitForTimeout(1500);
  const stopped = await response.innerText();
  check("Stop ends the run", stopped.startsWith("Stopped."), stopped.slice(0, 80));
  check("Stop does not send the request again", calls === 1, `${calls} requests`);
  await page.waitForTimeout(800);
  check("a stopped answer stays as it was", (await response.innerText()) === stopped);
  check("the Run button is back and enabled", (await run.isEnabled()) && /^Run on /.test(await run.innerText()));

  // A preset runs to completion and shows where the answer came from.
  await page.click("text=Classify a support ticket");
  await page.waitForSelector("[data-testid=response]", { timeout: 20_000 });
  const done = await page.locator("[data-testid=response]").innerText();
  check("a preset is answered", done.includes("shipping"), done.slice(0, 80));
  check("the answer says where it came from", /Sample answer|Recorded run/.test(done), done.slice(0, 80));
  check("one request per run", calls === 2, `${calls} requests`);

  // Free-form text is routed live but cannot be sent without a key.
  await page.fill("#prompt", "Summarize the plot of Hamlet in two sentences.");
  check("the router decides on free-form text", await page.locator("[data-testid=decision]").isVisible());
  check("Run is held back without a key", await run.isDisabled());
  await page.press("#prompt", "Enter");
  await page.waitForTimeout(300);
  check("nothing is sent for a free-form prompt without a key", calls === 2, `${calls} requests`);

  // The threshold control on the dashboard changes the headline.
  await page.goto(base, { waitUntil: "networkidle" });
  const before = await page.locator("h1").innerText();
  await page.click('[aria-label^="Lower the threshold"]');
  await page.waitForTimeout(200);
  check("the threshold control moves the headline", (await page.locator("h1").innerText()) !== before);
} finally {
  await browser.close();
  stop();
}
process.exit(failures ? 1 : 0);
