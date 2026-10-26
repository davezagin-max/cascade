#!/usr/bin/env node
/**
 * Checks every feature a visitor can reach, against a running production build.
 *
 *   pnpm build && pnpm exec next start -p 3150
 *   pnpm functional                      (BASE_URL overrides http://localhost:3150)
 *
 * Headless Chromium at desktop (1280x800) and phone (390x844, 360 too) widths, in light
 * and dark themes. After each action it asserts a visible outcome: text, counts, values
 * checked against data/summary.json, URLs, or request counts. It fails on console errors,
 * uncaught page errors, failed responses from this app, and any sideways scroll.
 *
 * Nothing here calls Claude. The playground only replays fixtures. A network guard aborts
 * every request that leaves the app's origin, and the report counts what it blocked.
 * Requests to /api/route-and-run are spaced so the app's 20-per-minute limit never trips
 * during the normal run; the rate-limit check runs last, after the window has cleared.
 */
import { readFileSync } from "node:fs";
import { chromium } from "playwright";

const BASE = (process.env.BASE_URL ?? "http://localhost:3150").replace(/\/$/, "");
const ROOT = new URL("..", import.meta.url);
const summary = JSON.parse(readFileSync(new URL("data/summary.json", ROOT), "utf8"));
const fixtures = JSON.parse(readFileSync(new URL("fixtures/route-and-run.json", ROOT), "utf8"));
const presetList = JSON.parse(readFileSync(new URL("fixtures/presets.json", ROOT), "utf8"));

const SWEEP = summary.router.sweep;
const BASELINE_ARM = summary.router.baseline_arm;
const TASK_IDS = summary.tasks.map((task) => task.id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const norm = (text) => text.replace(/\s+/g, " ").trim();
const pct = (fraction, digits = 1) => `${(fraction * 100).toFixed(digits)}%`;
const armName = (key) => {
  const [model, effort] = key.split("-");
  const names = { haiku: "Haiku", sonnet: "Sonnet", opus: "Opus" };
  return `${names[model]}, ${effort} effort`;
};
const armShort = (key) => {
  const [model, effort] = key.split("-");
  const names = { haiku: "Haiku", sonnet: "Sonnet", opus: "Opus" };
  return `${names[model]} · ${effort}`;
};
/** The text an answer shows once Markdown fences and emphasis are stripped. */
const plain = (text) =>
  norm(text.replace(/```[a-zA-Z]*\n?/g, "").replace(/\*\*/g, "").replace(/`/g, ""));

// ---- reporting ----------------------------------------------------------------------

const results = [];
let group = "";
function check(name, ok, detail = "") {
  results.push({ group, name, ok: Boolean(ok), detail });
  const tag = ok ? "ok  " : "FAIL";
  const suffix = ok || !detail ? "" : `  -- ${String(detail).slice(0, 240)}`;
  console.log(`${tag} [${group}] ${name}${suffix}`);
}
function heading(title) {
  group = title;
  console.log(`\n== ${title}`);
}

// ---- problems the browser reports (console errors, page errors, failed responses) ----

const problems = [];
/** Requests we cause on purpose to fail (the 404 page, a failing task fetch). */
let expectFailures = false;
const expectedFailureUrl = /\/nope-not-a-page|\/run\/tasks\/no-such-task\.json|\/run\/tasks\/cls-sarcasm\.json/;

function watch(page, where) {
  page.on("pageerror", (error) => problems.push({ where, text: `uncaught: ${error.message}` }));
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    if (expectFailures && /Failed to load resource/.test(message.text())) return;
    problems.push({ where, text: `console: ${message.text()}` });
  });
  page.on("response", (response) => {
    const url = response.url();
    if (!url.startsWith(BASE) || response.status() < 400) return;
    if (expectFailures && expectedFailureUrl.test(url)) return;
    problems.push({ where, text: `HTTP ${response.status()} ${url}` });
  });
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url() === `${BASE}/api/route-and-run`) postTimes.push(Date.now());
  });
}

function drainProblems(label) {
  const found = problems.splice(0, problems.length);
  check(
    `${label}: no console errors, page errors, or failed responses`,
    found.length === 0,
    found.map((p) => `${p.where}: ${p.text}`).join(" | "),
  );
}

// ---- rate limit pacing for /api/route-and-run (20 per minute per client) --------------

const postTimes = [];
async function pacePost() {
  for (;;) {
    const now = Date.now();
    while (postTimes.length && now - postTimes[0] > 60_000) postTimes.shift();
    if (postTimes.length < 17) return;
    await sleep(60_000 - (now - postTimes[0]) + 300);
  }
}

/**
 * The limiter lives in the server's memory, so a window left open by an earlier run would
 * refuse this run's first requests. Probe with an empty body (cheap, no model call) and
 * wait out any window the server reports before starting.
 */
async function clearRateWindow() {
  // A probe can be accepted while the window still holds up to 19 requests from an earlier
  // run, so first wait out a whole window; then probe as a last check.
  await sleep(61_000);
  for (;;) {
    const res = await fetch(`${BASE}/api/route-and-run`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    });
    if (res.status !== 429) {
      postTimes.push(Date.now());
      return;
    }
    await sleep((Number(res.headers.get("retry-after") ?? 60) + 1) * 1000);
  }
}
await clearRateWindow();

// ---- browsers and pages -----------------------------------------------------------------

const browser = await chromium.launch();
const contexts = [];
/** Requests that would have left the app's origin. Expected to stay at zero. */
const blockedExternal = [];

async function newContext({ width, height, scheme = "light" }) {
  const context = await browser.newContext({
    viewport: { width, height },
    colorScheme: scheme,
    deviceScaleFactor: 1,
  });
  await context.route("**/*", (route) => {
    const url = route.request().url();
    if (url.startsWith(BASE) || url.startsWith("data:") || url.startsWith("blob:")) return route.continue();
    blockedExternal.push(url);
    return route.abort();
  });
  contexts.push(context);
  return context;
}

async function newPage(options) {
  const context = await newContext(options);
  const page = await context.newPage();
  const where = `${options.width}px${options.scheme === "dark" ? " dark" : ""}`;
  watch(page, where);
  return page;
}

async function open(page, path, { wait = "networkidle" } = {}) {
  const response = await page.goto(`${BASE}${path}`, { waitUntil: wait });
  return response;
}

/** Fails if the page scrolls sideways. Called after each interaction at phone widths. */
async function noSideScroll(page, label) {
  const result = await page.evaluate(() => {
    const doc = document.documentElement;
    const offenders = [...document.querySelectorAll("body *")]
      .filter((el) => el.getBoundingClientRect().right > doc.clientWidth + 1)
      .filter((el) => {
        for (let p = el.parentElement; p; p = p.parentElement) {
          const overflow = getComputedStyle(p).overflowX;
          if (overflow === "auto" || overflow === "scroll" || overflow === "hidden") return false;
        }
        return true;
      })
      .slice(0, 3)
      .map((el) => `${el.tagName.toLowerCase()}.${String(el.className).slice(0, 50)}`);
    return { scroll: doc.scrollWidth, client: doc.clientWidth, offenders };
  });
  check(
    `${label}: no sideways scroll`,
    result.scroll <= result.client,
    `scrollWidth ${result.scroll} > ${result.client} ${result.offenders.join(", ")}`,
  );
}

/** Waits for a finished answer; if none comes, reports what the page said instead. */
async function waitForAnswer(page) {
  try {
    await page.waitForSelector("[data-testid=response]", { timeout: 30_000 });
  } catch {
    const alert = await page.locator("[role=alert]").first().innerText().catch(() => "");
    throw new Error(`no answer arrived${alert ? `; the page said: ${norm(alert)}` : ""}`);
  }
}

// ---- helpers for the dashboard ------------------------------------------------------

const headlineText = (page) => page.locator('[data-testid=headline]').innerText().then(norm);
const thresholdOutput = (page) => page.locator("output").first().innerText().then(norm);
const ring = (page) => page.locator('[role=img][aria-label^="Cascade at "]').first();

function expectedPoint(index) {
  const point = SWEEP[index];
  const cs = point.cost_saved;
  return {
    threshold: point.threshold,
    headlineParts: [pct(Math.abs(cs)), pct(point.quality_retained)],
    ringQuality: pct(point.quality),
    label: pct(point.threshold, 0),
    point,
  };
}

async function expectSweepIndex(page, index, label) {
  const want = expectedPoint(index);
  const headline = await headlineText(page);
  const output = await thresholdOutput(page);
  const ringLabel = (await ring(page).getAttribute("aria-label")) ?? "";
  const valuetext = (await page.locator("#threshold").getAttribute("aria-valuetext")) ?? "";
  check(
    `${label}: the slider shows ${want.label} and the range is index ${index}`,
    output === want.label && (await page.locator("#threshold").inputValue()) === String(index),
    `output ${output}, value ${await page.locator("#threshold").inputValue()}`,
  );
  check(
    `${label}: headline shows ${want.headlineParts[0]} lower cost and ${want.headlineParts[1]} of quality`,
    want.headlineParts.every((part) => headline.includes(part)),
    headline,
  );
  check(
    `${label}: the chart ring moves to ${want.label} sure (${want.ringQuality} of tasks passed)`,
    ringLabel.startsWith(`Cascade at ${want.label} sure`) && ringLabel.includes(want.ringQuality),
    ringLabel,
  );
  check(`${label}: the slider's spoken value matches the headline`, valuetext.startsWith(`${want.label} sure`), valuetext);
}

// ---- 1. Dashboard at desktop width, light theme -----------------------------------------

heading("Dashboard, desktop light");
const desk = await newPage({ width: 1280, height: 800 });
await open(desk, "/");

check("home: page title names the product", (await desk.title()).startsWith("Cascade"), await desk.title());
check(
  "home: first heading is the one-sentence pitch",
  (await desk.locator("h1").innerText()).includes("Not every request needs the most expensive model."),
);
check(
  "synthetic notice carries the brief's exact sentence",
  norm(await desk.locator('[data-testid=synthetic-banner]').innerText()).includes(
    "Synthetic placeholder data. These are not measured results. Run uv run cascade run with an API key to replace them.",
  ),
);
const sweepIndexDefault = SWEEP.findIndex((point) => Math.abs(point.threshold - summary.router.threshold) < 1e-9);
check("slider starts at the default threshold of 0.90 (data)", sweepIndexDefault >= 0, String(summary.router.threshold));
await expectSweepIndex(desk, sweepIndexDefault, "threshold default");

const headlineNow = await headlineText(desk);
check(
  "headline: router saves a percentage of cost and keeps a percentage of quality (simulated)",
  /\d+\.\d% lower cost \d+\.\d% of the quality kept/.test(headlineNow),
  headlineNow,
);
check(
  "result card carries a simulated tag",
  (await desk.locator("h2#result-title").innerText()).includes("SIMULATED"),
);
check(
  "tougher comparison names the best affordable setting and the gap in points",
  await (async () => {
    const text = norm(await desk.locator('[data-testid=best-fixed]').innerText());
    const point = SWEEP[sweepIndexDefault];
    const gap = (Math.abs(point.quality - point.best_fixed.quality) * 100).toFixed(1);
    const want = `${gap} points ${point.quality < point.best_fixed.quality ? "behind" : "ahead of"}`;
    return text.includes(want) && text.includes(armName(point.best_fixed.arm)) && text.includes(pct(point.best_fixed.quality));
  })(),
  await desk.locator('[data-testid=best-fixed]').innerText(),
);
const massDetail = await desk.locator('[role=img][aria-label*=": "]').evaluateAll((els) =>
  els.map((el) => el.getAttribute("aria-label")),
);
const mixLabel = massDetail.find((label) => label.includes("effort: ") && !label.includes("%")) ?? "";
const mixTotal = [...mixLabel.matchAll(/: (\d+)/g)].reduce((sum, match) => sum + Number(match[1]), 0);
check(
  "task mix bar: the tasks sent to each setting add up to the 63 tasks",
  mixTotal === summary.meta.n_tasks && norm(await desk.locator("body").innerText()).includes(`Where Cascade sent the ${summary.meta.n_tasks} tasks`),
  `sum ${mixTotal}`,
);
// Step the slider down and back up with the buttons; each step must move the headline and the ring.
await desk.click('[aria-label^="Lower the threshold"]');
await expectSweepIndex(desk, sweepIndexDefault - 1, "after − once");
await desk.click('[aria-label^="Raise the threshold"]');
await desk.click('[aria-label^="Raise the threshold"]');
await expectSweepIndex(desk, sweepIndexDefault + 1, "after + twice");
check(
  "+ is disabled at the top of the range",
  await (async () => {
    while (!(await desk.locator('[aria-label^="Raise the threshold"]').isDisabled())) {
      await desk.click('[aria-label^="Raise the threshold"]');
    }
    return true;
  })(),
);
await expectSweepIndex(desk, SWEEP.length - 1, "top of range");
for (let i = 0; i < SWEEP.length - 1; i++) await desk.click('[aria-label^="Lower the threshold"]');
await expectSweepIndex(desk, 0, "bottom of range");
check("− is disabled at the bottom of the range", await desk.locator('[aria-label^="Lower the threshold"]').isDisabled());
await desk.locator("#threshold").focus();
await desk.keyboard.press("ArrowRight");
await expectSweepIndex(desk, 1, "keyboard ArrowRight on the slider");
await desk.locator("#threshold").fill(String(sweepIndexDefault));
await expectSweepIndex(desk, sweepIndexDefault, "slider set by value to the default");

// Pareto chart: hover and keyboard focus show a tooltip with the arm's numbers.
const armPoints = desk.locator('svg [role=img][tabindex="0"]');
check("Pareto chart: 9 arm points and the router point are focusable", (await armPoints.count()) === 10, String(await armPoints.count()));
await desk.locator('[role=img][aria-label^="Opus, high effort"]').first().hover();
const tip = desk.locator('[role=tooltip]');
check(
  "Pareto chart: hovering the Opus, high effort point shows its numbers",
  (await tip.count()) > 0 && (await tip.first().innerText()).includes("per 1,000 requests"),
  (await tip.count()) ? await tip.first().innerText() : "no tooltip",
);
await desk.locator('[role=img][aria-label^="Cascade at "]').first().focus();
check(
  "Pareto chart: keyboard focus on the router point shows its tooltip",
  (await tip.first().innerText()).includes("tasks passed"),
  (await tip.count()) ? await tip.first().innerText() : "no tooltip",
);
await desk.mouse.move(5, 5);

// Arms table: 9 rows, baseline marked, frontier flags match the data.
const armRows = desk.locator("#arms tbody tr");
check("arms table: one row per model and effort (9)", (await armRows.count()) === 9, String(await armRows.count()));
const opusHighRow = norm(await armRows.filter({ hasText: "Opus · high" }).first().innerText());
const baselineArm = summary.arms.find((arm) => arm.key === BASELINE_ARM);
check(
  "arms table: baseline row shows its pass rate and cost per 1,000 requests",
  opusHighRow.includes("baseline") && opusHighRow.includes(pct(baselineArm.quality)),
  opusHighRow,
);
const frontierYes = summary.arms.filter((arm) => arm.on_frontier).length;
const shownYes = (await armRows.evaluateAll((rows) => rows.filter((row) => row.lastElementChild?.textContent === "yes").length));
check("arms table: frontier column matches the data", shownYes === frontierYes, `shown ${shownYes}, data ${frontierYes}`);

// Run facts, categories, latency, router, caching, method, footer.
const runText = norm(await desk.locator("section#run").innerText());
check(
  "run facts: tasks, rows, Batch API and synchronous counts match the data",
  runText.includes(`tasks ${summary.meta.n_tasks}`) &&
    runText.includes(`result rows ${summary.meta.n_rows.toLocaleString("en-US")}`) &&
    runText.includes(`via Batch API ${summary.meta.batch_rows.toLocaleString("en-US")} rows at half price`) &&
    runText.includes(`timed synchronously ${summary.meta.sync_rows} rows`),
  runText.slice(0, 300),
);
const categoryText = norm(await desk.locator("section#categories").innerText());
check(
  "categories: every task type and the router row appear",
  ["Classification", "Extraction", "Reasoning and math", "Code", "Open-ended"].every((name) => categoryText.includes(name)) &&
    /\brouter\b/.test(categoryText),
);
const categoryRows = await desk.locator("section#categories tbody tr").count();
check("categories: one row per arm plus the router (10)", categoryRows === 10, String(categoryRows));
const latencyText = norm(await desk.locator("section#latency").innerText());
check(
  "latency: p50 and p95 are shown for every arm",
  latencyText.includes("p50 / p95") && summary.arms.every((arm) => latencyText.includes(armShort(arm.key))),
);
const routerText = norm(await desk.locator("section#router").innerText());
check(
  "router: explains the penalty, cross-validation and the baseline in plain words",
  routerText.includes(`L2 = ${summary.router.l2}`) &&
    routerText.includes("cross-validation") &&
    routerText.includes("against the best fixed arm") &&
    routerText.includes("best in hindsight"),
);
const cachingText = norm(await desk.locator("section#caching").innerText());
check(
  "caching: shows the saving percentage and dollars from the data, with the 512-token minimum and 5-minute lifetime",
  cachingText.includes(pct(summary.caching.savings_pct)) &&
    cachingText.includes(`$${summary.caching.savings_usd.toFixed(4)}`) &&
    cachingText.includes("512-token minimum") &&
    cachingText.includes("5-minute lifetime"),
  cachingText.slice(0, 200),
);
check(
  "honesty: no line says 'Measured' as a claim while the run is synthetic",
  !(await desk.locator("body").innerText()).includes("Measured") && !/What was measured/.test(await desk.locator("body").innerText()),
);
const simulatedHeadings = await desk.locator("main section h2").evaluateAll((els) =>
  els.map((el) => [el.id, el.textContent ?? ""]),
);
const taggedIds = simulatedHeadings.filter(([, text]) => text.toLowerCase().includes("simulated")).map(([id]) => id);
check(
  "every results section heading carries a simulated tag (run, arms, categories, latency, router, caching)",
  ["run", "arms", "categories", "latency", "router", "caching"].every((id) => taggedIds.some((tagged) => tagged.startsWith(id))),
  taggedIds.join(","),
);
const footerLink = desk.locator("footer a");
check(
  "footer: pricing link opens the published rate card in a new tab",
  (await footerLink.getAttribute("href")) === "https://platform.claude.com/docs/en/about-claude/pricing" &&
    (await footerLink.getAttribute("target")) === "_blank" &&
    (await footerLink.getAttribute("rel"))?.includes("noreferrer"),
);
check("footer: pricing verification date is shown", (await desk.locator("footer").innerText()).includes("verified 2026-10-11"));

// Navigation.
const nav = desk.getByRole("navigation", { name: "Main" });
check("nav: Results, Tasks and Playground links", (await nav.locator("a").allInnerTexts()).map(norm).join("|") === "Results|Tasks|Playground");
await nav.getByRole("link", { name: "Tasks", exact: true }).click();
await desk.waitForURL(`${BASE}/tasks`);
check("nav: Tasks opens /tasks and marks it current", (await desk.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Tasks", exact: true }).getAttribute("aria-current")) === "page");
await desk.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Playground", exact: true }).click();
await desk.waitForURL(`${BASE}/playground`);
check("nav: Playground opens /playground", desk.url() === `${BASE}/playground`);
await desk.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Results", exact: true }).click();
await desk.waitForURL(`${BASE}/`);
check("nav: Results returns to the dashboard", desk.url() === `${BASE}/`);
await desk.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Tasks", exact: true }).click();
await desk.waitForURL(`${BASE}/tasks`);
await desk.getByRole("link", { name: "CASCADE" }).click();
await desk.waitForURL(`${BASE}/`);
check("nav: the CASCADE logo returns to the dashboard", desk.url() === `${BASE}/`);

// Keyboard: the skip link is first and jumps to the main content.
await open(desk, "/");
await desk.keyboard.press("Tab");
const firstFocus = await desk.evaluate(() => document.activeElement?.textContent?.trim() ?? "");
check("keyboard: the first Tab stop is the skip link", firstFocus === "Skip to content", firstFocus);
await desk.keyboard.press("Enter");
check("keyboard: the skip link moves to #main", (await desk.evaluate(() => location.hash)) === "#main");

// Calls to action.
await desk.getByRole("link", { name: "Browse the 63 test tasks" }).click();
await desk.waitForURL(`${BASE}/tasks`);
check("home: 'Browse the 63 test tasks' opens the task explorer", desk.url() === `${BASE}/tasks`);
await drainProblems("desktop light, dashboard and nav");

// The main button plays a sample in one click, with one request.
await desk.close();
const playDesk = await newPage({ width: 1280, height: 800 });
await pacePost();
await open(playDesk, "/");
const baselinePosts = postTimes.length;
await playDesk.getByRole("link", { name: "Try it on a sample prompt" }).click();
await playDesk.waitForURL(`${BASE}/playground?try=support-ticket`);
await waitForAnswer(playDesk);
const tryAnswer = norm(await playDesk.locator("[data-testid=response]").innerText());
check("home: 'Try it on a sample prompt' opens the playground at ?try=support-ticket", playDesk.url().endsWith("?try=support-ticket"));
check("home: the sample plays its answer (the ticket is classified as shipping)", tryAnswer.includes("shipping"), tryAnswer.slice(0, 120));
await sleep(600);
check("home: that click sends exactly one request", postTimes.length - baselinePosts === 1, `${postTimes.length - baselinePosts} sent`);
await drainProblems("try-it sample");
await playDesk.close();

// ---- 2. Dashboard at desktop width, dark theme (toggle) -------------------------------

heading("Theme toggle and dark theme");
const themed = await newPage({ width: 1280, height: 800 });
await open(themed, "/");
const lightBg = await themed.evaluate(() => getComputedStyle(document.body).backgroundColor);
await themed.getByRole("button", { name: "Switch to dark theme" }).click();
check("theme toggle: the page switches to dark", (await themed.evaluate(() => document.documentElement.dataset.theme)) === "dark");
const darkBg = await themed.evaluate(() => getComputedStyle(document.body).backgroundColor);
check("theme toggle: dark background differs from light", darkBg !== lightBg, `${lightBg} vs ${darkBg}`);
check("theme toggle: the button now offers light", (await themed.getByRole("button", { name: "Switch to light theme" }).count()) === 1);
check("dark: the synthetic notice and headline stay visible", (await themed.locator('[data-testid=synthetic-banner]').isVisible()) && (await headlineText(themed)).includes("lower cost"));
await themed.reload({ waitUntil: "networkidle" });
check("theme toggle: the choice survives a reload", (await themed.evaluate(() => document.documentElement.dataset.theme)) === "dark");
await themed.getByRole("button", { name: "Switch to light theme" }).click();
check("theme toggle: back to light", (await themed.evaluate(() => document.documentElement.dataset.theme)) === "light");
await drainProblems("theme toggle");
await themed.close();

const darkContext = await newPage({ width: 1280, height: 800, scheme: "dark" });
await open(darkContext, "/");
check("dark theme: follows the system preference on first visit", (await darkContext.evaluate(() => document.documentElement.dataset.theme)) === "dark");
await darkContext.click('[aria-label^="Lower the threshold"]');
await expectSweepIndex(darkContext, sweepIndexDefault - 1, "dark: after −");
await drainProblems("dark dashboard");
await darkContext.close();

// ---- 3. Tasks: filters, detail, arm picker, loading ---------------------------------

heading("Task explorer");
const tasks = await newPage({ width: 1280, height: 800 });
await open(tasks, "/tasks");
check(
  "tasks: heading names the 63 tasks and the simulated tag",
  (await tasks.locator("h1").innerText()).includes(`The ${summary.meta.n_tasks} test tasks, and how each model did`) &&
    (await tasks.locator("h1").innerText()).includes("SIMULATED"),
);
check("tasks: synthetic notice is shown above the list", await tasks.locator('[data-testid=synthetic-banner]').isVisible());

const chips = tasks.locator('[role=group][aria-label="Filter by task type"] button');
const chipTexts = (await chips.allInnerTexts()).map(norm);
const categoryCounts = {};
for (const task of summary.tasks) categoryCounts[task.category] = (categoryCounts[task.category] ?? 0) + 1;
check(
  "tasks: filter shows All 63 and each category with its count",
  chipTexts[0] === `All ${summary.meta.n_tasks}` &&
    Object.values(categoryCounts).every((n) => chipTexts.some((text) => text.endsWith(` ${n}`))),
  chipTexts.join(" | "),
);
const rows = tasks.locator("ul > li button[aria-expanded]");
check("tasks: All shows all 63 rows", (await rows.count()) === summary.meta.n_tasks, String(await rows.count()));
for (const chip of ["Classification", "Extraction", "Reasoning and math", "Code", "Open-ended"]) {
  const button = chips.filter({ hasText: chip }).first();
  await button.click();
  const expected = categoryCounts[{ "Classification": "classification", "Extraction": "extraction", "Reasoning and math": "reasoning", "Code": "code", "Open-ended": "open_ended" }[chip]];
  const count = await rows.count();
  check(`tasks: filter "${chip}" shows ${expected} tasks and is pressed`, count === expected && (await button.getAttribute("aria-pressed")) === "true", `${count} rows`);
}
await chips.filter({ hasText: "All" }).first().click();
check("tasks: All shows 63 again", (await rows.count()) === summary.meta.n_tasks);

// A code task: hidden unit tests, the answer for each setting, and the grade.
const codeTitle = summary.tasks.find((task) => task.id === "code-compare-versions").title;
const codeButton = tasks.locator("ul > li button[aria-expanded]", { hasText: codeTitle }).first();
await codeButton.click();
const codePanel = tasks.locator("#task-code-compare-versions");
check("tasks: a row opens and reports aria-expanded", (await codeButton.getAttribute("aria-expanded")) === "true");
await codePanel.getByText("Grader:").first().waitFor({ timeout: 15_000 });
const codeText = norm(await codePanel.innerText()).toLowerCase();
check("tasks: code task shows its prompt, grader name and hidden tests", codeText.includes("grader: hidden unit tests") && codeText.includes("prompt") && codeText.includes("compare_versions("), codeText.slice(0, 160));
check("tasks: the reference answer is hidden until opened", (await codePanel.locator("details").first().getAttribute("open")) === null);
await codePanel.getByText("Reference answer and format contract").click();
check("tasks: the reference answer opens", (await codePanel.locator("details").first().getAttribute("open")) !== null);
const pickerButtons = codePanel.locator('[role=group][aria-label="Choose a model and effort setting"] button');
check("tasks: model and effort picker has nine settings", (await pickerButtons.count()) === 9, String(await pickerButtons.count()));
const routerPick = summary.tasks.find((task) => task.id === "code-compare-versions").router_pick;
check(
  "tasks: the picker starts on Cascade's pick for this task",
  norm(await codePanel.innerText()).includes(`Cascade chose ${armName(routerPick)}`),
);
await pickerButtons.filter({ hasText: "Opus, med" }).first().click();
const opusMedBlock = norm(await codePanel.innerText()).toLowerCase();
check(
  "tasks: choosing Opus, medium lists its three tries with a grade and a simulated tag",
  opusMedBlock.includes("opus, medium effort: 3 tries") && /pass|fail/.test(opusMedBlock),
  opusMedBlock.slice(0, 200),
);
const tryCards = codePanel.locator("li.border");
check("tasks: each try shows tokens and cost", (await tryCards.count()) === 3 && (await tryCards.first().innerText()).includes("in ·"));
check("tasks: each try shows the answer it gave", (await codePanel.locator("li pre").count()) >= 3);
await codeButton.click();
check("tasks: clicking the row again closes it", (await codeButton.getAttribute("aria-expanded")) === "false" && (await codePanel.count()) === 0);

// A judged task: the Opus judge is described, and the synthetic note says it was not called.
const judgeButton = tasks.locator("ul > li button[aria-expanded]", { hasText: "open-ab-paradox" }).first();
await judgeButton.click();
const judgePanel = tasks.locator("#task-open-ab-paradox");
await judgePanel.getByText("Grader:").first().waitFor({ timeout: 15_000 });
const judgeText = norm(await judgePanel.innerText()).toLowerCase();
check("tasks: judged task names the Opus judge and its rubric", judgeText.includes("grader: opus judge, rubric") && /x\d/.test(judgeText), judgeText.slice(0, 160));
check("tasks: synthetic try says no judge was called", judgeText.includes("no judge was called."), judgeText.slice(0, 200));
await judgeButton.click();

// An exact-match task names the expected label.
await tasks.locator("ul > li button[aria-expanded]", { hasText: "cls-ticket-billing" }).first().click();
const exactPanel = tasks.locator("#task-cls-ticket-billing");
await exactPanel.getByText("Grader:").first().waitFor({ timeout: 15_000 });
check("tasks: exact-match task names its grader and expected label", norm(await exactPanel.innerText()).toLowerCase().includes("grader: exact match") && norm(await exactPanel.innerText()).toLowerCase().includes("expected billing"));
await tasks.locator("ul > li button[aria-expanded]", { hasText: "cls-ticket-billing" }).first().click();

// A JSON task names the JSON grader and shows the expected object.
await tasks.locator("ul > li button[aria-expanded]", { hasText: "ext-invoice-header" }).first().click();
const jsonPanel = tasks.locator("#task-ext-invoice-header");
await jsonPanel.getByText("Grader:").first().waitFor({ timeout: 15_000 });
check("tasks: JSON task names its grader and the expected fields", norm(await jsonPanel.innerText()).toLowerCase().includes("grader: json schema + fields") && norm(await jsonPanel.innerText()).includes("INV-20417"));
await tasks.locator("ul > li button[aria-expanded]", { hasText: "ext-invoice-header" }).first().click();

// A failing detail fetch shows a message and a retry that works.
expectFailures = true;
await tasks.route("**/run/tasks/cls-sarcasm.json", (route) => route.fulfill({ status: 500, body: "nope" }));
await tasks.locator("ul > li button[aria-expanded]", { hasText: "cls-sarcasm" }).first().click();
const errorPanel = tasks.locator("#task-cls-sarcasm");
await errorPanel.getByText("Could not load this task.").waitFor({ timeout: 15_000 });
check("tasks: a failed task fetch says so", (await errorPanel.innerText()).includes("Could not load this task."));
await tasks.unroute("**/run/tasks/cls-sarcasm.json");
await errorPanel.getByRole("button", { name: "Try again" }).click();
await tasks.locator("#task-cls-sarcasm").getByText("Grader:").waitFor({ timeout: 15_000 });
check("tasks: Try again loads the task", (await tasks.locator("#task-cls-sarcasm").innerText()).toLowerCase().includes("grader:"));
expectFailures = false;

// Every detail file loads and has all 27 tries (9 settings x 3 trials).
const detailCheck = await tasks.evaluate(async (ids) => {
  const results = await Promise.all(
    ids.map((id) => fetch(`/run/tasks/${id}.json`).then((res) => (res.ok ? res.json() : null))),
  );
  return {
    loaded: results.filter(Boolean).length,
    complete: results.filter((detail) => detail && detail.cells?.length === 27 && detail.prompt).length,
  };
}, TASK_IDS);
check(`tasks: all ${TASK_IDS.length} task detail files load with 27 tries each`, detailCheck.loaded === TASK_IDS.length && detailCheck.complete === TASK_IDS.length, JSON.stringify(detailCheck));
await drainProblems("task explorer");
await tasks.close();

// ---- 4. Playground: presets, routing outcomes, stop, replacement, free-form ----------

heading("Playground");
const pg = await newPage({ width: 1280, height: 800 });
await pacePost();
await open(pg, "/playground");
check("playground: heading", (await pg.locator("h1").innerText()).includes("Watch Cascade pick a model for a prompt"));
check(
  "playground: demo line says the answers are pre-written and the router was trained on simulated results",
  norm(await pg.getByRole("status").innerText()).includes(
    "Demo: the answers here are pre-written examples, and the router was trained on simulated test results. Add your Anthropic API key to run any prompt live.",
  ),
);
check("playground: starts idle with a prompt to pick a sample", (await pg.getByText("Pick a sample prompt above", { exact: false }).count()) === 1);
check("playground: decision panel waits for a prompt", (await pg.getByText("Appears here as soon as there is a prompt.", { exact: false }).count()) === 1);
check("playground: prompt box limits input to 4,000 characters", (await pg.locator("#prompt").getAttribute("maxlength")) === "4000");

const outcomes = [];
for (const preset of presetList.filter((item) => item.route === "/api/route-and-run")) {
  const fixture = fixtures[preset.body.demoId];
  await pacePost();
  await pg.getByRole("button", { name: preset.label, exact: true }).click();
  await waitForAnswer(pg);
  await sleep(250);
  const run = pg.getByRole("button", { name: /^Run on / });
  const runLabel = norm(await run.innerText());
  const response = norm(await pg.locator("[data-testid=response]").innerText());
  const decision = norm(await pg.locator("[data-testid=decision]").innerText());
  outcomes.push(fixture.data.arm.key);
  check(
    `preset "${preset.label}": routed to ${armName(fixture.data.arm.key)} (the decision matches the fixture)`,
    runLabel === `Run on ${armName(fixture.data.arm.key)}`,
    runLabel,
  );
  check(
    `preset "${preset.label}": the decision panel explains the choice and the bar`,
    decision.toLowerCase().includes("cascade's choice") && decision.includes("the bar is 90%") && decision.toLowerCase().includes("every setting, cheapest first"),
  );
  check(
    `preset "${preset.label}": the answer is labelled a pre-written sample and shows its text`,
    response.includes("Pre-written sample answer, not a model run.") && response.includes(plain(fixture.data.answer).slice(0, 40)),
    response.slice(0, 160),
  );
  check(`preset "${preset.label}": the token meter says it is a sample`, response.includes("Sample (not a real run):"));
  const nineRows = await pg.locator('[data-testid=decision] ul').first().locator("li").count();
  check(`preset "${preset.label}": all nine settings are listed with their chance`, nineRows === 9, String(nineRows));
}
const outcomeSet = new Set(outcomes);
check(
  "playground: the presets cover Haiku, Sonnet and Opus routings",
  [...outcomeSet].some((key) => key.startsWith("haiku")) &&
    [...outcomeSet].some((key) => key.startsWith("sonnet")) &&
    [...outcomeSet].some((key) => key.startsWith("opus")),
  [...outcomeSet].join(","),
);
const diceFixture = fixtures["dice-probability"];
await pacePost();
await pg.getByRole("button", { name: "Solve a probability problem", exact: true }).click();
await waitForAnswer(pg);
check(
  "playground: the fallback routing says nothing cleared the bar and it uses the strongest",
  (await pg.locator("[data-testid=decision]").innerText()).includes("nothing clears the bar, so it uses the strongest"),
);
check("playground: the fallback routing answers with the strongest setting", diceFixture.data.arm.key === "opus-high");

// Stop: cancels the replay, keeps what arrived, and sends nothing more.
const before = postTimes.length;
await pacePost();
await pg.getByRole("button", { name: "Explain a concept with one analogy", exact: true }).click();
await pg.getByRole("button", { name: "Stop", exact: true }).waitFor({ timeout: 10_000 });
await sleep(450);
await pg.getByRole("button", { name: "Stop", exact: true }).click();
await sleep(1500);
const stopped = norm(await pg.locator('[aria-live=polite].min-h-40').first().innerText());
check("stop: the answer stops and says it was cancelled", stopped.startsWith("Stopped. The replay was cancelled; nothing was being billed."), stopped.slice(0, 120));
check("stop: the Run button returns", (await pg.getByRole("button", { name: /^Run on / }).count()) === 1);
const stoppedAgain = norm(await pg.locator('[aria-live=polite].min-h-40').first().innerText());
await sleep(1200);
check("stop: nothing more arrives after stopping", norm(await pg.locator('[aria-live=polite].min-h-40').first().innerText()) === stoppedAgain);
check("stop: one request was sent and no retry followed", postTimes.length - before === 1, `${postTimes.length - before} sent`);

// A new choice replaces a running one.
await pacePost();
await pacePost();
await pg.getByRole("button", { name: "Explain a concept with one analogy", exact: true }).click();
await sleep(300);
await pg.getByRole("button", { name: "Classify a support ticket", exact: true }).click();
await waitForAnswer(pg);
const replaced = norm(await pg.locator("[data-testid=response]").innerText());
check("a newer choice replaces the one running", replaced.includes("shipping") && !replaced.includes("race condition"), replaced.slice(0, 120));

// Free-form text: the choice works with no key, but nothing is sent.
const sentBefore = postTimes.length;
await pg.fill("#prompt", "Summarize the plot of Hamlet in two sentences.");
const freeDecision = norm(await pg.locator("[data-testid=decision]").innerText()).toLowerCase();
check("free-form: the decision updates as you type, with no key", freeDecision.includes("cascade's choice") && freeDecision.includes("chance of getting it right"));
check("free-form: Run is held back without a key", await pg.getByRole("button", { name: /^Run on / }).isDisabled());
check(
  "free-form: the help line offers the key or a sample",
  (await pg.locator("#prompt-help").innerText()).includes("add your Anthropic key") && (await pg.locator("#prompt-help").innerText()).includes("pick a sample prompt"),
);
await pg.press("#prompt", "Enter");
await sleep(400);
check("free-form: pressing Enter sends nothing", postTimes.length === sentBefore);

// The choice changes with the prompt: a short greeting and a long, technical request.
await pg.fill("#prompt", "Hi");
const greetingLabel = norm(await pg.locator("[data-testid=decision] span.text-xl").innerText());
await pg.fill("#prompt", "Prove that the sum of two odd integers is even, then derive the closed form of the recurrence a(n) = 3a(n-1) - 2a(n-2) with a(0) = 1 and a(1) = 4, showing every step, and check your answer against the first five terms.");
const longLabel = norm(await pg.locator("[data-testid=decision] span.text-xl").innerText());
check("free-form: the choice responds to the prompt (short greeting and long proof differ)", greetingLabel !== longLabel, `${greetingLabel} vs ${longLabel}`);
await pg.fill("#prompt", "");
check("free-form: an empty prompt returns the decision panel to its waiting state", (await pg.getByText("Appears here as soon as there is a prompt.", { exact: false }).count()) === 1);

// Typing a preset's exact prompt makes it runnable as that preset.
const explainPrompt = presetList.find((item) => item.body.demoId === "explain-race").body.prompt;
await pg.fill("#prompt", explainPrompt);
check("free-form: a preset's exact text is runnable as that preset", await pg.getByRole("button", { name: /^Run on / }).isEnabled());
await pacePost();
await pg.getByRole("button", { name: /^Run on / }).click();
await waitForAnswer(pg);
check("the Run button plays the preset it matches", (await pg.locator("[data-testid=response]").innerText()).toLowerCase().includes("race condition"));

// The add-key dialog: validation, cancel, escape, and a well-formed key that is removed at once.
await pg.fill("#prompt", "Summarize the plot of Hamlet in two sentences.");
await pg.locator("#prompt-help button", { hasText: "add your Anthropic key" }).click();
check("key dialog: opens from the help line", await pg.locator("dialog[open]").count() === 1);
check("key dialog: heading names the key", (await pg.locator("dialog[open] h2").innerText()).includes("Add your Anthropic key"));
await pg.locator('dialog[open] input[name=key]').fill("not-a-key");
await pg.getByRole("button", { name: "Use this key", exact: true }).click();
check("key dialog: a malformed key is refused with a plain message", (await pg.locator("dialog[open] [role=alert]").innerText()).includes("Anthropic keys start with sk-ant-"));
await pg.getByRole("button", { name: "Cancel", exact: true }).click();
check("key dialog: Cancel closes it", (await pg.locator("dialog[open]").count()) === 0);
await pg.getByRole("button", { name: "Add key", exact: true }).click();
check("key dialog: the banner's Add key button opens it", (await pg.locator("dialog[open]").count()) === 1);
await pg.keyboard.press("Escape");
await sleep(150);
check("key dialog: Escape closes it", (await pg.locator("dialog[open]").count()) === 0);
await pg.getByRole("button", { name: "Add key", exact: true }).click();
await pg.mouse.click(4, 4);
await sleep(150);
check("key dialog: a click on the backdrop closes it", (await pg.locator("dialog[open]").count()) === 0);

// A well-formed placeholder key switches the banner to live and is removed again. No run is
// started, so nothing is sent to the model; the guard below would have blocked it anyway.
await pg.getByRole("button", { name: "Add key", exact: true }).click();
check(
  "key dialog: 'Create a key' opens the console's key page in a new tab",
  (await pg.locator("dialog[open] a", { hasText: "Create a key" }).getAttribute("href")) === "https://platform.claude.com/settings/keys" &&
    (await pg.locator("dialog[open] a", { hasText: "Create a key" }).getAttribute("target")) === "_blank",
);
await pg.locator("dialog[open] input[name=key]").fill("sk-ant-uitest-placeholder-0000000000");
await pg.locator("dialog[open] input[name=remember]").check();
await pg.getByRole("button", { name: "Use this key", exact: true }).click();
await sleep(200);
check("key dialog: a well-formed key switches the banner to live", (await pg.getByText("Live: the model Cascade picks answers with your key.").count()) === 1);
check("key dialog: 'Remember for this tab' stores the key for this tab only", (await pg.evaluate(() => sessionStorage.getItem("anthropic-key") !== null)));
await pg.reload({ waitUntil: "networkidle" });
check("key dialog: a reload keeps the remembered key (live mode)", (await pg.getByText("Live: the model Cascade picks answers with your key.").count()) === 1);
await pg.getByRole("button", { name: "Change key", exact: true }).click();
await pg.getByRole("button", { name: "Remove key", exact: true }).click();
await sleep(200);
check("key dialog: removing the key returns to demo mode", (await pg.getByRole("status").innerText()).includes("Demo: the answers here are pre-written examples"));
check("key dialog: removing the key clears it from this tab's storage", (await pg.evaluate(() => sessionStorage.getItem("anthropic-key") === null)));

// A stray ?try value plays nothing.
await pg.goto(`${BASE}/playground?try=no-such-preset`, { waitUntil: "networkidle" });
check("playground: an unknown ?try value plays nothing", (await pg.getByText("Pick a sample prompt above", { exact: false }).count()) === 1);
await drainProblems("playground desktop");
await pg.close();

// Dark playground: presets and the decision still render and answer.
const darkPg = await newPage({ width: 1280, height: 800, scheme: "dark" });
await pacePost();
await open(darkPg, "/playground");
await darkPg.getByRole("button", { name: "Classify a support ticket", exact: true }).click();
await waitForAnswer(darkPg);
check("dark playground: a preset answers", (await darkPg.locator("[data-testid=response]").innerText()).includes("shipping"));
await drainProblems("dark playground");
await darkPg.close();

// ---- 5. Phone widths: nothing scrolls sideways after each action --------------------

for (const width of [390, 360]) {
  heading(`Phone ${width}px`);
  const phone = await newPage({ width, height: 844 });
  await open(phone, "/");
  await noSideScroll(phone, `${width}px dashboard on load`);
  check(`${width}px: the headline and the result are on screen`, (await headlineText(phone)).includes("lower cost"));
  await phone.click('[aria-label^="Raise the threshold"]');
  await phone.waitForTimeout(250);
  await noSideScroll(phone, `${width}px dashboard after +`);
  check(`${width}px: + still moves the headline`, (await thresholdOutput(phone)) === pct(SWEEP[sweepIndexDefault + 1].threshold, 0));
  await phone.click('[aria-label^="Lower the threshold"]');
  await phone.click('[aria-label^="Lower the threshold"]');
  await phone.waitForTimeout(250);
  await noSideScroll(phone, `${width}px dashboard after −`);
  check(`${width}px: the nav links are reachable`, (await phone.getByRole("navigation", { name: "Main" }).locator("a").count()) === 3);
  await phone.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Tasks", exact: true }).click();
  await phone.waitForURL(`${BASE}/tasks`);
  await noSideScroll(phone, `${width}px tasks on load`);
  await phone.locator("ul > li button[aria-expanded]", { hasText: "cls-ticket-billing" }).first().click();
  await phone.locator("#task-cls-ticket-billing").waitFor({ timeout: 15_000 });
  await phone.locator("#task-cls-ticket-billing").getByRole("button", { name: /^Opus, low/ }).first().click();
  await noSideScroll(phone, `${width}px tasks after opening a task and choosing a setting`);
  await phone.goto(`${BASE}/playground`, { waitUntil: "networkidle" });
  await noSideScroll(phone, `${width}px playground on load`);
  await pacePost();
  await phone.getByRole("button", { name: "Write a short customer email", exact: true }).click();
  await waitForAnswer(phone);
  await noSideScroll(phone, `${width}px playground after a preset`);
  await phone.fill("#prompt", `${"A very long unbroken token ".repeat(12)}${"x".repeat(240)}`);
  await noSideScroll(phone, `${width}px playground with a long unbroken prompt`);
  await drainProblems(`phone ${width}px`);
  await phone.close();
}

// ---- 6. Not found -----------------------------------------------------------------------

heading("Not found");
const missing = await newPage({ width: 1280, height: 800 });
expectFailures = true;
const missingResponse = await open(missing, "/nope-not-a-page");
check("unknown address returns 404", missingResponse?.status() === 404, String(missingResponse?.status()));
check("unknown address explains itself", (await missing.locator("main").innerText()).includes("Nothing is routed here"));
const missingHrefs = await missing.locator("main a").evaluateAll((els) => els.map((el) => el.getAttribute("href")));
check("unknown address offers links to the results, tasks and playground", ["/", "/tasks", "/playground"].every((href) => missingHrefs.includes(href)), missingHrefs.join(" "));
await missing.getByRole("link", { name: "Browse the tasks" }).click();
await missing.waitForURL(`${BASE}/tasks`);
check("unknown address: links lead to a real page", missing.url() === `${BASE}/tasks`);
expectFailures = false;
await drainProblems("not found");
await missing.close();

// ---- 7. HTTP checks for the static files and the API route ------------------------------

heading("HTTP and API");
async function api(body, { headers = {}, raw = false, method = "POST" } = {}) {
  if (method === "POST") await pacePost();
  const init = { method, headers: { "content-type": "application/json", ...headers } };
  if (method === "POST") {
    init.body = raw ? body : JSON.stringify(body);
    postTimes.push(Date.now());
  }
  const res = await fetch(`${BASE}/api/route-and-run`, init);
  const type = res.headers.get("content-type") ?? "";
  const text = await res.text();
  return { status: res.status, type, text, json: type.includes("json") ? JSON.parse(text) : null };
}

const supportFixture = fixtures["support-ticket"];
const supportPrompt = presetList.find((item) => item.body.demoId === "support-ticket").body.prompt;
/** The route always answers with a server-sent event stream; the last frame is the result. */
const sse = (text) =>
  text
    .split("\n\n")
    .filter((chunk) => chunk.startsWith("data: "))
    .map((chunk) => JSON.parse(chunk.slice(6)));
const good = await api({ demoId: "support-ticket", prompt: supportPrompt });
const goodFrames = sse(good.text);
const goodDone = goodFrames.at(-1);
check(
  "API: a preset with no key replays its fixture (200, demo, placeholder)",
  good.status === 200 && goodDone?.type === "done" && goodDone.meta.mode === "demo" && goodDone.meta.placeholder === true,
  `${good.status} ${good.text.slice(-160)}`,
);
check(
  "API: the replayed answer and arm match the fixture",
  goodDone?.data.answer === supportFixture.data.answer && goodDone?.data.arm.key === supportFixture.data.arm.key,
);

const stream = await api({ demoId: "dice-probability", prompt: presetList.find((item) => item.body.demoId === "dice-probability").body.prompt });
const frames = sse(stream.text);
check("API: the stream is event-stream and starts with meta", stream.type.startsWith("text/event-stream") && frames[0]?.type === "meta", stream.type);
check("API: the stream sends a decision event first, then text, then done", frames[1]?.type === "event" && frames[1].event.type === "decision" && frames[1].event.arm.key === "opus-high" && frames.at(-1)?.type === "done");
const streamed = frames.filter((frame) => frame.type === "event" && frame.event.type === "text").map((frame) => frame.event.delta).join("");
check("API: the streamed text equals the stored answer", streamed === fixtures["dice-probability"].data.answer);
check("API: the done frame carries the final answer", frames.at(-1)?.data?.answer === fixtures["dice-probability"].data.answer);

const badJson = await api("{not json", { raw: true });
check("API: malformed JSON is a 400 with a plain message", badJson.status === 400 && badJson.json?.error?.message === "Request body must be JSON.", badJson.text.slice(0, 120));
const noPrompt = await api({ demoId: "support-ticket" });
check("API: a body without a prompt is a 400 invalid shape", noPrompt.status === 400 && noPrompt.json?.error?.message === "Request body has an invalid shape.");
const emptyPrompt = await api({ prompt: "" });
check("API: an empty prompt is a 400", emptyPrompt.status === 400 && emptyPrompt.json?.error?.kind === "invalid");
const longPrompt = await api({ prompt: "a".repeat(4001) });
check("API: a 4,001-character prompt is a 400", longPrompt.status === 400 && longPrompt.json?.error?.kind === "invalid");
const oversized = await api({ prompt: "b".repeat(70 * 1024) });
check("API: a body over 64 KB is a 413", oversized.status === 413 && oversized.json?.error?.kind === "too_large" && oversized.json.error.message === "Request body is too large.", `${oversized.status}`);
const freeNoKey = await api({ prompt: "Summarize the plot of Hamlet in two sentences." });
check("API: a free-form prompt with no key is a 404 demo miss", freeNoKey.status === 404 && freeNoKey.json?.error?.kind === "demo_miss");
const swapped = await api({ demoId: "support-ticket", prompt: "Write a limerick about tea." });
check("API: a preset id with another prompt is refused (400)", swapped.status === 400 && swapped.json?.error?.kind === "invalid");
const unknownId = await api({ demoId: "no-such-preset", prompt: "Hello" });
check("API: an unknown preset id is a 404 demo miss", unknownId.status === 404 && unknownId.json?.error?.kind === "demo_miss");
const malformedKey = await api({ prompt: "Hello" }, { headers: { "x-anthropic-key": "sk-ant-short" } });
check(
  "API: a malformed key is refused before any call to Claude (401, plain message)",
  malformedKey.status === 401 && malformedKey.json?.error?.kind === "auth" && malformedKey.json.error.message === "That does not look like an Anthropic API key.",
  malformedKey.text.slice(0, 160),
);
const getRoute = await fetch(`${BASE}/api/route-and-run`);
check("API: GET is not allowed on the route", getRoute.status === 405, String(getRoute.status));

const icon = await fetch(`${BASE}/icon.svg`);
check("static: the app icon is served as SVG", icon.status === 200 && (icon.headers.get("content-type") ?? "").includes("svg"));
const detail = await fetch(`${BASE}/run/tasks/cls-sarcasm.json`);
check("static: a task detail file is served as JSON", detail.status === 200 && (detail.headers.get("content-type") ?? "").includes("json"));
const absent = await fetch(`${BASE}/run/tasks/no-such-task.json`);
check("static: a missing task detail file is a 404", absent.status === 404, String(absent.status));

// Rate limit last: wait out the window, send 20 cheap requests, then the 21st is refused.
while (postTimes.length && Date.now() - postTimes.at(-1) < 61_000) await sleep(1000);
postTimes.length = 0;
let twentyOk = true;
for (let i = 0; i < 20; i++) {
  const res = await fetch(`${BASE}/api/route-and-run`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
  });
  postTimes.push(Date.now());
  if (res.status !== 400) twentyOk = false;
}
check("rate limit: the first 20 requests in a minute reach the handler", twentyOk);
const limited = await fetch(`${BASE}/api/route-and-run`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: "{}",
});
const limitedBody = await limited.json();
check(
  "rate limit: the 21st request is a 429 with Retry-After and a plain message",
  limited.status === 429 && Number(limited.headers.get("retry-after")) > 0 && limitedBody.error?.kind === "rate_limit",
  `${limited.status}`,
);

// ---- done -------------------------------------------------------------------------------

check("network guard: nothing left the app's origin (no call to Claude or any other host)", blockedExternal.length === 0, blockedExternal.slice(0, 3).join(" "));

await browser.close();

const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length} of ${results.length} checks ok.`);
if (failed.length) {
  console.log("Failed:");
  for (const result of failed) console.log(`  [${result.group}] ${result.name}${result.detail ? ` -- ${result.detail}` : ""}`);
  process.exit(1);
}
