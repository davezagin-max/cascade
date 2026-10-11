import { ArmsTable } from "@/components/dashboard/ArmsTable";
import { CachingPanel } from "@/components/dashboard/CachingPanel";
import { CategoryHeatmap } from "@/components/dashboard/CategoryHeatmap";
import { HeroLab } from "@/components/dashboard/HeroLab";
import { LatencyChart } from "@/components/dashboard/LatencyChart";
import { ConfusionMatrix, FeatureWeights, RouterFacts } from "@/components/dashboard/RouterPanel";
import { SimTag } from "@/components/SimTag";
import { Section } from "@/components/SiteChrome";
import { armLabel, count, pct, usd } from "@/lib/format";
import { RUN, SYNTHETIC } from "@/lib/run";

/**
 * The results dashboard. Static: everything on it is read from the exported run at
 * build time. The one interactive piece, the threshold control, only switches
 * between numbers that were already computed.
 */
export default function Home() {
  const { meta, arms, router, caching, categories } = RUN;
  const sim = <SimTag show={SYNTHETIC} />;
  // Rows with no grade: requests that failed at the API, or answers the judge could not score.
  const ungraded = arms.reduce((total, arm) => total + arm.ungraded, 0);
  const facts: [string, string][] = [
    ["tasks", count(meta.n_tasks)],
    ["arms", `${arms.length} (${meta.models.length} models x ${meta.efforts.length} efforts)`],
    ["trials per cell", count(meta.trials)],
    ["result rows", ungraded ? `${count(meta.n_rows)} (${count(ungraded)} ungraded)` : count(meta.n_rows)],
    ["via Batch API", `${count(meta.batch_rows)} rows at half price`],
    ["timed synchronously", `${count(meta.sync_rows)} rows`],
    ["answers billed", `${usd(meta.answers_billed_usd)} (list ${usd(meta.answers_list_usd)})`],
    ["judge billed", usd(meta.judge_usd)],
  ];

  return (
    <main id="main" className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-10 px-4 pb-4 pt-6 sm:pt-8">
      <HeroLab
        synthetic={SYNTHETIC}
        arms={arms.map((arm) => ({
          key: arm.key,
          model: arm.model_short,
          effort: arm.effort,
          cost: arm.cost_per_1k,
          quality: arm.quality,
          ci: arm.ci,
          frontier: arm.on_frontier,
          p50: arm.latency_p50_ms,
        }))}
        armOrder={router.confusion.predicted}
        sweep={router.sweep}
        defaultThreshold={router.threshold}
        baselineKey={router.baseline_arm}
        baseline={router.baseline}
        nTasks={router.n_tasks}
        suiteTasks={meta.n_tasks}
        folds={router.folds}
      />

      <Section
        id="run"
        label="The run"
        title={<>What was measured {sim}</>}
        note={
          SYNTHETIC ? (
            <>
              No real run exists yet. These rows were written by <code className="font-mono">cascade simulate</code>{" "}
              (seed {meta.generator?.seed}) in exactly the schema a real run uses. Pass or fail is drawn from assumed
              model abilities; the grades are real grader output on stand-in answers. Token counts, costs, and
              latencies are assumptions.
            </>
          ) : (
            <>
              Run <code className="font-mono">{meta.run_id}</code>, recorded {meta.created_at.slice(0, 10)}.
            </>
          )
        }
      >
        <dl className="grid grid-cols-2 gap-x-6 gap-y-3 font-mono text-xs sm:grid-cols-4">
          {facts.map(([name, value]) => (
            <div key={name} className="border-t border-grid pt-1.5">
              <dt className="text-muted">{name}</dt>
              <dd className="tabular mt-0.5 text-sm">{value}</dd>
            </div>
          ))}
        </dl>
      </Section>

      <Section
        id="arms"
        label="Every arm"
        title={<>Quality, cost, and latency by model and effort {sim}</>}
        note="Pass rate is the share of tasks fully correct, averaged over trials. The interval is a 95% bootstrap over tasks, not rows, because trials of one task are not independent. Cost is the mean per request at list price; an arm is on the frontier when no other arm is both cheaper and better."
      >
        <ArmsTable arms={arms} baselineKey={router.baseline_arm} />
      </Section>

      <Section
        id="categories"
        label="By task type"
        title={<>Where the tiers separate {sim}</>}
        note={`Pass rate per category. The last row is the router at a threshold of ${router.threshold.toFixed(2)}, cross-validated.`}
      >
        <CategoryHeatmap arms={arms} categories={categories} router={router} />
      </Section>

      <Section
        id="latency"
        label="Latency"
        title={<>Median and tail latency {sim}</>}
        note={`Batch results carry no timing, so each arm sends the same stratified sample of tasks synchronously and times it (${arms[0]?.latency_n ?? 0} requests per arm here). The bar runs from p50 to p95 on a log scale.`}
      >
        <LatencyChart arms={arms} />
      </Section>

      <Section
        id="router"
        label="The router"
        title={<>Predicted arm versus the best arm in hindsight {sim}</>}
        note={
          <>
            One logistic regression: an ability per arm plus a weight per prompt feature, shared across arms. It
            scores a prompt from cheap text features (no model call) and picks the cheapest arm predicted to clear
            the threshold. Trained on the graded answers to {count(router.n_tasks)} tasks, evaluated by{" "}
            {router.folds}-fold cross-validation grouped by task, against always using{" "}
            {armLabel(router.baseline_arm)}.{" "}
            {router.l2_selection
              ? `The penalty on the weights (L2 = ${router.l2}) is not hand-picked: it is chosen by cross-validation on held-out log loss, inside each training fold.`
              : `The penalty on the weights is fixed at L2 = ${router.l2}.`}
          </>
        }
      >
        <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
          <div className="min-w-0 space-y-5">
            <ConfusionMatrix router={router} />
            <RouterFacts router={router} />
          </div>
          <div className="min-w-0">
            <FeatureWeights router={router} />
            <p className="mt-3 text-xs text-muted">
              Weights apply to standardized features, so their sizes are comparable. A positive weight means more
              of that feature predicts an easier prompt. With about {meta.n_tasks} prompts these are rough, and the
              interval on every arm above says the same thing.
            </p>
          </div>
        </div>
      </Section>

      {caching && (
        <Section
          id="caching"
          label="Prompt caching"
          title={
            <>
              What a cached prefix saves <SimTag show={caching.synthetic} />
            </>
          }
          note={
            caching.synthetic
              ? "Token counts here are estimated from text length; the dollar figures follow from the published prices. A real run reads the counts from the API's usage report."
              : `Measured on ${caching.model} at ${caching.effort} effort.`
          }
        >
          <CachingPanel caching={caching} />
        </Section>
      )}

      <Section id="method" label="Method" title="How to read these numbers">
        <ul className="grid gap-x-10 gap-y-3 text-sm text-muted md:grid-cols-2">
          <li>
            <span className="font-medium text-fg">Graders.</span> Classification and math are exact match on a final
            answer. Extraction must be valid against a JSON Schema and then match field by field. Code runs against
            hidden unit tests in a subprocess. Open-ended answers are scored by Opus against a written rubric, without
            being told which model wrote them.
          </li>
          <li>
            <span className="font-medium text-fg">Cost basis.</span> Costs on this page are list price per request,
            which is what production traffic pays. The eval itself runs through the Batch API at{" "}
            {pct(1 - meta.pricing.batch_multiplier, 0)} off; that discount is shown separately.
          </li>
          <li>
            <span className="font-medium text-fg">Quality retained.</span> The router&apos;s pass rate divided by the
            baseline arm&apos;s pass rate on the same tasks. Cost saved is one minus the same ratio for cost.
          </li>
          <li>
            <span className="font-medium text-fg">The harder baseline.</span> Beating &quot;always{" "}
            {armLabel(router.baseline_arm)}&quot; on cost is easy: any cheaper arm does it. So each threshold is also
            compared with the strongest single arm that costs no more than the router does there. A router earns its
            place only where it is ahead of that arm.
          </li>
          <li>
            <span className="font-medium text-fg">Small sample.</span> {meta.n_tasks} tasks is enough to compare
            tiers and too few to trust a single percentage point. Read the intervals, not the decimals.
          </li>
        </ul>
        <p className={`mt-4 font-mono text-[11px] text-muted ${SYNTHETIC ? "" : "hidden"}`}>
          {meta.generator?.note}
        </p>
      </Section>
    </main>
  );
}
