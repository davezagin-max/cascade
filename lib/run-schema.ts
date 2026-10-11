/**
 * The shape of the JSON that `cascade export` writes. The dashboard reads those
 * files as plain data; this schema is what `tests/data.test.ts` validates them
 * against, so a change on the Python side that the UI does not expect fails CI.
 * Only types are imported from here by the app, so zod stays out of the bundle.
 */
import { z } from "zod";

const Mix = z.record(z.string(), z.number());

export const ArmSummary = z.object({
  key: z.string(),
  model: z.string(),
  model_short: z.string(),
  effort: z.string(),
  /** Share of tasks passed, averaged per task. */
  quality: z.number().min(0).max(1),
  /** 95% bootstrap interval for `quality`, resampling tasks. */
  ci: z.tuple([z.number(), z.number()]),
  mean_score: z.number(),
  cost_per_request: z.number(),
  cost_per_1k: z.number(),
  cost_per_1k_batch: z.number(),
  latency_p50_ms: z.number().nullable(),
  latency_p95_ms: z.number().nullable(),
  latency_n: z.number(),
  input_tokens_mean: z.number(),
  output_tokens_mean: z.number(),
  n_rows: z.number(),
  refusals: z.number(),
  truncated: z.number(),
  errors: z.number(),
  ungraded: z.number(),
  on_frontier: z.boolean(),
  by_category: z.record(
    z.string(),
    z.object({
      quality: z.number(),
      mean_score: z.number(),
      cost_per_1k: z.number(),
      n_tasks: z.number(),
    }),
  ),
});

const Outcome = z.object({
  quality: z.number(),
  cost_per_request: z.number(),
  cost_saved: z.number(),
  quality_retained: z.number(),
  mix: Mix,
});

export const RouterSummary = z.object({
  version: z.string(),
  threshold: z.number(),
  l2: z.number(),
  folds: z.number(),
  n_tasks: z.number(),
  baseline_arm: z.string(),
  baseline: z.object({ quality: z.number(), cost_per_request: z.number() }),
  headline: Outcome,
  oracle: Outcome,
  sweep: z.array(Outcome.extend({ threshold: z.number() })),
  confusion: z.object({
    predicted: z.array(z.string()),
    oracle: z.array(z.string()),
    matrix: z.array(z.array(z.number())),
    exact: z.number(),
    over_routed: z.number(),
    under_routed: z.number(),
  }),
  by_category: z.record(
    z.string(),
    z.object({
      n_tasks: z.number(),
      quality: z.number(),
      cost_per_request: z.number(),
      baseline_quality: z.number(),
      baseline_cost_per_request: z.number(),
      mix: Mix,
    }),
  ),
  calibration: z.object({ predicted: z.number(), realized: z.number() }),
  weights: z.array(z.object({ name: z.string(), label: z.string(), weight: z.number() })),
  arm_bias: z.array(z.object({ key: z.string(), bias: z.number() })),
});

const CacheCall = z.object({
  n: z.number(),
  input_tokens: z.number(),
  output_tokens: z.number(),
  cache_creation_input_tokens: z.number(),
  cache_read_input_tokens: z.number(),
  cost_usd: z.number(),
});

export const CachingSummary = z.object({
  synthetic: z.boolean(),
  model: z.string(),
  effort: z.string(),
  calls: z.number(),
  prefix_tokens: z.number(),
  short_prefix_tokens: z.number(),
  conditions: z.array(
    z.object({
      key: z.enum(["no_cache", "cached", "below_minimum"]),
      label: z.string(),
      calls: z.array(CacheCall),
      input_tokens: z.number(),
      cache_creation_input_tokens: z.number(),
      cache_read_input_tokens: z.number(),
      output_tokens: z.number(),
      total_usd: z.number(),
    }),
  ),
  savings_usd: z.number(),
  savings_pct: z.number(),
  input_savings_pct: z.number(),
  break_even_call: z.number().nullable(),
  rules: z.object({
    min_tokens: z.number(),
    ttl_seconds: z.number(),
    write_multiplier: z.number(),
    read_multiplier: z.number(),
  }),
});

export const TaskIndexEntry = z.object({
  id: z.string(),
  title: z.string(),
  category: z.string(),
  difficulty: z.number(),
  grader: z.enum(["exact", "json", "code", "judge"]),
  cells: z.record(z.string(), z.object({ pass_rate: z.number(), mean_score: z.number() })),
  router_pick: z.string().nullable(),
  oracle_arm: z.string().nullable(),
});

export const RunSummary = z.object({
  meta: z.object({
    run_id: z.string(),
    /** True when the run came from the simulator. Every "simulated" label keys off this. */
    synthetic: z.boolean(),
    created_at: z.string(),
    generator: z.object({ name: z.string(), seed: z.number(), note: z.string() }).nullable(),
    suite_hash: z.string().nullable(),
    models: z.array(z.string()),
    efforts: z.array(z.string()),
    trials: z.number(),
    n_tasks: z.number(),
    n_rows: z.number(),
    sync_rows: z.number(),
    batch_rows: z.number(),
    answers_billed_usd: z.number(),
    answers_list_usd: z.number(),
    judge_usd: z.number(),
    baseline_arm: z.string(),
    pricing: z.object({
      source: z.string(),
      last_verified: z.string(),
      batch_multiplier: z.number(),
      per_mtok: z.record(
        z.string(),
        z.object({
          input: z.number(),
          output: z.number(),
          cache_write_5m: z.number(),
          cache_read: z.number(),
        }),
      ),
    }),
  }),
  categories: z.array(z.object({ key: z.string(), label: z.string(), n_tasks: z.number() })),
  arms: z.array(ArmSummary),
  router: RouterSummary,
  caching: CachingSummary.nullable(),
  tasks: z.array(TaskIndexEntry),
});

export const TaskDetail = z.object({
  id: z.string(),
  title: z.string(),
  category: z.string(),
  difficulty: z.number(),
  prompt: z.string(),
  system: z.string(),
  grader: z.object({ type: z.enum(["exact", "json", "code", "judge"]) }).passthrough(),
  reference: z.string(),
  /** Distinct answers; cells point into this list. */
  answers: z.array(z.string()),
  cells: z.array(
    z.object({
      arm: z.string(),
      trial: z.number(),
      mode: z.enum(["batch", "sync"]),
      status: z.enum(["ok", "truncated", "refusal", "error"]),
      score: z.number().nullable(),
      passed: z.boolean().nullable(),
      answer: z.number(),
      details: z.record(z.string(), z.unknown()),
      input_tokens: z.number(),
      output_tokens: z.number(),
      cost_usd: z.number(),
      latency_ms: z.number().nullable(),
    }),
  ),
});

export type ArmSummary = z.infer<typeof ArmSummary>;
export type RouterSummary = z.infer<typeof RouterSummary>;
export type CachingSummary = z.infer<typeof CachingSummary>;
export type TaskIndexEntry = z.infer<typeof TaskIndexEntry>;
export type RunSummary = z.infer<typeof RunSummary>;
export type TaskDetail = z.infer<typeof TaskDetail>;
