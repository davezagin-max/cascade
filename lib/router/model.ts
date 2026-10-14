/**
 * The router at serving time: score a prompt, pick an arm, explain the choice.
 *
 * Everything here is deterministic arithmetic on `data/router.json`, which the
 * Python trainer exports. No model is called to route. The same module runs in
 * the playground route (the authoritative decision) and in the browser (the
 * instant preview while typing), so the two cannot disagree.
 */
import routerJson from "@/data/router.json";
import { MODELS, type Effort, type ModelId } from "@/lib/config";
import { extractFeatures, type FeatureSpec } from "./features";

export interface RouterArm {
  key: string;
  model: ModelId;
  effort: Effort;
  /** The arm's learned ability: its pass logit on an average prompt. */
  bias: number;
  /** Mean cost per request measured by the eval, at list price. */
  cost_per_request: number;
  pass_rate: number;
  input_tokens_mean: number;
  output_tokens_mean: number;
}

export interface RouterModel {
  version: string;
  trained_on: { run_id: string; synthetic: boolean; n_tasks: number; n_rows: number };
  threshold: number;
  /** logit(threshold), precomputed so that routing never needs `exp`. */
  threshold_logit: number;
  baseline_arm: string;
  features: FeatureSpec[];
  mean: number[];
  scale: number[];
  weights: number[];
  /** Cheapest first, by measured cost. */
  arms: RouterArm[];
}

export const ROUTER = routerJson as RouterModel;

export interface Contribution {
  name: string;
  label: string;
  /** Raw feature value in [0, 1). */
  value: number;
  /** Standard deviations above (+) or below (-) the eval suite's mean. */
  z: number;
  /** What this feature adds to the pass logit. Positive means "easier". */
  contribution: number;
}

export interface Decision {
  arm: RouterArm;
  /** True when no arm reached the threshold and the most able arm was used. */
  fallback: boolean;
  /** Sum of contributions: the prompt's predicted easiness, shared by every arm. */
  score: number;
  features: number[];
  contributions: Contribution[];
  /** Pass logit per arm, in the same order as `model.arms`. */
  logits: number[];
}

/**
 * Mirrors `score_prompt` in `evals/cascade/router.py` operation for operation:
 * the same subtraction, division, multiplication, and left-to-right sum.
 */
export function scorePrompt(model: RouterModel, prompt: string) {
  const features = extractFeatures(model.features, prompt);
  const contributions: number[] = [];
  const zs: number[] = [];
  let score = 0;
  for (let i = 0; i < features.length; i++) {
    const z = (features[i] - model.mean[i]) / model.scale[i];
    const contribution = model.weights[i] * z;
    zs.push(z);
    contributions.push(contribution);
    score += contribution;
  }
  return { features, zs, contributions, score };
}

/** The cheapest arm that clears the threshold, else the arm with the highest ability. */
export function chooseArm(arms: RouterArm[], score: number, thresholdLogit: number) {
  for (const arm of arms) {
    if (arm.bias + score >= thresholdLogit) return { arm, fallback: false };
  }
  let best = arms[0];
  for (const arm of arms.slice(1)) if (arm.bias > best.bias) best = arm;
  return { arm: best, fallback: true };
}

export function route(model: RouterModel, prompt: string): Decision {
  const { features, zs, contributions, score } = scorePrompt(model, prompt);
  const { arm, fallback } = chooseArm(model.arms, score, model.threshold_logit);
  return {
    arm,
    fallback,
    score,
    features,
    contributions: model.features.map((feature, i) => ({
      name: feature.name,
      label: feature.label,
      value: features[i],
      z: zs[i],
      contribution: contributions[i],
    })),
    logits: model.arms.map((candidate) => candidate.bias + score),
  };
}

/** For display only. Routing compares logits and never calls this. */
export function probability(logit: number): number {
  return 1 / (1 + Math.exp(-logit));
}

const MODEL_NAMES: Record<string, string> = {
  [MODELS.haiku]: "Haiku",
  [MODELS.sonnet]: "Sonnet",
  [MODELS.opus]: "Opus",
};

/** "Opus, high effort": the plain name used in sentences and headings. */
export function armLabel(arm: { model: string; effort: string }): string {
  return `${MODEL_NAMES[arm.model] ?? arm.model}, ${arm.effort} effort`;
}

export function armLabelFromKey(key: string): string {
  const [model, effort] = key.split("-");
  return `${model.charAt(0).toUpperCase()}${model.slice(1)} · ${effort}`;
}

export function findArm(model: RouterModel, key: string): RouterArm | undefined {
  return model.arms.find((arm) => arm.key === key);
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

/** The strongest signals, most influential first. */
export function topContributions(decision: Decision, count = 4): Contribution[] {
  return [...decision.contributions]
    .filter((item) => Math.abs(item.contribution) >= 0.05)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, count);
}

/** Plain-English account of a decision, built from the numbers above. */
export function explain(model: RouterModel, decision: Decision): string[] {
  const chosenLogit = decision.arm.bias + decision.score;
  const bar = percent(model.threshold);
  const difficulty =
    decision.score > 0.75
      ? "easier than a typical test task"
      : decision.score < -0.75
        ? "harder than a typical test task"
        : "about as hard as a typical test task";
  const lines = [`This prompt looks ${difficulty}.`];

  if (decision.fallback) {
    lines.push(
      `No setting is predicted to get it right ${bar} of the time, so it falls back to the strongest one, ` +
        `${armLabel(decision.arm)} (predicted ${percent(probability(chosenLogit))}).`,
    );
  } else {
    const index = model.arms.indexOf(decision.arm);
    const skipped = index > 0 ? model.arms[index - 1] : null;
    lines.push(
      `${armLabel(decision.arm)} is the cheapest setting predicted to get it right at least ${bar} of the ` +
        `time (predicted ${percent(probability(chosenLogit))}).` +
        (skipped
          ? ` The next cheaper one, ${armLabel(skipped)}, is predicted at ` +
            `${percent(probability(skipped.bias + decision.score))}.`
          : " Nothing cheaper exists."),
    );
  }

  const top = topContributions(decision, 3);
  if (top.length > 0) {
    const describe = (item: Contribution) =>
      `${item.label.toLowerCase()} ${item.z >= 0 ? "above" : "below"} average`;
    const easier = top.filter((item) => item.contribution > 0).map(describe);
    const harder = top.filter((item) => item.contribution < 0).map(describe);
    const parts: string[] = [];
    if (easier.length) parts.push(`pointing to a cheaper setting: ${easier.join(", ")}`);
    if (harder.length) parts.push(`pointing to a stronger setting: ${harder.join(", ")}`);
    lines.push(`Biggest clues ${parts.join("; ")}.`);
  }
  return lines;
}
