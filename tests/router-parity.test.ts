import { describe, expect, it } from "vitest";
import vectorsJson from "@/data/router-vectors.json";
import { asciiLower, extractFeatures, normalize } from "@/lib/router/features";
import { ROUTER, chooseArm, explain, probability, route, scorePrompt, type RouterArm } from "@/lib/router/model";

interface Vector {
  prompt: string;
  features: number[];
  contributions: number[];
  score: number;
  arm: string;
}
const vectors = vectorsJson as Vector[];

const feature = (prompt: string, name: string) =>
  extractFeatures(ROUTER.features, prompt)[ROUTER.features.findIndex((f) => f.name === name)];

describe("router parity with the Python trainer", () => {
  it("has vectors for every eval task plus the adversarial strings", () => {
    expect(vectors.length).toBeGreaterThanOrEqual(70);
    expect(vectors.some((v) => v.prompt === "")).toBe(true);
    expect(vectors.some((v) => v.prompt.includes("\r\n"))).toBe(true);
    expect(vectors.some((v) => /\p{Extended_Pictographic}/u.test(v.prompt))).toBe(true);
  });

  it("reproduces every feature value, contribution, score, and decision exactly", () => {
    for (const vector of vectors) {
      const scored = scorePrompt(ROUTER, vector.prompt);
      // toEqual on numbers is exact equality: no tolerance is allowed here.
      expect(scored.features).toEqual(vector.features);
      expect(scored.contributions).toEqual(vector.contributions);
      expect(scored.score).toBe(vector.score);
      expect(route(ROUTER, vector.prompt).arm.key).toBe(vector.arm);
    }
  });

  it("covers several routing outcomes, so parity is not trivially true", () => {
    expect(new Set(vectors.map((v) => v.arm)).size).toBeGreaterThanOrEqual(3);
  });
});

describe("feature extraction rules", () => {
  it("normalises line endings and lower-cases only A-Z", () => {
    expect(normalize("a\r\nb\rc\nd")).toBe("a\nb\nc\nd");
    expect(asciiLower("ABC É K")).toBe("abc É K");
  });

  it("counts code points, not UTF-16 units", () => {
    expect(feature("\u{1f600}", "length")).toBe(1 / 401);
    expect(feature("ab1", "digit_density")).toBe(1 / 4);
  });

  it("matches ASCII only, like Python with re.ASCII", () => {
    expect(feature("Keys", "structure_cues")).toBe(0);
    expect(feature("keys", "structure_cues")).toBe(1 / 4);
  });

  it("recognises bulleted and numbered lines after leading whitespace", () => {
    const text = "intro\n- one\n  * two\n 3. three\n[4] four\nnot - a bullet";
    expect(feature(text, "list_items")).toBe(4 / 8);
  });

  it("is stateless across calls (cached global regexes do not leak lastIndex)", () => {
    const prompt = "Classify this as one of: a, b. Classify again.";
    expect(extractFeatures(ROUTER.features, prompt)).toEqual(extractFeatures(ROUTER.features, prompt));
  });
});

describe("routing rule", () => {
  const arm = (key: string, bias: number) => ({ key, bias }) as RouterArm;
  const arms = [arm("cheap", -1), arm("mid", 0.5), arm("best", 2)];
  const threshold = Math.log(0.8 / 0.2);

  it("takes the cheapest arm that clears the threshold", () => {
    expect(chooseArm(arms, 3, threshold)).toMatchObject({ arm: { key: "cheap" }, fallback: false });
    expect(chooseArm(arms, 1, threshold)).toMatchObject({ arm: { key: "mid" }, fallback: false });
    expect(chooseArm(arms, 0, threshold)).toMatchObject({ arm: { key: "best" }, fallback: false });
    expect(chooseArm([arm("zero", 0), arms[2]], threshold, threshold).arm.key).toBe("zero");
  });

  it("falls back to the most able arm when nothing clears it", () => {
    const shuffled = [arms[2], arms[0], arms[1]];
    expect(chooseArm(shuffled, -9, threshold)).toMatchObject({ arm: { key: "best" }, fallback: true });
  });

  it("orders the served arms cheapest first and precomputes the threshold logit", () => {
    const costs = ROUTER.arms.map((a) => a.cost_per_request);
    expect(costs).toEqual([...costs].sort((a, b) => a - b));
    expect(ROUTER.threshold_logit).toBeCloseTo(Math.log(ROUTER.threshold / (1 - ROUTER.threshold)), 5);
  });
});

describe("explanation", () => {
  it("names the chosen arm, the bar, and the strongest signals", () => {
    const decision = route(ROUTER, "Classify this support message as one of: billing, shipping.");
    const text = explain(ROUTER, decision).join(" ");
    expect(text).toContain(`${Math.round(ROUTER.threshold * 100)}%`);
    expect(text).toMatch(/Haiku|Sonnet|Opus/);
    expect(text).toContain("Strongest signals");
    const chosen = probability(decision.arm.bias + decision.score);
    expect(text).toContain(`${Math.round(chosen * 100)}%`);
  });

  it("says so when it has to fall back", () => {
    const hard = { ...ROUTER, threshold: 0.999, threshold_logit: 99 };
    const decision = route(hard, "anything");
    expect(decision.fallback).toBe(true);
    expect(explain(hard, decision).join(" ")).toContain("falls back to the most able one");
  });
});
