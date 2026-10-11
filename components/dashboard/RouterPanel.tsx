import { ArmGlyph } from "@/components/ArmGlyph";
import { armCode, armLabel, armParts, pct } from "@/lib/format";
import type { RouterSummary } from "@/lib/run-schema";

/** Rows are the hindsight-best arm, columns the arm the router chose. The diagonal is exact. */
export function ConfusionMatrix({ router }: { router: RouterSummary }) {
  const { predicted, oracle, matrix } = router.confusion;
  const peak = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse font-mono text-[11px]">
        <caption className="sr-only">
          Router choice against the cheapest arm that did as well as any arm, per task
        </caption>
        <thead>
          <tr>
            <td />
            <th scope="colgroup" colSpan={predicted.length} className="pb-1 text-center font-normal text-muted">
              router chose
            </th>
          </tr>
          <tr>
            <th scope="col" className="pr-2 text-right font-normal text-muted">
              best in hindsight
            </th>
            {predicted.map((key) => (
              <th key={key} scope="col" title={armLabel(key)} className="w-7 pb-1 text-center font-normal sm:w-8">
                <ArmGlyph {...armParts(key)} size={11} />
                <span className="sr-only">{armLabel(key)}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {oracle.map((key, row) => (
            <tr key={key}>
              <th scope="row" className="whitespace-nowrap pr-2 text-right font-normal">
                {key === "none" ? (
                  <span className="text-muted">no arm passed</span>
                ) : (
                  <>
                    <span className="text-muted">{armCode(key)}</span> <ArmGlyph {...armParts(key)} size={11} />
                  </>
                )}
              </th>
              {matrix[row].map((value, column) => {
                const diagonal = predicted[column] === key;
                return (
                  <td
                    key={predicted[column]}
                    className="tabular h-7 w-7 border border-grid text-center sm:h-8 sm:w-8"
                    style={{
                      outline: diagonal ? "1.5px solid var(--accent)" : undefined,
                      outlineOffset: "-2px",
                      background: value
                        ? `color-mix(in srgb, var(--fg) ${Math.round((value / peak) * 80) + 8}%, transparent)`
                        : undefined,
                      color: value / peak > 0.55 ? "var(--bg)" : value ? "var(--fg)" : "var(--border)",
                    }}
                  >
                    {value || "·"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Each feature's weight on the pass logit, largest effect first. */
export function FeatureWeights({ router }: { router: RouterSummary }) {
  const peak = Math.max(...router.weights.map((item) => Math.abs(item.weight)), 0.001);
  return (
    <ul className="font-mono text-[11px]">
      <li className="grid grid-cols-[minmax(0,11rem)_minmax(0,1fr)_3rem] gap-x-2 pb-1 text-[10px] text-muted">
        <span>feature</span>
        <span className="flex justify-between">
          <span>harder</span>
          <span>easier</span>
        </span>
        <span className="text-right">weight</span>
      </li>
      {router.weights.map((item) => {
        const share = (Math.abs(item.weight) / peak) * 50;
        return (
          <li
            key={item.name}
            className="grid grid-cols-[minmax(0,11rem)_minmax(0,1fr)_3rem] items-center gap-x-2 border-t border-grid py-[3px]"
          >
            <span className="truncate" title={item.label}>
              {item.label}
            </span>
            <span className="relative h-2.5" aria-hidden>
              <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <span
                className="absolute inset-y-0.5 bg-fg"
                style={item.weight >= 0 ? { left: "50%", width: `${share}%` } : { right: "50%", width: `${share}%` }}
              />
            </span>
            <span className="tabular text-right">{item.weight >= 0 ? "+" : ""}{item.weight.toFixed(2)}</span>
          </li>
        );
      })}
    </ul>
  );
}

export function RouterFacts({ router }: { router: RouterSummary }) {
  const { confusion, calibration, oracle, n_tasks: n } = router;
  const facts: [string, string, string][] = [
    ["exact", `${confusion.exact} of ${n}`, "the router picked the hindsight-best arm"],
    ["over-routed", `${confusion.over_routed} of ${n}`, "it picked a dearer arm than needed: wasted money"],
    ["under-routed", `${confusion.under_routed} of ${n}`, "it picked a cheaper arm than needed: quality at risk"],
    [
      "calibration",
      `${pct(calibration.predicted)} predicted, ${pct(calibration.realized)} realized`,
      "mean predicted pass probability of the chosen arms, against how often they passed",
    ],
    [
      "hindsight ceiling",
      `saves ${pct(oracle.cost_saved)} at ${pct(oracle.quality_retained)} quality`,
      "a perfect router that sees every trial's outcome; no real router can reach it",
    ],
  ];
  return (
    <dl className="grid gap-x-6 gap-y-2 font-mono text-xs sm:grid-cols-2">
      {facts.map(([name, value, meaning]) => (
        <div key={name} className="border-t border-grid pt-1.5">
          <dt className="text-muted">{name}</dt>
          <dd className="tabular mt-0.5 font-semibold">{value}</dd>
          <dd className="mt-0.5 font-sans text-[11px] font-normal text-muted">{meaning}</dd>
        </div>
      ))}
    </dl>
  );
}
