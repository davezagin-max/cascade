import { ArmGlyph } from "@/components/ArmGlyph";
import { armLabel, count, duration, pct, usd } from "@/lib/format";
import type { ArmSummary } from "@/lib/run-schema";

/** One row per arm: the numbers behind the chart. */
export function ArmsTable({ arms, baselineKey }: { arms: ArmSummary[]; baselineKey: string }) {
  const th = "whitespace-nowrap px-1.5 py-1.5 text-right font-normal sm:px-2";
  const td = "tabular whitespace-nowrap px-1.5 py-1.5 text-right sm:px-2";
  return (
    <div className="overflow-x-auto border border-border">
      <table className="w-full border-collapse font-mono text-xs">
        <thead className="border-b border-border bg-surface text-muted">
          <tr>
            <th scope="col" className="px-1.5 py-1.5 text-left font-normal sm:px-2">
              arm
            </th>
            <th scope="col" className={th}>
              <span className="sm:hidden">pass</span>
              <span className="hidden sm:inline">pass rate</span>
            </th>
            <th scope="col" className={`${th} hidden sm:table-cell`}>
              95% interval
            </th>
            <th scope="col" className={`${th} hidden md:table-cell`}>
              mean grade
            </th>
            <th scope="col" className={th}>
              <span className="sm:hidden">$/1k</span>
              <span className="hidden sm:inline">$ / 1k req</span>
            </th>
            <th scope="col" className={`${th} hidden md:table-cell`}>
              $ / 1k batch
            </th>
            <th scope="col" className={th}>
              p50
            </th>
            <th scope="col" className={`${th} hidden sm:table-cell`}>
              p95
            </th>
            <th scope="col" className={`${th} hidden lg:table-cell`}>
              out tokens
            </th>
            <th scope="col" className={`${th} hidden sm:table-cell`}>
              frontier
            </th>
          </tr>
        </thead>
        <tbody>
          {arms.map((arm) => (
            <tr key={arm.key} className="border-b border-grid last:border-b-0">
              <th scope="row" className="whitespace-nowrap px-1.5 py-1.5 text-left font-normal sm:px-2">
                <ArmGlyph model={arm.model_short} effort={arm.effort} muted={!arm.on_frontier} />{" "}
                <span className={arm.on_frontier ? "" : "text-muted"}>{armLabel(arm.key)}</span>
                {arm.key === baselineKey && (
                  <span className="ml-1.5 hidden text-[10px] text-muted sm:inline">baseline</span>
                )}
              </th>
              <td className={td}>
                <span className="inline-flex items-center justify-end gap-2">
                  <span aria-hidden className="hidden h-1.5 w-14 bg-grid sm:inline-block">
                    <span className="block h-full bg-fg" style={{ width: `${arm.quality * 100}%` }} />
                  </span>
                  {pct(arm.quality)}
                </span>
              </td>
              <td className={`${td} hidden text-muted sm:table-cell`}>
                {pct(arm.ci[0], 0)} to {pct(arm.ci[1], 0)}
              </td>
              <td className={`${td} hidden text-muted md:table-cell`}>{arm.mean_score.toFixed(3)}</td>
              <td className={td}>{usd(arm.cost_per_1k)}</td>
              <td className={`${td} hidden text-muted md:table-cell`}>{usd(arm.cost_per_1k_batch)}</td>
              <td className={td}>{duration(arm.latency_p50_ms)}</td>
              <td className={`${td} hidden text-muted sm:table-cell`}>{duration(arm.latency_p95_ms)}</td>
              <td className={`${td} hidden text-muted lg:table-cell`}>{count(arm.output_tokens_mean)}</td>
              <td className={`${td} hidden sm:table-cell`}>{arm.on_frontier ? "yes" : <span className="text-muted">no</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
