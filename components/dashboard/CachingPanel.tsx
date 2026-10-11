import { count, pct, usd } from "@/lib/format";
import type { CachingSummary } from "@/lib/run-schema";

/** Cumulative cost per call, with and without the cache breakpoint. */
function CumulativeChart({ caching }: { caching: CachingSummary }) {
  const [plain, cached] = caching.conditions;
  const running = (calls: { cost_usd: number }[]) => {
    let total = 0;
    return calls.map((call) => (total += call.cost_usd));
  };
  const a = running(plain.calls);
  const b = running(cached.calls);
  const peak = Math.max(...a, ...b);
  const n = a.length;
  const path = (series: number[]) =>
    series.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(1, n - 1)) * 100).toFixed(2)},${(40 - (v / peak) * 38).toFixed(2)}`).join("");
  return (
    <div className="font-mono text-[11px]">
      <div className="flex items-stretch gap-2">
        <div className="flex w-12 flex-col justify-between text-right text-[10px] text-muted">
          <span>{usd(peak)}</span>
          <span>$0</span>
        </div>
        <div className="min-w-0 flex-1">
          <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="h-36 w-full border-b border-l border-border" role="img" aria-label={`Cumulative cost over ${n} calls: ${usd(a[n - 1])} without caching, ${usd(b[n - 1])} with caching`}>
            {[10, 20, 30].map((y) => (
              <line key={y} x1="0" x2="100" y1={y} y2={y} stroke="var(--grid)" vectorEffect="non-scaling-stroke" />
            ))}
            <path d={path(a)} fill="none" stroke="var(--fg)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
            <path d={path(b)} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
          </svg>
          <div className="mt-1 flex justify-between text-[10px] text-muted">
            <span>call 1</span>
            <span>call {n}</span>
          </div>
        </div>
      </div>
      <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-muted">
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-px w-5 bg-fg" /> no breakpoint: {usd(a[n - 1])}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-0.5 w-5 bg-accent" /> cached prefix: {usd(b[n - 1])}
        </span>
      </p>
    </div>
  );
}

export function CachingPanel({ caching }: { caching: CachingSummary }) {
  const { rules } = caching;
  const th = "px-2 py-1.5 text-right font-normal";
  const td = "tabular px-2 py-1.5 text-right";
  return (
    <div className="grid gap-x-10 gap-y-6 lg:grid-cols-2">
      <div className="min-w-0">
        <CumulativeChart caching={caching} />
        <div className="mt-4 overflow-x-auto border border-border">
          <table className="w-full border-collapse font-mono text-xs">
            <thead className="border-b border-border bg-surface text-muted">
              <tr>
                <th scope="col" className="px-2 py-1.5 text-left font-normal">
                  condition
                </th>
                <th scope="col" className={th}>
                  input
                </th>
                <th scope="col" className={th}>
                  cache write
                </th>
                <th scope="col" className={th}>
                  cache read
                </th>
                <th scope="col" className={th}>
                  cost
                </th>
              </tr>
            </thead>
            <tbody>
              {caching.conditions.map((condition) => (
                <tr key={condition.key} className="border-b border-grid last:border-b-0">
                  <th scope="row" className="px-2 py-1.5 text-left font-normal">
                    {condition.label}
                  </th>
                  <td className={td}>{count(condition.input_tokens)}</td>
                  <td className={td}>{count(condition.cache_creation_input_tokens)}</td>
                  <td className={td}>{count(condition.cache_read_input_tokens)}</td>
                  <td className={td}>{usd(condition.total_usd)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <div className="min-w-0 text-sm">
        <p>
          The same {count(caching.prefix_tokens)}-token handbook is sent as the system prompt on {caching.calls} calls
          in a row. With a cache breakpoint the first call pays{" "}
          <span className="font-mono">{rules.write_multiplier.toFixed(2)}x</span> the input price to write the cache,
          and every later call reads it at <span className="font-mono">{rules.read_multiplier.toFixed(2)}x</span>.
          {caching.break_even_call !== null && (
            <> Caching is ahead from call {caching.break_even_call} onward.</>
          )}
        </p>
        <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-2 font-mono text-xs">
          <div className="border-t border-grid pt-1.5">
            <dt className="text-muted">saved on this run</dt>
            <dd className="tabular mt-0.5 text-sm font-semibold">
              {usd(caching.savings_usd)} ({pct(caching.savings_pct)})
            </dd>
          </div>
          <div className="border-t border-grid pt-1.5">
            <dt className="text-muted">saved on input tokens alone</dt>
            <dd className="tabular mt-0.5 text-sm font-semibold">{pct(caching.input_savings_pct)}</dd>
          </div>
        </dl>
        <ul className="mt-4 space-y-2 text-muted">
          <li>
            <span className="font-mono text-fg">{count(rules.min_tokens)}-token minimum.</span> A shorter prefix is
            accepted and silently not cached. The third row is that case: a {count(caching.short_prefix_tokens)}-token
            prefix with a breakpoint, and no cache tokens at all.
          </li>
          <li>
            <span className="font-mono text-fg">{Math.round(rules.ttl_seconds / 60)}-minute lifetime.</span> A cached
            prefix expires five minutes after it was last written or read, and every read renews it. Traffic sparser
            than that pays the write premium again and again.
          </li>
          <li>
            <span className="font-mono text-fg">Sequential first call.</span> Requests sent in parallel can arrive
            before the first has written the cache, and then each pays full price. The experiment sends them one
            after another.
          </li>
        </ul>
      </div>
    </div>
  );
}
