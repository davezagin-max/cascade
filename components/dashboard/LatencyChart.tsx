import { ArmGlyph } from "@/components/ArmGlyph";
import { armCode, armLabel, duration } from "@/lib/format";
import type { ArmSummary } from "@/lib/run-schema";

const TICKS_MS = [500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000];

/** p50 to p95 latency per arm on a shared log scale. Built from HTML so it reflows on a phone. */
export function LatencyChart({ arms }: { arms: ArmSummary[] }) {
  const timed = arms.filter((arm) => arm.latency_p50_ms !== null && arm.latency_p95_ms !== null);
  if (timed.length === 0) {
    return <p className="text-sm text-muted">This run has no timed requests, so there is no latency to show.</p>;
  }
  const lo = Math.log10(Math.min(...timed.map((arm) => arm.latency_p50_ms ?? 1)) / 1.5);
  const hi = Math.log10(Math.max(...timed.map((arm) => arm.latency_p95_ms ?? 1)) * 1.15);
  const at = (ms: number) => ((Math.log10(ms) - lo) / (hi - lo)) * 100;
  const ticks = TICKS_MS.filter((t) => at(t) > 2 && at(t) < 98);

  return (
    <div className="font-mono text-xs">
      <div className="grid grid-cols-[4.25rem_minmax(0,1fr)_6.75rem] items-end gap-x-2 text-[10px] text-muted sm:grid-cols-[9rem_minmax(0,1fr)_9rem]">
        <span />
        <div className="relative h-4">
          {ticks.map((t) => (
            <span
              key={t}
              className={`absolute -translate-x-1/2 whitespace-nowrap ${/^1/.test(String(t)) ? "" : "hidden sm:inline"}`}
              style={{ left: `${at(t)}%` }}
            >
              {duration(t)}
            </span>
          ))}
        </div>
        <span className="text-right">p50 / p95</span>
      </div>
      <ul className="mt-1">
        {timed.map((arm) => {
          const p50 = arm.latency_p50_ms ?? 0;
          const p95 = arm.latency_p95_ms ?? 0;
          return (
            <li
              key={arm.key}
              className="grid grid-cols-[4.25rem_minmax(0,1fr)_6.75rem] items-center gap-x-2 border-t border-grid py-1 sm:grid-cols-[9rem_minmax(0,1fr)_9rem]"
            >
              <span className="whitespace-nowrap" title={armLabel(arm.key)}>
                <ArmGlyph model={arm.model_short} effort={arm.effort} />{" "}
                <span className="sm:hidden">{armCode(arm.key)}</span>
                <span className="hidden sm:inline">{armLabel(arm.key)}</span>
              </span>
              <div className="relative h-3" aria-hidden>
                {ticks.map((t) => (
                  <span key={t} className="absolute inset-y-0 w-px bg-grid" style={{ left: `${at(t)}%` }} />
                ))}
                <span
                  className="absolute top-1/2 h-1 -translate-y-1/2 bg-fg/35"
                  style={{ left: `${at(p50)}%`, width: `${Math.max(at(p95) - at(p50), 0.5)}%` }}
                />
                <span className="absolute top-0 h-3 w-0.5 bg-fg" style={{ left: `${at(p50)}%` }} />
                <span className="absolute top-0.5 h-2 w-px bg-fg" style={{ left: `${at(p95)}%` }} />
              </div>
              <span className="tabular whitespace-nowrap text-right">
                {duration(p50)} <span className="text-muted">/ {duration(p95)}</span>
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
