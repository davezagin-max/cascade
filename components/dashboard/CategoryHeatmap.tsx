import { ArmGlyph } from "@/components/ArmGlyph";
import { armLabel, pct } from "@/lib/format";
import type { ArmSummary, RouterSummary } from "@/lib/run-schema";

interface Props {
  arms: ArmSummary[];
  categories: { key: string; label: string; n_tasks: number }[];
  router: RouterSummary;
}

const SHORT: Record<string, string> = {
  classification: "Class",
  extraction: "Extr",
  reasoning: "Math",
  code: "Code",
  open_ended: "Open",
};

/** A cell shaded by pass rate; the text flips colour once the shade gets dark. */
function Cell({ value }: { value: number | undefined }) {
  if (value === undefined) return <td className="px-1.5 py-1.5 text-right text-muted sm:px-2">n/a</td>;
  const strong = value >= 0.6;
  return (
    <td
      className="tabular px-1.5 py-1.5 text-right sm:px-2"
      style={{
        background: `color-mix(in srgb, var(--fg) ${Math.round(value * 88)}%, transparent)`,
        color: strong ? "var(--bg)" : "var(--fg)",
      }}
    >
      {pct(value, 0)}
    </td>
  );
}

/** Pass rate for every arm in every task category, plus the router's row. */
export function CategoryHeatmap({ arms, categories, router }: Props) {
  return (
    <div className="overflow-x-auto border border-border">
      <table className="w-full border-collapse font-mono text-xs">
        <thead className="border-b border-border bg-surface text-muted">
          <tr>
            <th scope="col" className="px-2 py-1.5 text-left font-normal">
              arm
            </th>
            {categories.map((category) => (
              <th key={category.key} scope="col" className="px-1.5 py-1.5 text-right font-normal sm:px-2">
                <span className="hidden sm:inline">{category.label}</span>
                <span className="sm:hidden" title={category.label}>
                  {SHORT[category.key] ?? category.label.slice(0, 4)}
                </span>
                <span className="ml-1 hidden text-[10px] lg:inline">n={category.n_tasks}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {arms.map((arm) => (
            <tr key={arm.key}>
              <th scope="row" className="whitespace-nowrap px-2 py-1.5 text-left font-normal">
                <ArmGlyph model={arm.model_short} effort={arm.effort} />{" "}
                <span className="hidden sm:inline">{armLabel(arm.key)}</span>
                <span className="sm:hidden">{arm.effort}</span>
              </th>
              {categories.map((category) => (
                <Cell key={category.key} value={arm.by_category[category.key]?.quality} />
              ))}
            </tr>
          ))}
          <tr className="border-t-2 border-accent">
            <th scope="row" className="whitespace-nowrap px-2 py-1.5 text-left font-semibold text-accent">
              router
            </th>
            {categories.map((category) => (
              <Cell key={category.key} value={router.by_category[category.key]?.quality} />
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
