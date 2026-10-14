import Link from "next/link";
import { ThemeToggle } from "@/components/kit/ThemeToggle";
import { NavLinks } from "./NavLinks";
import { SimTag } from "./SimTag";

/**
 * The synthetic-data notice. Rendered near the top of each results page while the
 * committed run is the simulator's. It is calm on purpose (a note, not an alarm) but
 * never dismissible, and it carries the same "simulated" tag that marks every
 * simulated number, so the tag explains itself.
 */
export function SyntheticNotice({ className = "" }: { className?: string }) {
  return (
    <div
      role="note"
      aria-label="About this data"
      data-testid="synthetic-banner"
      className={`flex flex-col gap-x-3 gap-y-1 border border-border border-l-2 border-l-accent bg-surface px-3 py-2 text-[13px] leading-snug sm:flex-row sm:items-baseline ${className}`}
    >
      <SimTag show className="shrink-0 self-start sm:self-auto" />
      <p>
        <strong className="font-semibold">Synthetic placeholder data.</strong> These are not measured results. Run{" "}
        <code className="whitespace-nowrap font-mono text-[12px]">uv run cascade run</code> with an API key to replace
        them.
      </p>
    </div>
  );
}

export function SiteHeader() {
  return (
    <header className="border-b border-border">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-4 px-4 pt-2.5 sm:pb-2.5">
        <Link href="/" className="flex items-center gap-2 font-mono text-sm font-semibold tracking-[0.2em]">
          <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden>
            <path d="M1 3h7v5h5v5h4" fill="none" stroke="var(--accent)" strokeWidth="2" />
          </svg>
          CASCADE
        </Link>
        <ThemeToggle className="ml-auto rounded-sm! px-2! py-1! font-mono text-xs! sm:order-3" />
        {/* On a phone the links take their own row; from sm up they sit beside the wordmark. */}
        <div className="-mx-2 mt-1 w-full sm:order-2 sm:mx-0 sm:mt-0 sm:w-auto">
          <NavLinks />
        </div>
      </div>
    </header>
  );
}

export function SiteFooter({ pricingSource, verified }: { pricingSource: string; verified: string }) {
  return (
    <footer className="mt-16 border-t border-border">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-x-6 gap-y-1 px-4 py-4 text-xs text-muted">
        <p>Built with Claude Code and Claude Opus 5.5.</p>
        <p className="font-mono">
          Prices from{" "}
          <a href={pricingSource} target="_blank" rel="noreferrer" className="underline underline-offset-2">
            the published rate card
          </a>
          , verified {verified}.
        </p>
      </div>
    </footer>
  );
}

/** A dashboard section: a monospace label, a plain-English title, then content. */
export function Section({
  id,
  label,
  title,
  note,
  children,
}: {
  id: string;
  label: string;
  title: React.ReactNode;
  note?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="border-t border-border pt-5">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">{label}</p>
      <h2 id={`${id}-title`} className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-lg font-semibold">
        {title}
      </h2>
      {note && <p className="mt-1 max-w-3xl text-sm text-muted">{note}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}
