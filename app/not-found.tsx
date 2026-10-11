import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = { title: "Not found" };

export default function NotFound() {
  return (
    <main id="main" className="mx-auto flex w-full max-w-6xl flex-1 flex-col justify-center px-4 py-16">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">404</p>
      <h1 className="mt-1 text-2xl font-semibold tracking-tight">Nothing is routed here</h1>
      <p className="mt-2 max-w-xl text-sm text-muted">
        That address does not match a page. The results, the task explorer, and the playground are one click away.
      </p>
      <div className="mt-5 flex flex-wrap gap-2 text-sm">
        <Link href="/" className="bg-accent px-3 py-2 font-medium text-accent-fg hover:opacity-90">
          See the results
        </Link>
        <Link href="/tasks" className="border border-border px-3 py-2 hover:bg-surface">
          Browse the tasks
        </Link>
        <Link href="/playground" className="border border-border px-3 py-2 hover:bg-surface">
          Open the playground
        </Link>
      </div>
    </main>
  );
}
