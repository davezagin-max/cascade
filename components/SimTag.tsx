/**
 * Marks a number as simulated. Rendered only while the loaded run is synthetic,
 * so callers pass the run's flag and never hard-code the label.
 */
export function SimTag({ show, className = "" }: { show: boolean; className?: string }) {
  if (!show) return null;
  return (
    <span
      title="This number comes from the simulator, not from a measured run."
      className={`inline-block border border-dashed border-accent px-1 align-middle font-mono text-[10px] font-medium uppercase leading-4 tracking-wider text-accent ${className}`}
    >
      simulated
    </span>
  );
}
