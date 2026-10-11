/**
 * The exported eval run, read at build time.
 *
 * Import this only from server components. The summary is about 100 KB; client
 * components get the slices they need as props, so it never enters the browser bundle.
 */
import summaryJson from "@/data/summary.json";
import type { RunSummary } from "./run-schema";

export const RUN = summaryJson as unknown as RunSummary;

/**
 * True while the committed run is the simulator's. Every "simulated" tag and the
 * banner are driven by this flag, so they disappear on their own when a real run
 * is exported.
 */
export const SYNTHETIC = RUN.meta.synthetic;
