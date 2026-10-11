import type { Metadata } from "next";
import { SimTag } from "@/components/SimTag";
import { TaskExplorer } from "@/components/tasks/TaskExplorer";
import { RUN, SYNTHETIC } from "@/lib/run";

export const metadata: Metadata = { title: "Tasks" };

export default function TasksPage() {
  const { tasks, arms, categories, meta } = RUN;
  return (
    <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pb-4 pt-6 sm:pt-8">
      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted">Task explorer</p>
      <h1 className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-2xl font-semibold tracking-tight">
        {meta.n_tasks} tasks, {arms.length} arms, {meta.trials} trial{meta.trials === 1 ? "" : "s"} each{" "}
        <SimTag show={SYNTHETIC} />
      </h1>
      <p className="mb-5 mt-2 max-w-3xl text-sm text-muted">
        Every task in the suite is hand-written, with a grader that needs no judgement call wherever one is possible.
        Open a task to see its prompt, how it is graded, and what each model and effort level answered.
        {SYNTHETIC &&
          " While the data is synthetic, the answers shown are each task's own stand-in answers, graded by the real graders (open-ended tasks get drawn rubric scores, because the judge cannot be called without a key)."}
      </p>
      <TaskExplorer
        tasks={tasks}
        arms={arms.map((arm) => arm.key)}
        categories={categories}
        synthetic={SYNTHETIC}
      />
    </main>
  );
}
