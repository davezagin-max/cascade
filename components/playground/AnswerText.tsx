import type { ReactNode } from "react";

/**
 * Renders a model answer. Models write Markdown, so the three constructs that
 * appear in short answers are shown properly: fenced code, **bold**, and `code`.
 * Everything else is kept as written. It builds React elements only, so nothing
 * in an answer is ever interpreted as HTML.
 */
function inline(text: string, keyBase: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g);
  return parts.map((part, i) => {
    const key = `${keyBase}-${i}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={key}>{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={key} className="border border-border bg-surface px-1">
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

export function AnswerText({ text, streaming = false }: { text: string; streaming?: boolean }) {
  // An odd number of fences means the last code block is still being written.
  const segments = text.split(/```[a-zA-Z]*\n?/);
  return (
    <div className="space-y-2 p-3 font-mono text-[13px] leading-relaxed">
      {segments.map((segment, i) => {
        const last = i === segments.length - 1;
        const caret = streaming && last ? " caret" : "";
        if (i % 2 === 1) {
          return (
            <pre
              key={i}
              className={`overflow-x-auto whitespace-pre border border-border bg-surface p-2 text-xs${caret}`}
            >
              {segment.replace(/\n$/, "")}
            </pre>
          );
        }
        const trimmed = segment.replace(/^\n+|\n+$/g, "");
        if (!trimmed && !(streaming && last)) return null;
        return (
          <p key={i} className={`whitespace-pre-wrap break-words${caret}`}>
            {inline(trimmed, String(i))}
          </p>
        );
      })}
    </div>
  );
}
