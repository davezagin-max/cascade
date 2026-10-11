/** "smooth" unless the visitor asked for reduced motion. For scripted scrolling only. */
export function scrollBehavior(): ScrollBehavior {
  if (typeof window === "undefined") return "auto";
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}
