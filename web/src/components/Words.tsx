import { Fragment } from "react";

/**
 * Splits a string into masked words so GSAP can slide each one up from
 * behind its own baseline. The text stays readable to assistive tech.
 */
export function Words({ text, className }: { text: string; className?: string }) {
  return (
    <>
      <span className="sr-only">{text} </span>
      {text.split(" ").map((word, i) => (
        <Fragment key={i}>
          <span className={className ? `w ${className}` : "w"} aria-hidden="true">
            <span>{word}</span>
          </span>{" "}
        </Fragment>
      ))}
    </>
  );
}
