import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import type { RefObject } from "react";

gsap.registerPlugin(useGSAP, ScrollTrigger, DrawSVGPlugin);

gsap.defaults({ ease: "power3.out", duration: 0.8 });

const NO_REDUCED_MOTION = "(prefers-reduced-motion: no-preference)";

/**
 * Runs `build` inside a scoped GSAP context, but only when the user hasn't
 * asked for reduced motion. Everything created inside is reverted on cleanup.
 */
export function useMotion(
  build: () => void,
  scope: RefObject<HTMLElement | null>,
  dependencies: unknown[] = [],
) {
  useGSAP(
    () => {
      const mm = gsap.matchMedia();
      mm.add(NO_REDUCED_MOTION, () => {
        build();
      });
      return () => mm.revert();
    },
    { scope, dependencies, revertOnUpdate: true },
  );
}

export { gsap, ScrollTrigger };
