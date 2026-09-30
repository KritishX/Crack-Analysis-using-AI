import { useRef } from "react";
import { Icon } from "./Icon";
import { Words } from "./Words";
import { gsap, useMotion } from "../lib/motion";
import "./Hero.css";

// Main fracture plus four branches, in a 1200×640 viewBox.
const MAIN =
  "M610 -10 L596 58 L622 104 L580 172 L604 222 L556 290 L574 338 L520 410 L538 452 L492 524 L506 570 L470 650";
const BRANCHES = [
  "M580 172 L520 196 L492 244 L440 262 L418 310",
  "M556 290 L620 312 L648 372 L710 398 L738 452",
  "M538 452 L590 480 L600 540 L648 572",
  "M622 104 L676 128 L704 90 L760 100",
];

export function Hero() {
  const root = useRef<HTMLElement>(null);

  useMotion(() => {
    const tl = gsap.timeline({ defaults: { ease: "expo.out" } });

    tl.from(".hero__eyebrow", { y: 10, opacity: 0, duration: 0.9 })
      .from(".hero__title .w > span", { yPercent: 115, duration: 1.3, stagger: 0.07 }, "-=0.7")
      .from(".hero__lede, .hero__actions", { y: 16, opacity: 0, duration: 1, stagger: 0.1 }, "-=0.9")
      .from(".hero__stage", { y: 40, opacity: 0, scale: 0.98, duration: 1.4 }, "-=1");

    // The fracture propagates: main crack first, then each branch.
    tl.from(".crack__main", { drawSVG: "0%", duration: 2.2, ease: "power2.inOut" }, "-=0.8")
      .from(".crack__branch", { drawSVG: "0%", duration: 1.1, stagger: 0.22, ease: "power2.out" }, "-=1.4")
      .from(".hero__frame", { opacity: 0, scale: 1.04, duration: 0.9 }, "-=0.4")
      .from(".hero__tag", { opacity: 0, y: 6, duration: 0.6, stagger: 0.08 }, "-=0.5");

    gsap.to(".hero__tag", {
      y: -3,
      duration: 2.6,
      ease: "sine.inOut",
      yoyo: true,
      repeat: -1,
      stagger: { each: 0.5, from: "random" },
    });

    // Slow parallax as the page scrolls away from the hero.
    gsap.to(".hero__crack", {
      scale: 1.08,
      yPercent: 4,
      ease: "none",
      scrollTrigger: { trigger: ".hero__stage", start: "top bottom", end: "bottom top", scrub: 0.6 },
    });
  }, root);

  return (
    <section className="hero" id="top" ref={root}>
      <div className="container hero__head">
        <p className="eyebrow hero__eyebrow">
          <Icon name="layers" size={16} />
          ResNet18 · trained on SDNET2018
        </p>

        <h1 className="display hero__title">
          <Words text="Cracks," /> <Words text="caught" /> <Words text="early." className="serif" />
        </h1>

        <p className="lede hero__lede">
          Drop in a photo of concrete — a wall, a deck, a stretch of pavement — and get a clear
          answer on whether it&rsquo;s cracked.
        </p>

        <div className="hero__actions">
          <a className="btn btn--primary" href="#analyze">
            <Icon name="image" size={18} />
            Analyze an image
          </a>
          <a className="link" href="#how">
            How it works
            <Icon name="arrowRight" size={16} />
          </a>
        </div>
      </div>

      <div className="container">
        <div className="hero__stage" role="img" aria-label="A crack propagating across a concrete surface">
          <div className="hero__surface" />
          <svg className="hero__crack" viewBox="0 0 1200 640" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
            <g className="crack__shadow">
              <path className="crack__main" d={MAIN} />
              {BRANCHES.map((d, i) => (
                <path key={i} className="crack__branch" d={d} />
              ))}
            </g>
            <g className="crack__line">
              <path className="crack__main" d={MAIN} />
              {BRANCHES.map((d, i) => (
                <path key={i} className="crack__branch" d={d} />
              ))}
            </g>
          </svg>

          <div className="hero__frame" aria-hidden="true">
            <i /><i /><i /><i />
            <span className="hero__label">Crack</span>
          </div>

          <span className="hero__tag hero__tag--a">224 × 224 input</span>
          <span className="hero__tag hero__tag--b">Binary classifier</span>
          <span className="hero__tag hero__tag--c">Weighted loss</span>
        </div>
      </div>
    </section>
  );
}
