import { useRef } from "react";
import { Icon } from "./Icon";
import { Words } from "./Words";
import { gsap, useMotion } from "../lib/motion";
import "./Model.css";

const SPECS: { value: number; unit?: string; label: string; format?: "sci" }[] = [
  { value: 224, unit: "px", label: "Square input, every image resized" },
  { value: 5, label: "Training epochs" },
  { value: 32, label: "Images per batch" },
  { value: 1e-4, label: "Adam learning rate", format: "sci" },
];

const SURFACES = [
  ["D", "Bridge decks"],
  ["P", "Pavements"],
  ["W", "Walls"],
] as const;

export function Model() {
  const root = useRef<HTMLElement>(null);

  useMotion(() => {
    const trigger = { trigger: ".model__balance", start: "top 80%" };

    gsap.from(".model__head .w > span", {
      yPercent: 115,
      duration: 1.1,
      stagger: 0.06,
      ease: "expo.out",
      scrollTrigger: { trigger: ".model__head", start: "top 82%" },
    });
    gsap.from(".model__head .eyebrow, .model__head .lede", {
      y: 14,
      opacity: 0,
      stagger: 0.12,
      scrollTrigger: { trigger: ".model__head", start: "top 82%" },
    });

    gsap.from(".seg__fill", { scaleX: 0, duration: 1.6, ease: "expo.out", stagger: 0.12, scrollTrigger: trigger });
    gsap.from(".seg__label", { opacity: 0, y: 8, duration: 0.8, delay: 0.5, stagger: 0.1, scrollTrigger: trigger });

    gsap.utils.toArray<HTMLElement>("[data-count]").forEach((el) => {
      const end = Number(el.dataset.count);
      const counter = { v: 0 };
      gsap.to(counter, {
        v: end,
        duration: 1.6,
        ease: "power3.out",
        onUpdate: () => {
          el.textContent = String(Math.round(counter.v));
        },
        scrollTrigger: { trigger: el, start: "top 90%", once: true },
      });
    });

    gsap.from(".spec, .surface", {
      y: 24,
      opacity: 0,
      duration: 0.9,
      stagger: 0.07,
      ease: "expo.out",
      scrollTrigger: { trigger: ".specs", start: "top 88%" },
    });
  }, root);

  return (
    <section className="model section" id="model" ref={root}>
      <div className="container">
        <header className="model__head">
          <p className="eyebrow">
            <Icon name="layers" size={16} />
            The model
          </p>
          <h2 className="title">
            <Words text="Built for lopsided" /> <Words text="data." className="serif" />
          </h2>
          <p className="lede">
            About 82% of SDNET2018 is uncracked concrete. A model that always answered &ldquo;no
            crack&rdquo; would look 82% accurate and be useless.
          </p>
        </header>

        <div className="model__balance">
          <div className="seg">
            <p className="seg__title">Class balance</p>
            <div className="seg__track">
              <div className="seg__fill seg__fill--clear" style={{ width: "82%" }} />
              <div className="seg__fill seg__fill--crack" style={{ width: "18%" }} />
            </div>
            <div className="seg__legend">
              <span className="seg__label"><i className="dot dot--clear" />Uncracked ≈ 82%</span>
              <span className="seg__label"><i className="dot dot--crack" />Cracked ≈ 18%</span>
            </div>
          </div>

          <div className="seg">
            <p className="seg__title">Stratified split</p>
            <div className="seg__track">
              <div className="seg__fill seg__fill--a" style={{ width: "70%" }} />
              <div className="seg__fill seg__fill--b" style={{ width: "15%" }} />
              <div className="seg__fill seg__fill--c" style={{ width: "15%" }} />
            </div>
            <div className="seg__legend">
              <span className="seg__label"><i className="dot dot--a" />Train 70%</span>
              <span className="seg__label"><i className="dot dot--b" />Validation 15%</span>
              <span className="seg__label"><i className="dot dot--c" />Test 15%</span>
            </div>
          </div>
        </div>

        <p className="model__note">
          The loss function weights each class by <code>total ÷ count</code>, so the rarer cracked
          class counts for about <strong>4.5 times</strong> as much as an uncracked one. The
          checkpoint is chosen on validation F1, not accuracy.
        </p>

        <div className="specs">
          {SPECS.map((s) => (
            <div className="spec" key={s.label}>
              <p className="spec__value">
                {s.format === "sci" ? (
                  <span>1×10<sup>−4</sup></span>
                ) : (
                  <span data-count={s.value}>{s.value}</span>
                )}
                {s.unit && <small>{s.unit}</small>}
              </p>
              <p className="spec__label">{s.label}</p>
            </div>
          ))}
        </div>

        <div className="surfaces">
          <p className="surfaces__title">Three surface types in the dataset</p>
          <ul>
            {SURFACES.map(([code, name]) => (
              <li className="surface" key={code}>
                <span className="surface__code">{code}</span>
                {name}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
