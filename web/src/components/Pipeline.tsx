import { useRef } from "react";
import { Icon, type IconName } from "./Icon";
import { Words } from "./Words";
import { gsap, useMotion } from "../lib/motion";
import "./Pipeline.css";

const STEPS: { icon: IconName; title: string; body: string; file: string }[] = [
  {
    icon: "archive",
    title: "Extract",
    body: "SDNET2018 is unpacked into a local folder with a progress bar. The images stay out of Git.",
    file: "1_zip_file_extraction.py",
  },
  {
    icon: "shield",
    title: "Clean",
    body: "Every JPEG is verified and its size checked. Each file is labeled from its folder — C means cracked, U means not — and written to one metadata CSV.",
    file: "3_data_cleaning.py",
  },
  {
    icon: "split",
    title: "Split",
    body: "A stratified 70 / 15 / 15 split keeps the crack-to-clean ratio identical across train, validation and test.",
    file: "4_data_optimization.py",
  },
  {
    icon: "layers",
    title: "Train",
    body: "ResNet18, pretrained on ImageNet, is fine-tuned with a class-weighted loss. The checkpoint with the best validation F1 is the one that ships.",
    file: "model_training.py",
  },
];

export function Pipeline() {
  const root = useRef<HTMLElement>(null);

  useMotion(() => {
    gsap.from(".pipeline__intro .w > span", {
      yPercent: 115,
      duration: 1.1,
      stagger: 0.06,
      ease: "expo.out",
      scrollTrigger: { trigger: ".pipeline__intro", start: "top 82%" },
    });
    gsap.from(".pipeline__intro .eyebrow, .pipeline__intro .lede", {
      y: 14,
      opacity: 0,
      stagger: 0.12,
      scrollTrigger: { trigger: ".pipeline__intro", start: "top 82%" },
    });

    // The rail fills as you read down the steps.
    gsap.fromTo(
      ".steps__progress",
      { scaleY: 0 },
      {
        scaleY: 1,
        ease: "none",
        scrollTrigger: { trigger: ".steps", start: "top 65%", end: "bottom 65%", scrub: 0.4 },
      },
    );

    gsap.utils.toArray<HTMLElement>(".step").forEach((step) => {
      gsap.from(step, {
        y: 36,
        opacity: 0,
        duration: 1,
        ease: "expo.out",
        scrollTrigger: { trigger: step, start: "top 85%" },
      });
    });
  }, root);

  return (
    <section className="pipeline section" id="how" ref={root}>
      <div className="container pipeline__grid">
        <div className="pipeline__intro">
          <p className="eyebrow">
            <Icon name="terminal" size={16} />
            How it works
          </p>
          <h2 className="title">
            <Words text="From a zip file to a" /> <Words text="trained model." className="serif" />
          </h2>
          <p className="lede">Four scripts, run in order. Each one hands a clean artifact to the next.</p>
        </div>

        <ol className="steps">
          <span className="steps__rail" aria-hidden="true">
            <span className="steps__progress" />
          </span>
          {STEPS.map((s, i) => (
            <li className="step" key={s.title}>
              <span className="chip-icon step__icon">
                <Icon name={s.icon} size={18} />
              </span>
              <div>
                <p className="step__num">Step {i + 1}</p>
                <h3 className="step__title">{s.title}</h3>
                <p className="step__body">{s.body}</p>
                <code className="step__file">{s.file}</code>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
