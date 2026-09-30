import { useRef } from "react";
import { Icon } from "./Icon";
import { gsap, useMotion } from "../lib/motion";
import "./Nav.css";

export function Nav() {
  const ref = useRef<HTMLElement>(null);

  useMotion(() => {
    gsap.from(".nav__inner > *", { y: -12, opacity: 0, duration: 0.9, stagger: 0.08, delay: 0.1 });
  }, ref);

  return (
    <header className="nav" ref={ref}>
      <div className="nav__inner">
        <a className="nav__brand" href="#top" aria-label="Crack Analysis — home">
          <span className="nav__mark">
            <Icon name="crack" size={16} strokeWidth={1.8} />
          </span>
          Crack Analysis
        </a>
        <nav className="nav__links" aria-label="Sections">
          <a href="#how">How it works</a>
          <a href="#model">Model</a>
          <a
            href="https://github.com/KritishX/Crack-Analysis-using-AI"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub repository"
            className="nav__gh"
          >
            <Icon name="github" size={18} />
          </a>
          <a className="btn btn--primary btn--small" href="#analyze">
            Analyze
          </a>
        </nav>
      </div>
    </header>
  );
}
