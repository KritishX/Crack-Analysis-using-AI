import { useCallback, useEffect, useRef, useState, type DragEvent, type RefObject } from "react";
import { Icon } from "./Icon";
import { Words } from "./Words";
import { ACCEPTED_TYPES, formatBytes, predict, validateFile, type Prediction } from "../lib/api";
import { gsap, useMotion } from "../lib/motion";
import { useService } from "../lib/useService";
import "./Analyzer.css";

interface Entry {
  id: string;
  file: File;
  url: string;
  status: "loading" | "done" | "error";
  result?: Prediction;
  error?: string;
}

const HISTORY_LIMIT = 8;
const LOW_CONFIDENCE = 0.7;

let nextId = 0;

export function Analyzer() {
  const root = useRef<HTMLElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const confidenceEl = useRef<HTMLSpanElement>(null);

  const { state: service, recheck } = useService();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const entriesRef = useRef(entries);
  entriesRef.current = entries;

  const active = entries.find((e) => e.id === activeId) ?? null;

  const patch = useCallback((id: string, next: Partial<Entry>) => {
    setEntries((prev) => prev.map((e) => (e.id === id ? { ...e, ...next } : e)));
  }, []);

  const run = useCallback(
    (id: string, file: File) => {
      predict(file).then(
        (result) => patch(id, { status: "done", result, error: undefined }),
        (err: Error) => patch(id, { status: "error", error: err.message }),
      );
    },
    [patch],
  );

  const addFile = useCallback(
    (file: File) => {
      const problem = validateFile(file);
      setNotice(problem);
      if (problem) return;

      const entry: Entry = {
        id: String(++nextId),
        file,
        url: URL.createObjectURL(file),
        status: "loading",
      };
      setEntries((prev) => {
        const next = [entry, ...prev];
        next.slice(HISTORY_LIMIT).forEach((dropped) => URL.revokeObjectURL(dropped.url));
        return next.slice(0, HISTORY_LIMIT);
      });
      setActiveId(entry.id);
      run(entry.id, file);
    },
    [run],
  );

  const retry = (entry: Entry) => {
    patch(entry.id, { status: "loading", error: undefined });
    run(entry.id, entry.file);
    recheck();
  };

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (file) addFile(file);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [addFile]);

  useEffect(
    () => () => entriesRef.current.forEach((e) => URL.revokeObjectURL(e.url)),
    [],
  );

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) addFile(file);
  };

  const copyResult = async () => {
    if (!active?.result) return;
    const payload = { file: active.file.name, ...active.result };
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable (insecure context or denied) */
    }
  };

  // Section entrance.
  useMotion(() => {
    gsap.from(".analyzer__head .w > span", {
      yPercent: 115,
      duration: 1.1,
      stagger: 0.06,
      ease: "expo.out",
      scrollTrigger: { trigger: ".analyzer__head", start: "top 82%" },
    });
    gsap.from(".analyzer__head .eyebrow, .analyzer__status", {
      y: 12,
      opacity: 0,
      stagger: 0.1,
      scrollTrigger: { trigger: ".analyzer__head", start: "top 82%" },
    });
    gsap.from(".analyzer__card", {
      y: 48,
      opacity: 0,
      duration: 1.2,
      ease: "expo.out",
      scrollTrigger: { trigger: ".analyzer__card", start: "top 88%" },
    });
  }, root);

  // Per-analysis motion: scan line while loading, reveal when the result lands.
  useMotion(
    () => {
      if (!active) return;

      if (active.status === "loading") {
        gsap.fromTo(
          ".media__scan",
          { top: "0%" },
          { top: "100%", duration: 1.3, ease: "sine.inOut", repeat: -1, yoyo: true },
        );
      }

      if (active.status === "done" && active.result) {
        const target = active.result.confidence * 100;
        const counter = { v: 0 };
        const el = confidenceEl.current;
        if (el) el.textContent = "0.0";

        gsap
          .timeline({ defaults: { ease: "expo.out" } })
          .from(".verdict__badge", { scale: 0.6, opacity: 0, duration: 0.7, ease: "back.out(2)" })
          .from(".verdict__title .w > span", { yPercent: 110, duration: 0.9, stagger: 0.06 }, "-=0.5")
          .to(
            counter,
            {
              v: target,
              duration: 1.3,
              onUpdate: () => {
                if (el) el.textContent = counter.v.toFixed(1);
              },
              onComplete: () => {
                if (el) el.textContent = target.toFixed(1);
              },
            },
            "-=0.7",
          )
          .from(".bar__fill", { scaleX: 0, duration: 1.2, stagger: 0.1 }, "-=1.2")
          .from(".meta__item", { y: 10, opacity: 0, duration: 0.7, stagger: 0.06 }, "-=0.9")
          .from(".result__actions > *", { y: 8, opacity: 0, duration: 0.6, stagger: 0.06 }, "-=0.6");
      }

      if (active.status === "error") {
        gsap.from(".panel__error", { y: 10, opacity: 0, duration: 0.6 });
      }
    },
    root,
    [active?.id, active?.status],
  );

  const offline = service.kind === "no_model" || service.kind === "unreachable";

  return (
    <section className="analyzer section" id="analyze" ref={root}>
      <div className="container">
        <header className="analyzer__head">
          <p className="eyebrow">
            <Icon name="image" size={16} />
            Analyze
          </p>
          <h2 className="title">
            <Words text="Show it a surface." /> <Words text="Get a verdict." className="serif" />
          </h2>
          <ServicePill state={service} />
        </header>

        <div className="analyzer__card">
          {/* ---------- media ---------- */}
          <div
            className={`media${dragging ? " media--drag" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
            }}
            onDrop={onDrop}
          >
            <input
              ref={input}
              type="file"
              accept={ACCEPTED_TYPES.join(",")}
              hidden
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) addFile(file);
                e.target.value = "";
              }}
            />

            {active ? (
              <>
                <img className="media__img" src={active.url} alt={`Uploaded image: ${active.file.name}`} />
                {active.status === "loading" && <div className="media__scan" aria-hidden="true" />}
                <div className="media__bar">
                  <span className="media__name">{active.file.name}</span>
                  <button className="btn btn--small media__replace" type="button" onClick={() => input.current?.click()}>
                    <Icon name="upload" size={15} />
                    Replace
                  </button>
                </div>
              </>
            ) : (
              <button className="drop" type="button" onClick={() => input.current?.click()}>
                <span className="chip-icon drop__icon">
                  <Icon name="upload" size={20} />
                </span>
                <span className="drop__title">Drop an image here</span>
                <span className="drop__hint">
                  or <u>browse</u> · paste with ⌘V / Ctrl+V
                </span>
                <span className="drop__fine">JPEG, PNG, WebP or BMP · up to 10 MB</span>
              </button>
            )}

            {dragging && (
              <div className="media__overlay" aria-hidden="true">
                Release to analyze
              </div>
            )}
          </div>

          {/* ---------- panel ---------- */}
          <div className="panel" aria-live="polite">
            {notice && (
              <div className="callout panel__notice" role="alert">
                <Icon name="alert" size={18} />
                <span>{notice}</span>
              </div>
            )}

            {!active && <Idle offline={offline} state={service} recheck={recheck} />}

            {active?.status === "loading" && (
              <div className="panel__loading">
                <p className="eyebrow">Analyzing</p>
                <p className="panel__loading-title">Looking at the surface&hellip;</p>
                <div className="skeleton" />
                <div className="skeleton skeleton--short" />
              </div>
            )}

            {active?.status === "error" && (
              <div className="panel__error">
                <div className="callout" role="alert">
                  <Icon name="alert" size={18} />
                  <span>
                    <strong>Couldn&rsquo;t analyze this image.</strong>
                    <br />
                    {active.error}
                  </span>
                </div>
                {service.kind === "unreachable" && <ServerHelp state={service} />}
                <button className="btn btn--primary" type="button" onClick={() => retry(active)}>
                  <Icon name="refresh" size={17} />
                  Try again
                </button>
              </div>
            )}

            {active?.status === "done" && active.result && (
              <Result
                entry={active}
                result={active.result}
                confidenceRef={confidenceEl}
                copied={copied}
                onCopy={copyResult}
                onAnother={() => input.current?.click()}
              />
            )}
          </div>
        </div>

        {entries.length > 0 && (
          <div className="history" role="list" aria-label="Recent images">
            {entries.map((e) => (
              <button
                key={e.id}
                role="listitem"
                type="button"
                className={`history__item${e.id === activeId ? " is-active" : ""}`}
                onClick={() => setActiveId(e.id)}
                aria-label={`${e.file.name}${e.result ? (e.result.label === "crack" ? ", crack" : ", no crack") : ""}`}
                aria-current={e.id === activeId}
              >
                <img src={e.url} alt="" />
                {e.status === "done" && e.result && (
                  <i className={`history__dot history__dot--${e.result.label}`} />
                )}
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */

function ServicePill({ state }: { state: ReturnType<typeof useService>["state"] }) {
  const text =
    state.kind === "checking"
      ? "Checking model…"
      : state.kind === "online"
        ? `Model online · ${state.device.toUpperCase()}`
        : state.kind === "no_model"
          ? "Model weights missing"
          : "Inference server offline";
  return (
    <p className={`analyzer__status pill pill--${state.kind}`}>
      <i />
      {text}
    </p>
  );
}

function ServerHelp({ state }: { state: ReturnType<typeof useService>["state"] }) {
  return (
    <div className="callout">
      <Icon name="terminal" size={18} />
      <div>
        {state.kind === "no_model" ? (
          <>
            <strong>No trained weights found.</strong> Run the training script once to produce{" "}
            <code>best_model.pth</code>, then restart the API.
            <code className="code">python model_training.py</code>
          </>
        ) : (
          <>
            <strong>Start the inference server</strong> from the repository root.
            <code className="code">{"pip install -r api/requirements.txt\nuvicorn api.main:app --port 8000"}</code>
          </>
        )}
      </div>
    </div>
  );
}

function Idle({
  offline,
  state,
  recheck,
}: {
  offline: boolean;
  state: ReturnType<typeof useService>["state"];
  recheck: () => void;
}) {
  return (
    <div className="idle">
      <p className="eyebrow">Result</p>
      <p className="idle__title">Waiting for an image.</p>
      <ul className="tips">
        <li>
          <span className="chip-icon"><Icon name="crack" size={18} /></span>
          <span>
            <strong>Get close.</strong> The model learned from 256 × 256 px patches, so tight crops of
            a few square feet work best.
          </span>
        </li>
        <li>
          <span className="chip-icon"><Icon name="image" size={18} /></span>
          <span>
            <strong>Even light.</strong> Hard shadows and glare can look like cracks. Shoot straight
            on when you can.
          </span>
        </li>
        <li>
          <span className="chip-icon"><Icon name="shield" size={18} /></span>
          <span>
            <strong>A screen, not an inspection.</strong> This flags likely cracks; it doesn&rsquo;t
            replace an engineer.
          </span>
        </li>
      </ul>
      {offline && (
        <>
          <ServerHelp state={state} />
          <button className="btn btn--quiet btn--small" type="button" onClick={recheck}>
            <Icon name="refresh" size={15} />
            Check again
          </button>
        </>
      )}
    </div>
  );
}

function Result({
  entry,
  result,
  confidenceRef,
  copied,
  onCopy,
  onAnother,
}: {
  entry: Entry;
  result: Prediction;
  confidenceRef: RefObject<HTMLSpanElement | null>;
  copied: boolean;
  onCopy: () => void;
  onAnother: () => void;
}) {
  const cracked = result.label === "crack";
  const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

  return (
    <div className={`result result--${result.label}`}>
      <div className="verdict">
        <span className="verdict__badge">
          <Icon name={cracked ? "crack" : "check"} size={22} strokeWidth={1.8} />
        </span>
        <h3 className="verdict__title">
          <Words text={cracked ? "Crack detected" : "No crack detected"} />
        </h3>
      </div>

      <p className="confidence">
        <span className="confidence__value">
          <span ref={confidenceRef}>{(result.confidence * 100).toFixed(1)}</span>
          <small>%</small>
        </span>
        <span className="confidence__label">confidence</span>
      </p>

      {result.confidence < LOW_CONFIDENCE && (
        <div className="callout">
          <Icon name="alert" size={18} />
          <span>
            <strong>Close call.</strong> Try a closer, more evenly lit crop for a firmer answer.
          </span>
        </div>
      )}

      <div className="bars">
        {(
          [
            ["Crack", result.probabilities.crack, "crack"],
            ["No crack", result.probabilities.no_crack, "no_crack"],
          ] as const
        ).map(([label, value, key]) => (
          <div className="bar" key={key}>
            <div className="bar__row">
              <span>{label}</span>
              <span className="bar__value">{pct(value)}</span>
            </div>
            <div className="bar__track">
              <div className={`bar__fill bar__fill--${key}`} style={{ transform: `scaleX(${value})` }} />
            </div>
          </div>
        ))}
      </div>

      <dl className="meta">
        <div className="meta__item">
          <dt>Dimensions</dt>
          <dd>{result.width} × {result.height}</dd>
        </div>
        <div className="meta__item">
          <dt>File size</dt>
          <dd>{formatBytes(entry.file.size)}</dd>
        </div>
        <div className="meta__item">
          <dt>Inference</dt>
          <dd>{result.inference_ms} ms</dd>
        </div>
        <div className="meta__item">
          <dt>Device</dt>
          <dd>{result.device.toUpperCase()}</dd>
        </div>
      </dl>

      <div className="result__actions">
        <button className="btn btn--primary" type="button" onClick={onAnother}>
          <Icon name="upload" size={17} />
          Analyze another
        </button>
        <button className="btn btn--quiet" type="button" onClick={onCopy}>
          <Icon name={copied ? "check" : "copy"} size={17} />
          {copied ? "Copied" : "Copy JSON"}
        </button>
      </div>
    </div>
  );
}
