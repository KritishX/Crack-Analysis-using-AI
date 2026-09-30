const BASE = import.meta.env.VITE_API_URL ?? "";

export interface Prediction {
  label: "crack" | "no_crack";
  confidence: number;
  probabilities: { no_crack: number; crack: number };
  inference_ms: number;
  width: number;
  height: number;
  device: string;
}

export interface Health {
  status: "ok" | "no_model";
  model_loaded: boolean;
  device: string;
}

export type ServiceState =
  | { kind: "checking" }
  | { kind: "online"; device: string }
  | { kind: "no_model" }
  | { kind: "unreachable" };

export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
export const MAX_BYTES = 10 * 1024 * 1024;

export async function fetchHealth(signal?: AbortSignal): Promise<ServiceState> {
  try {
    const res = await fetch(`${BASE}/api/health`, { signal });
    if (!res.ok) return { kind: "unreachable" };
    const body = (await res.json()) as Health;
    return body.model_loaded ? { kind: "online", device: body.device } : { kind: "no_model" };
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    return { kind: "unreachable" };
  }
}

export async function predict(file: File): Promise<Prediction> {
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${BASE}/api/predict`, { method: "POST", body: form });
  } catch {
    throw new Error("Can't reach the inference server. Is it running?");
  }

  if (!res.ok) {
    let detail =
      res.status >= 502 && res.status <= 504
        ? "Can't reach the inference server. Is it running?"
        : `The server responded with ${res.status}.`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body; keep the status message */
    }
    throw new Error(detail);
  }
  return (await res.json()) as Prediction;
}

export function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return "Use a JPEG, PNG, WebP or BMP image.";
  if (file.size > MAX_BYTES) return "Images must be under 10 MB.";
  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
