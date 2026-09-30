import { useCallback, useEffect, useState } from "react";
import { fetchHealth, type ServiceState } from "./api";

export function useService() {
  const [state, setState] = useState<ServiceState>({ kind: "checking" });

  const check = useCallback((signal?: AbortSignal) => {
    setState({ kind: "checking" });
    fetchHealth(signal).then(setState, () => {
      /* aborted */
    });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    check(controller.signal);
    return () => controller.abort();
  }, [check]);

  return { state, recheck: () => check() };
}
