// 全局响应式状态：应用快照、视图导航与运行时事件。

import { reactive } from "vue";
import { api, type AppSnapshot, type ServiceSettings } from "./api";

export const store = reactive({
  snapshot: null as AppSnapshot | null,
  loading: true,
  error: "",
  settings: null as ServiceSettings | null,
  sourceEndpoint: "",
  view: "chat" as "chat" | "library" | "settings" | "logs",
  serviceBusy: false,
  lastModelAlias: "",
  envInstall: {
    running: false,
    steps: [] as { index: number; total: number; name: string; status: string }[],
    lines: [] as string[],
  },
  download: {
    activeId: "",
    doneBytes: 0,
    totalBytes: 0,
    file: "",
    stage: "",
  },
  serviceLogs: [] as { ts: string; text: string; stream: "out" | "err" }[],
  serviceEvents: [] as Record<string, unknown>[],
});

export function defaultSettings(): ServiceSettings {
  return {
    port: 8000,
    device: "cuda:0",
    maxSeqLen: 512,
    maxOutputTokens: 128,
    preload: true,
    endpoint: "",
    modelRoot: "",
    downloadWorkers: 8,
  };
}

export async function refreshState(): Promise<AppSnapshot> {
  store.snapshot = await api.getState();
  if (store.snapshot && !store.lastModelAlias && store.snapshot.installed.length > 0) {
    store.lastModelAlias = store.snapshot.installed[0].alias;
  }
  return store.snapshot;
}

export async function bootstrap(): Promise<void> {
  store.loading = true;
  try {
    await refreshState();
    // 恢复上次选择的模型源（官方源留空即默认）。
    try {
      const saved = localStorage.getItem("airllm.endpoint");
      if (saved !== null) {
        store.sourceEndpoint = saved;
        if (saved) {
          store.settings = { ...defaultSettings(), endpoint: saved };
          api.updateSettings({ ...defaultSettings(), endpoint: saved }).catch(() => undefined);
        }
      }
    } catch {
      // localStorage 不可用时忽略
    }
  } catch (error) {
    store.error = String(error);
  } finally {
    store.loading = false;
  }
}

export function needsOnboarding(snapshot: AppSnapshot | null): boolean {
  if (!snapshot) return true;
  return !snapshot.env.depsReady || snapshot.installed.length === 0;
}

export function formatBytes(bytes: number): string {
  if (!bytes) return "未知";
  const gb = bytes / 1e9;
  return gb >= 1 ? `${gb.toFixed(1)}GB` : `${(bytes / 1e6).toFixed(0)}MB`;
}
