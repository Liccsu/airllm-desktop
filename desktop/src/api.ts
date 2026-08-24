// 与 Rust 侧命令/事件的类型化封装。

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface GpuInfo {
  name: string;
  vramTotalMb: number;
  vramUsedMb: number;
}

export interface CatalogEntry {
  id: string;
  name: string;
  description: string;
  sizeBytes: number;
  vramGb: number;
  license: string;
  recommended: boolean;
}

export interface InstalledModel {
  alias: string;
  modelId: string;
  revision: string;
  modelDir: string;
}

export interface ServiceStatus {
  running: boolean;
  ready: boolean;
  port: number;
  model: string | null;
  device: string;
}

export interface EnvStatus {
  pythonOk: boolean;
  depsReady: boolean;
}

export interface ServiceSettings {
  port: number;
  device: string;
  maxSeqLen: number;
  maxOutputTokens: number;
  preload: boolean;
  endpoint: string;
  modelRoot: string;
  downloadWorkers: number;
}

export interface MemoryInfo {
  totalMb: number;
  usedMb: number;
}

export interface AppSnapshot {
  dataDir: string;
  catalog: CatalogEntry[];
  installed: InstalledModel[];
  env: EnvStatus;
  service: ServiceStatus;
  gpu: GpuInfo | null;
  memory: MemoryInfo;
  apiKey: string;
}

export const api = {
  getState: () => invoke<AppSnapshot>("get_state"),
  installEnv: () => invoke<boolean>("install_env"),
  downloadModel: (modelId: string, alias?: string, hfToken?: string) =>
    invoke<void>("download_model", { modelId, alias: alias ?? null, hfToken: hfToken ?? null }),
  cancelDownload: () => invoke<void>("cancel_download"),
  removeModel: (alias: string) => invoke<void>("remove_model", { alias }),
  importModel: (dir: string, alias: string) => invoke<void>("import_model", { dir, alias }),
  startService: (alias: string, settings: ServiceSettings) =>
    invoke<void>("start_service", { alias, settings }),
  stopService: () => invoke<void>("stop_service"),
  updateSettings: (settings: ServiceSettings) => invoke<void>("update_settings", { settings }),
  openPath: (target: string) => invoke<void>("open_path", { target }),
  getApiKey: () => invoke<string>("get_api_key"),
};

export async function on<T>(event: string, handler: (payload: T) => void): Promise<UnlistenFn> {
  return listen<T>(event, (eventData) => handler(eventData.payload));
}

/** 简洁 HTTP GET(健康检查使用,绕过浏览器限制在 Tauri 中调用前端 fetch)。 */
export async function fetchText(url: string, timeoutMs = 3000): Promise<string | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!response.ok) return null;
    return await response.text();
  } catch {
    return null;
  }
}
