<script setup lang="ts">
import { computed, onMounted, onUnmounted } from "vue";
import { api } from "../api";
import { refreshState, store } from "../store";
import Chat from "./Chat.vue";
import Library from "./Library.vue";
import Settings from "./Settings.vue";
import Logs from "./Logs.vue";

const views = {
  chat: Chat,
  library: Library,
  settings: Settings,
  logs: Logs,
} as const;

const svgAttrs = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"';

const navItems = [
  { key: "chat", label: "聊天", icon: `<svg ${svgAttrs} width="17" height="17"><path d="M5 4h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-6l-4 4v-4H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z"/></svg>` },
  { key: "library", label: "模型库", icon: `<svg ${svgAttrs} width="17" height="17"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>` },
  { key: "settings", label: "设置", icon: `<svg ${svgAttrs} width="17" height="17"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.5v2.6M12 18.9v2.6M2.5 12h2.6M18.9 12h2.6M5.3 5.3l1.9 1.9M16.8 16.8l1.9 1.9M18.7 5.3l-1.9 1.9M7.2 16.8l-1.9 1.9"/></svg>` },
  { key: "logs", label: "日志", icon: `<svg ${svgAttrs} width="17" height="17"><path d="M4 6h16M4 12h16M4 18h10"/></svg>` },
] as const;

const current = computed(() => views[store.view]);
const service = computed(() => store.snapshot?.service);
const serviceLabel = computed(() => {
  const s = service.value;
  if (!s) return "停止";
  if (s.ready) return "运行中";
  if (s.running) return "启动中";
  return "停止";
});
const serviceDot = computed(() => {
  const s = service.value;
  if (!s) return "idle";
  if (s.ready) return "ok";
  if (s.running) return "warn";
  return "idle";
});

let refreshTimer: ReturnType<typeof setInterval> | null = null;

onMounted(() => {
  refreshTimer = setInterval(() => {
    void refreshState().catch(() => undefined);
  }, 10000);
});

onUnmounted(() => {
  if (refreshTimer) clearInterval(refreshTimer);
});

async function stopService() {
  try {
    await api.stopService();
    await refreshState();
  } catch (e) {
    console.error(e);
  }
}
</script>

<template>
  <div class="home">
    <aside class="sidebar">
      <div class="brand">AirLLM<span class="brand-dot">.</span></div>
      <nav>
        <button
          v-for="item in navItems"
          :key="item.key"
          class="nav-item"
          :class="{ active: store.view === item.key }"
          @click="store.view = item.key"
        >
          <span class="nav-icon" v-html="item.icon"></span>
          {{ item.label }}
        </button>
      </nav>
      <div class="sidebar-foot">
        <div class="gpu-card card">
          <div class="gpu-name">GPU {{ store.snapshot?.gpu ? store.snapshot.gpu.name : "未检测到 NVIDIA 显卡" }}</div>
          <div v-if="store.snapshot?.gpu" class="gpu-mem mono">
            显存 {{ (store.snapshot.gpu.vramUsedMb / 1024).toFixed(1) }} / {{ (store.snapshot.gpu.vramTotalMb / 1024).toFixed(0) }} GB
          </div>
          <div v-if="store.snapshot" class="gpu-mem mono">
            内存 {{ (store.snapshot.memory.usedMb / 1024).toFixed(1) }} / {{ (store.snapshot.memory.totalMb / 1024).toFixed(0) }} GB
          </div>
        </div>
      </div>
    </aside>

    <main class="content">
      <header class="topbar">
        <div class="status-group">
          <span class="dot" :class="serviceDot"></span>
          <span class="status-text">{{ serviceLabel }}</span>
          <span v-if="service?.model" class="pill model-pill">{{ service.model }}</span>
          <span v-if="service?.ready" class="service-port mono">:{{ service.port }}</span>
        </div>
        <button
          v-if="service?.running"
          class="btn small ghost"
          @click="stopService"
        >停止服务</button>
      </header>
      <div class="view">
        <KeepAlive>
          <component :is="current" />
        </KeepAlive>
      </div>
    </main>
  </div>
</template>

<style scoped>
.home {
  height: 100%;
  display: flex;
}
.sidebar {
  width: 220px;
  flex: none;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 20px 14px;
}
.brand {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.02em;
  padding: 4px 10px 24px;
}
.brand-dot {
  color: var(--accent);
}
nav {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--text-2);
  font-size: 14px;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, color 0.15s ease;
}
.nav-item:hover {
  background: var(--surface-2);
  color: var(--text);
}
.nav-item.active {
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 600;
}
.nav-icon {
  font-size: 15px;
}
.sidebar-foot {
  margin-top: auto;
}
.gpu-card {
  padding: 12px 14px;
}
.gpu-name {
  font-size: 12.5px;
  font-weight: 600;
  margin-bottom: 4px;
  color: var(--text-2);
}
.gpu-mem {
  font-size: 12px;
  color: var(--text-3);
}
.gpu-note {
  font-size: 12px;
  color: var(--text-4);
  padding: 12px 14px;
}
.content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.topbar {
  height: 52px;
  flex: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  border-bottom: 1px solid var(--border);
}
.status-group {
  display: flex;
  align-items: center;
  gap: 10px;
}
.status-text {
  font-size: 13px;
  color: var(--text-2);
}
.model-pill {
  font-size: 12px;
}
.service-port {
  font-size: 12px;
  color: var(--text-4);
}
.view {
  flex: 1;
  min-height: 0;
}
</style>
