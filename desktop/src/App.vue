<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { on } from "./api";
import { bootstrap, needsOnboarding, refreshState, store } from "./store";
import Onboarding from "./views/Onboarding.vue";
import Home from "./views/Home.vue";

const showOnboarding = ref(false);

let unlisteners: (() => void)[] = [];

onMounted(async () => {
  await bootstrap();
  showOnboarding.value = needsOnboarding(store.snapshot);
  unlisteners.push(
    await on<{ success: boolean; modelId: string; alias: string }>("model-done", async () => {
      await refreshState();
    }),
    await on<{ index: number; total: number; name: string; status: string }>(
      "env-step",
      (payload) => {
        const existing = store.envInstall.steps.find((s) => s.index === payload.index);
        if (existing) existing.status = payload.status;
        else store.envInstall.steps.push(payload);
      },
    ),
    await on<{ text: string }>("env-line", (payload) => {
      store.envInstall.lines.push(payload.text);
      if (store.envInstall.lines.length > 500) store.envInstall.lines.shift();
    }),
    await on<Record<string, unknown>>("model-progress", (payload) => {
      // 引擎事件字段为 snake_case(done_bytes/total_bytes)；兼容 camelCase。
      const stage = String(payload.stage ?? "");
      const file = String(payload.file ?? "");
      const doneBytes = Number(
        payload.done_bytes ?? payload.doneBytes ?? 0,
      );
      const totalBytes = Number(
        payload.total_bytes ?? payload.totalBytes ?? 0,
      );
      if (stage === "started") {
        store.download.stage = stage;
        store.download.file = file;
      }
      store.download.doneBytes = doneBytes;
      store.download.totalBytes = totalBytes;
      if (stage === "done") store.download.file = "";
    }),
        await on<{ text: string }>("model-progress-line", (payload) => {
      store.serviceLogs.push({
        ts: new Date().toLocaleTimeString(),
        text: `[模型下载] ${payload.text}`,
        stream: "out",
      });
      if (store.serviceLogs.length > 1200) store.serviceLogs.splice(0, store.serviceLogs.length - 1200);
    }),
await on<Record<string, unknown>>("service-event", (payload) => {
      const eventName = String(payload.event ?? "");
      const ts = new Date().toLocaleTimeString();
      let text = JSON.stringify(payload, null, 2);
      if (eventName === "serve.started") {
        text = `服务已启动 http://127.0.0.1:${payload.port ?? ""}`;
      } else if (eventName === "serve.stopping") {
        text = "服务正在停止...";
      }
      store.serviceLogs.push({ ts, text, stream: "out" });
      if (store.serviceLogs.length > 1200) store.serviceLogs.splice(0, store.serviceLogs.length - 1200);
      void refreshState();
    }),
    await on<{ text: string }>("service-line", (payload) => {
      store.serviceLogs.push({
        ts: new Date().toLocaleTimeString(),
        text: payload.text,
        stream: "out",
      });
      if (store.serviceLogs.length > 1200) store.serviceLogs.splice(0, store.serviceLogs.length - 1200);
    }),
  );
});

onUnmounted(() => {
  unlisteners.forEach((unlisten) => unlisten());
});

const loading = computed(() => store.loading);
</script>

<template>
  <div v-if="loading" class="boot">
    <div class="boot-logo">AirLLM</div>
    <div class="boot-hint">正在准备引擎...</div>
    <div v-if="store.error" class="boot-error mono">{{ store.error }}</div>
  </div>
  <Onboarding v-else-if="showOnboarding" @finished="showOnboarding = false" />
  <Home v-else />
</template>

<style scoped>
.boot {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
}
.boot-logo {
  font-size: 32px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.02em;
}
.boot-hint {
  color: var(--text-3);
  font-size: 13px;
}
</style>
