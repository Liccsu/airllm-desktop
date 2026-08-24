<script setup lang="ts">
import { computed, ref } from "vue";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { api } from "../api";
import { defaultSettings, formatBytes, refreshState, store } from "../store";

const error = ref("");
const downloadingId = ref("");
const customId = ref("");
const customAlias = ref("");
const progressPercent = computed(() => {
  if (store.download.activeId !== downloadingId.value) return null;
  const total = store.download.totalBytes;
  if (!total) return null;
  return Math.min(100, Math.round((store.download.doneBytes / total) * 100));
});

const installedByModelId = computed(() => {
  const map = new Map<string, string>();
  for (const mod of store.snapshot?.installed ?? []) map.set(mod.modelId, mod.alias);
  return map;
});

const activeModel = computed(() => store.snapshot?.service.model ?? "");

async function refresh() {
  await refreshState();
}

async function download(id: string, alias?: string) {
  error.value = "";
  downloadingId.value = id;
  store.download.activeId = id;
  store.download.doneBytes = 0;
  store.download.totalBytes = 0;
  try {
    await api.downloadModel(id, alias);
    await refresh();
  } catch (e) {
    error.value = String(e);
  } finally {
    downloadingId.value = "";
    store.download.activeId = "";
  }
}

async function start(alias: string) {
  error.value = "";
  store.serviceBusy = true;
  try {
    await api.startService(alias, store.settings || defaultSettings());
    store.lastModelAlias = alias;
    await refresh();
  } catch (e) {
    error.value = String(e);
  } finally {
    store.serviceBusy = false;
  }
}

async function remove(alias: string) {
  error.value = "";
  if (!window.confirm(`确定删除模型「${alias}」？模型文件与分片将被移除。`)) return;
  try {
    await api.removeModel(alias);
    await refresh();
  } catch (e) {
    error.value = String(e);
  }
}

async function downloadCustom() {
  const id = customId.value.trim();
  if (!id) {
    error.value = "请输入 huggingface 模型 id";
    return;
  }
  await download(id, customAlias.value.trim() || undefined);
}

async function importLocal() {
  error.value = "";
  try {
    const selected = await openDialog({ directory: true, multiple: false });
    if (typeof selected !== "string" || !selected) return;
    const alias = selected.split(/[\\/]/).pop() || "";
    if (!alias) {
      error.value = "无法从目录名生成模型别名";
      return;
    }
    await api.importModel(selected, alias);
    await refresh();
  } catch (e) {
    error.value = String(e);
  }
}
</script>

<template>
  <div class="library">
    <div class="library-head">
      <h2>模型库</h2>
      <p class="head-sub">选择一个模型下载并启动。模型保存在本机数据目录中。</p>
    </div>

    <div v-if="error" class="error-bar">{{ error }}</div>

    <div v-if="store.download.activeId" class="download-card card">
      <div class="dl-row">
        <span class="mono dl-file">{{ store.download.file || "准备下载..." }}</span>
        <span>{{ progressPercent === null ? "..." : progressPercent + "%" }}</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill" :style="{ width: (progressPercent ?? 10) + '%' }"></div>
      </div>
    </div>

    <div class="grid">
      <div
        v-for="model in store.snapshot?.catalog ?? []"
        :key="model.id"
        class="card model-card"
        :class="{ strong: model.recommended }"
      >
        <div class="m-head">
          <strong>{{ model.name }}</strong>
          <span v-if="model.recommended" class="pill rec">推荐</span>
        </div>
        <div class="m-desc">{{ model.description }}</div>
        <div class="m-meta">
          <span>{{ formatBytes(model.sizeBytes) }}</span>
          <span>~{{ model.vramGb }}GB 显存</span>
          <span>{{ model.license }}</span>
        </div>
        <div class="m-actions">
          <template v-if="installedByModelId.has(model.id)">
            <button
              class="btn small primary"
              :disabled="store.serviceBusy"
              @click="start(installedByModelId.get(model.id)!)"
            >
              {{ activeModel === installedByModelId.get(model.id) ? "运行中" : "启动" }}
            </button>
            <button class="btn small ghost" @click="remove(installedByModelId.get(model.id)!)">删除</button>
          </template>
          <button
            v-else
            class="btn small primary"
            :disabled="downloadingId !== ''"
            @click="download(model.id)"
          >
            {{ downloadingId === model.id ? "下载中..." : "下载" }}
          </button>
        </div>
      </div>
    </div>

    <div class="custom card">
      <div class="section-title">自定义模型</div>
      <p class="custom-hint">输入任意 huggingface 模型 id（如 Qwen/Qwen3-4B），将与目录内模型一样下载安装。</p>
      <div class="custom-row">
        <input type="text" v-model="customId" placeholder="huggingface 模型 id" />
        <input type="text" v-model="customAlias" placeholder="别名（可选）" />
        <button class="btn" :disabled="downloadingId !== ''" @click="downloadCustom">下载</button>
      </div>
    </div>

    <div class="custom card">
      <div class="section-title">导入本地模型</div>
      <p class="custom-hint">选择一个包含 config.json 与权重文件的本地目录；以链接方式导入，不复制模型文件（别名与源目录保持一致同步）。</p>
      <div class="custom-row">
        <button class="btn ghost" @click="importLocal">选择目录并导入</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.library {
  height: 100%;
  overflow-y: auto;
  padding: 26px 28px 40px;
}
.library-head h2 {
  font-size: 20px;
  margin-bottom: 6px;
}
.head-sub {
  color: var(--text-3);
  font-size: 13px;
  margin-bottom: 20px;
}
.error-bar {
  background: rgba(224, 108, 117, 0.1);
  border: 1px solid rgba(224, 108, 117, 0.35);
  color: var(--danger);
  border-radius: 8px;
  padding: 9px 14px;
  font-size: 12.5px;
  margin-bottom: 16px;
}
.download-card {
  padding: 14px 16px;
  margin-bottom: 18px;
  max-width: 560px;
}
.dl-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 12.5px;
  color: var(--text-2);
}
.dl-file {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 14px;
}
.model-card {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.model-card.strong {
  border-color: rgba(88, 166, 240, 0.5);
}
.m-head {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14.5px;
}
.rec {
  background: var(--accent-soft);
  color: var(--accent);
  border-color: transparent;
}
.m-desc {
  font-size: 12.5px;
  color: var(--text-3);
  min-height: 34px;
}
.m-meta {
  display: flex;
  gap: 10px;
  font-size: 12px;
  color: var(--text-4);
  flex-wrap: wrap;
}
.m-actions {
  display: flex;
  gap: 8px;
  margin-top: 6px;
}
.custom {
  margin-top: 26px;
  padding: 18px 20px;
  max-width: 760px;
}
.custom-hint {
  color: var(--text-3);
  font-size: 12.5px;
  margin-bottom: 12px;
}
.custom-row {
  display: flex;
  gap: 10px;
}
.custom-row input:first-child {
  flex: 1.6;
}
.custom-row input:nth-child(2) {
  flex: 1;
}
</style>
