<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { api } from "../api";
import { defaultSettings, formatBytes, refreshState, store } from "../store";

const emit = defineEmits<{ finished: [] }>();

const step = ref(0);
const error = ref("");

const envLogEl = ref<HTMLElement | null>(null);

async function scrollEnvLog() {
  await nextTick();
  if (envLogEl.value) envLogEl.value.scrollTop = envLogEl.value.scrollHeight;
}

// 环境安装日志自动跟随最新行（以 length 为依赖，push 不改变数组引用）。
watch(() => store.envInstall.lines.length, () => void scrollEnvLog());
const stepsMeta = [
  { title: "欢迎使用 AirLLM", sub: "本地大模型，掌控在你的电脑里" },
  { title: "安装运行环境", sub: "自动准备 Python 与 PyTorch，无需任何命令行" },
  { title: "选择模型源", sub: "官方源或国内镜像，之后可修改" },
  { title: "选择模型", sub: "下载一个模型即可开始对话" },
  { title: "完成了！", sub: "启动服务，开始聊天" },
];

const selectedSource = ref<"official" | "mirror" | "custom">("official");
const customEndpoint = ref("");

function endpointFor(source: string): string {
  if (source === "mirror") return "https://hf-mirror.com";
  if (source === "custom") return customEndpoint.value.trim() || "https://hf-mirror.com";
  return "";
}

const modelRootInput = ref("");

async function applySource(goNext: boolean) {
  const value = endpointFor(selectedSource.value);
  const settings = { ...defaultSettings(), endpoint: value, modelRoot: modelRootInput.value.trim() };
  store.sourceEndpoint = value;
  store.settings = settings;
  try {
    await api.updateSettings(settings);
  } catch (e) {
    error.value = String(e);
  }
  try {
    localStorage.setItem("airllm.endpoint", value);
  } catch {
    // localStorage 不可用时仅会话内生效
  }
  if (goNext) step.value = 3;
}

const snapshot = computed(() => store.snapshot);
const gpu = computed(() => snapshot.value?.gpu ?? null);
const envInstalled = computed(() => snapshot.value?.env.depsReady ?? false);
const installing = computed(() => store.envInstall.running);
const installedIds = computed(() => new Set(snapshot.value?.installed.map((m) => m.modelId) ?? []));
const downloading = computed(() => store.download.activeId !== "");

async function refresh() {
  await refreshState();
}

async function installEnv() {
  error.value = "";
  store.envInstall.running = true;
  store.envInstall.lines = [];
  try {
    await api.installEnv();
    await refresh();
    if (store.snapshot?.env.depsReady) step.value = 2;
  } catch (e) {
    error.value = String(e);
  } finally {
    store.envInstall.running = false;
  }
}

async function startDownload(id: string, alias?: string) {
  error.value = "";
  store.download.activeId = id;
  store.download.doneBytes = 0;
  store.download.totalBytes = 0;
  try {
    await api.downloadModel(id, alias, store.hfToken || undefined);
    await refresh();
  } catch (e) {
    error.value = String(e);
  } finally {
    store.download.activeId = "";
  }
}

async function cancelDownload() {
  error.value = "";
  try {
    await api.cancelDownload();
  } catch (e) {
    error.value = String(e);
  } finally {
    store.download.activeId = "";
  }
}

async function startService(alias: string) {
  error.value = "";
  store.serviceBusy = true;
  try {
    await api.startService(alias, defaultSettings());
    store.lastModelAlias = alias;
    await refresh();
  } catch (e) {
    error.value = String(e);
  } finally {
    store.serviceBusy = false;
  }
}

function skipEnv() {
  step.value = 2;
}

const progressPercent = computed(() => {
  const total = store.download.totalBytes;
  if (!total) return null;
  return Math.min(100, Math.round((store.download.doneBytes / total) * 100));
});

const progressText = computed(() => {
  const total = store.download.totalBytes;
  if (!total) return `${(store.download.doneBytes / 1e9).toFixed(1)}GB 下载中...`;
  const pct = progressPercent.value ?? 0;
  return `${(store.download.doneBytes / 1e9).toFixed(1)}GB / ${(total / 1e9).toFixed(1)}GB  ${pct}%`;
});

async function finish() {
  emit("finished");
}
</script>

<template>
  <div class="onboarding">
    <aside class="side">
      <div class="brand">AirLLM<span class="brand-dot">.</span></div>
      <ol class="steps">
        <li v-for="(meta, index) in stepsMeta" :key="index" :class="{ active: step === index, done: step > index }">
          <span class="num">{{ step > index ? "✓" : index + 1 }}</span>
          <div>
            <div class="s-title">{{ meta.title }}</div>
            <div class="s-sub">{{ meta.sub }}</div>
          </div>
        </li>
      </ol>
      <div class="side-note">模型与数据仅保存在本机。</div>
    </aside>

    <main class="panel">
      <template v-if="step === 0">
        <h1>欢迎使用 AirLLM</h1>
        <p class="lead">
          一个在本地运行大模型的桌面应用：下载模型、一键启动、开箱即用。
          所有推理都发生在你的电脑上，数据不出本机。
        </p>
        <div class="hardware card">
          <div class="hw-row">
            <span class="hw-label">GPU</span>
            <span>{{ gpu ? gpu.name : "未检测到 NVIDIA 显卡" }}</span>
          </div>
          <div v-if="gpu" class="hw-row">
            <span class="hw-label">显存</span>
            <span>{{ (gpu.vramTotalMb / 1024).toFixed(0) }} GB（已用 {{ (gpu.vramUsedMb / 1024).toFixed(1) }} GB）</span>
          </div>
          <div class="hw-row">
            <span class="hw-label">数据目录</span>
            <span class="mono">{{ snapshot?.dataDir }}</span>
          </div>
        </div>
        <div v-if="error" class="error">{{ error }}</div>
        <div class="actions">
          <button class="btn primary" @click="step = 1">开始配置</button>
        </div>
      </template>

      <template v-else-if="step === 1">
        <h1>安装运行环境</h1>
        <p class="lead">
          将自动下载 Python 3.11、PyTorch（CUDA 12.8，约 2.5GB）与引擎依赖，
          整个过程 10-20 分钟，取决于网络速度。
        </p>
        <div v-if="envInstalled && !installing" class="ok-box">✓ 运行环境已就绪</div>
        <ul v-if="store.envInstall.steps.length" class="steps-list">
          <li v-for="s in store.envInstall.steps" :key="s.index" :class="s.status">
            <span class="dot" :class="s.status === 'done' ? 'ok' : s.status === 'failed' ? 'bad' : 'warn'"></span>
            {{ s.name }}
            <span class="step-n">{{ `${s.index}/${s.total}` }}</span>
          </li>
        </ul>
        <div ref="envLogEl" class="log-box mono" v-if="store.envInstall.lines.length">
          <div v-for="(line, i) in store.envInstall.lines.slice(-200)" :key="i">{{ line }}</div>
        </div>
        <div v-if="error" class="error">{{ error }}</div>
        <div class="actions">
          <template v-if="!envInstalled">
            <button class="btn primary" :disabled="installing" @click="installEnv">
              {{ installing ? "安装中..." : "开始安装" }}
            </button>
            <button class="btn ghost" :disabled="installing" @click="skipEnv">跳过（稍后安装）</button>
          </template>
          <button v-else class="btn primary" @click="step = 2">下一步</button>
        </div>
      </template>

      <template v-else-if="step === 2">
        <h1>模型源与下载目录</h1>
        <p class="lead">模型将从这里下载。国内网络环境建议使用镜像源；模型文件较大，也可指定其他磁盘目录，之后都能在设置页修改。</p>
        <div class="source-grid">
          <div class="source-card card" :class="{ active: selectedSource === 'official' }" @click="selectedSource = 'official'">
            <strong>Hugging Face 官方</strong>
            <span class="mono">https://huggingface.co</span>
            <span class="source-hint">默认源，境外网络环境</span>
          </div>
          <div class="source-card card" :class="{ active: selectedSource === 'mirror' }" @click="selectedSource = 'mirror'">
            <strong>国内镜像</strong>
            <span class="mono">https://hf-mirror.com</span>
            <span class="source-hint">国内高速，推荐</span>
          </div>
          <div class="source-card card" :class="{ active: selectedSource === 'custom' }" @click="selectedSource = 'custom'">
            <strong>自定义</strong>
            <span class="mono">输入镜像源地址</span>
            <input
              v-if="selectedSource === 'custom'"
              type="text"
              v-model="customEndpoint"
              placeholder="https://..."
              @click.stop
            />
          </div>
        </div>
        <div class="dir-row card">
          <label class="field">
            模型下载目录（留空使用默认位置）
            <input type="text" v-model="modelRootInput" placeholder="例如 D:/ai-models（留空 = 数据目录 models）" />
          </label>
          <button class="btn small ghost" @click="modelRootInput = ''">恢复默认</button>
        </div>
        <div class="actions">
          <button class="btn primary" @click="applySource(true)">下一步</button>
        </div>
      </template>

      <template v-else-if="step === 3">
        <h1>选择模型</h1>
        <p class="lead">建议从推荐模型开始。模型下载后即可启动服务进行对话。</p>
        <div class="catalog">
          <div v-for="model in snapshot?.catalog ?? []" :key="model.id" class="model-card card" :class="{ strong: model.recommended }">
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
              <button
                v-if="installedIds.has(model.id)"
                class="btn small"
                disabled
              >已安装</button>
              <button
                v-else
                class="btn small primary"
                :disabled="downloading"
                @click="startDownload(model.id)"
              >
                {{ downloading && store.download.activeId === model.id ? "下载中..." : "下载" }}
              </button>
            </div>
          </div>
        </div>
        <div v-if="downloading" class="download-card card">
          <div class="dl-row">
            <span class="mono dl-file">{{ store.download.file || "准备中..." }}</span>
            <span class="dl-pct mono">{{ progressText }}</span>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{ width: (progressPercent ?? 10) + '%' }"></div>
          </div>
          <div class="dl-actions">
            <button class="btn small ghost" @click="cancelDownload">取消下载</button>
          </div>
        </div>
        <div v-if="error" class="error">{{ error }}</div>
        <div class="actions">
          <button class="btn ghost" @click="step = 4">跳过（以后可在模型库下载）</button>
          <button
            v-if="installedIds.size > 0 && !downloading"
            class="btn primary"
            @click="step = 4"
          >下一步</button>
        </div>
      </template>

      <template v-else>
        <h1>完成了！</h1>
        <p class="lead">
          运行环境与模型都已就绪。启动模型服务后即可进入聊天；也可以稍后在应用内再启动。
        </p>
        <div v-if="store.lastModelAlias" class="ok-box">模型「{{ store.lastModelAlias }}」已安装</div>
        <div v-if="error" class="error">{{ error }}</div>
        <div class="actions">
          <button
            v-if="store.lastModelAlias"
            class="btn primary"
            :disabled="store.serviceBusy"
            @click="startService(store.lastModelAlias)"
          >{{ store.serviceBusy ? "启动中..." : "启动模型服务" }}</button>
          <button class="btn" @click="finish">进入应用</button>
        </div>
      </template>
    </main>
  </div>
</template>

<style scoped>
.onboarding {
  height: 100%;
  display: flex;
}
.side {
  width: 300px;
  flex: none;
  padding: 40px 28px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
}
.brand {
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 40px;
}
.brand-dot {
  color: var(--accent);
}
.steps {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 22px;
}
.steps li {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  opacity: 0.5;
  transition: opacity 0.2s ease;
}
.steps li.active,
.steps li.done {
  opacity: 1;
}
.num {
  width: 24px;
  height: 24px;
  flex: none;
  border-radius: 50%;
  border: 1px solid var(--border-2);
  background: var(--surface-2);
  color: var(--text-3);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  margin-top: 2px;
}
.steps li.active .num {
  color: #14171c;
  background: var(--accent);
  border-color: transparent;
}
.s-title {
  font-size: 14px;
  font-weight: 600;
}
.s-sub {
  font-size: 12px;
  color: var(--text-3);
}
.side-note {
  margin-top: auto;
  font-size: 12px;
  color: var(--text-4);
}
.panel {
  flex: 1;
  padding: 64px 72px;
  overflow-y: auto;
  max-width: 860px;
  margin: 0 auto;
}
h1 {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.02em;
  margin-bottom: 14px;
}
.lead {
  color: var(--text-2);
  font-size: 14.5px;
  line-height: 1.8;
  margin-bottom: 28px;
  max-width: 560px;
}
.hardware {
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 28px;
  max-width: 560px;
}
.hw-row {
  display: flex;
  gap: 12px;
  font-size: 13.5px;
}
.hw-label {
  width: 80px;
  flex: none;
  color: var(--text-3);
}
.error {
  background: rgba(224, 108, 117, 0.12);
  border: 1px solid rgba(224, 108, 117, 0.4);
  color: var(--danger);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  max-width: 560px;
  margin-bottom: 20px;
  word-break: break-all;
}
.ok-box {
  background: rgba(60, 185, 143, 0.12);
  border: 1px solid rgba(60, 185, 143, 0.4);
  color: var(--success);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13.5px;
  margin-bottom: 20px;
  max-width: 560px;
}
.steps-list {
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 16px;
  max-width: 560px;
}
.steps-list li {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13.5px;
  color: var(--text-2);
}
.steps-list li.failed {
  color: var(--danger);
}
.step-n {
  margin-left: auto;
  color: var(--text-4);
  font-size: 12px;
}
.log-box {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 11.5px;
  line-height: 1.7;
  color: var(--text-3);
  max-width: 560px;
  height: 180px;
  overflow-y: auto;
  margin-bottom: 20px;
}
.actions {
  display: flex;
  gap: 12px;
  margin-top: 12px;
}
.source-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
  max-width: 700px;
  margin-bottom: 24px;
}
.source-card {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}
.source-card.active {
  border-color: rgba(88, 166, 240, 0.7);
  background: var(--surface-2);
}
.source-card input {
  font-size: 12.5px;
  padding: 6px 9px;
}
.source-hint {
  color: var(--text-3);
  font-size: 12px;
}
.dir-row {
  max-width: 700px;
  padding: 14px 16px;
  margin-bottom: 8px;
  display: flex;
  align-items: flex-end;
  gap: 12px;
}
.dir-row .field {
  flex: 1;
}
.catalog {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 14px;
  max-width: 720px;
  margin-bottom: 20px;
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
  margin-top: 6px;
}
.download-card {
  padding: 14px 16px;
  max-width: 560px;
  margin-bottom: 20px;
}
.dl-actions {
  margin-top: 10px;
  display: flex;
  justify-content: flex-end;
}
.dl-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  font-size: 12.5px;
  color: var(--text-2);
}
.dl-file {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
