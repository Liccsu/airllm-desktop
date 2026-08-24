<script setup lang="ts">
import { computed, reactive, ref, watch } from "vue";
import { getVersion } from "@tauri-apps/api/app";
import { check } from "@tauri-apps/plugin-updater";
import { api } from "../api";
import { defaultSettings, refreshState, store } from "../store";

const form = reactive({
  port: 8000,
  device: "cuda:0",
  maxSeqLen: 512,
  maxOutputTokens: 128,
  preload: true,
  endpoint: "",
  modelRoot: "",
  downloadWorkers: 8,
});
const saved = ref(false);
const error = ref("");
const envInstalling = ref(false);
const envLines = ref<string[]>([]);
const keyCopied = ref(false);
const appVersion = ref("");
const updateState = ref("");
const hfToken = ref(store.hfToken);



async function checkUpdate() {
  updateState.value = "正在检查更新...";
  try {
    const update = await check();
    if (!update) {
      updateState.value = "已是最新版本";
      return;
    }
    updateState.value = `发现新版本 ${update.version}，正在下载并安装...`;
    await update.downloadAndInstall();
    updateState.value = "更新已安装，请重启应用生效";
  } catch (e) {
    updateState.value = `检查更新失败：${String(e)}`;
  }
}

void getVersion().then((v) => (appVersion.value = v)).catch(() => undefined);

const snapshot = computed(() => store.snapshot);
const apiKey = computed(() => snapshot.value?.apiKey ?? "");
const env = computed(() => snapshot.value?.env);

function syncForm() {
  const s = store.settings ?? defaultSettings();
  form.port = s.port;
  form.device = s.device;
  form.maxSeqLen = s.maxSeqLen;
  form.maxOutputTokens = s.maxOutputTokens;
  form.preload = s.preload;
  form.endpoint = s.endpoint ?? "";
  form.modelRoot = s.modelRoot ?? "";
  form.downloadWorkers = s.downloadWorkers ?? 8;
}
void syncForm();
// 设置由 bootstrap 异步从 Rust 恢复；仅首次就绪时同步表单（后续不再覆盖，避免清空输入）。
let syncedOnce = false;
watch(
  () => store.settings,
  (s) => {
    if (s && !syncedOnce) {
      syncedOnce = true;
      syncForm();
    }
  },
);

async function save() {
  error.value = "";
  // Token 随设置一并保存（仅存本机 localStorage，不落盘到配置文件）。
  store.hfToken = hfToken.value.trim();
  try {
    localStorage.setItem("airllm.hfToken", store.hfToken);
  } catch {
    // localStorage 不可用时仅会话内生效
  }
  if (form.port < 1 || form.port > 65535) {
    error.value = "端口必须在 1-65535 之间";
    return;
  }
  const settings = {
    port: form.port,
    device: form.device.trim() || "cuda:0",
    maxSeqLen: Math.max(128, form.maxSeqLen),
    maxOutputTokens: Math.max(8, form.maxOutputTokens),
    preload: form.preload,
    endpoint: form.endpoint.trim(),
    modelRoot: form.modelRoot.trim(),
    downloadWorkers: Math.max(1, Math.min(32, form.downloadWorkers)),
  };
  try {
    await api.updateSettings(settings);
    store.settings = settings;
    store.sourceEndpoint = settings.endpoint;
    try {
      localStorage.setItem("airllm.endpoint", settings.endpoint);
    } catch {
      // localStorage 不可用时忽略
    }
    saved.value = true;
    setTimeout(() => (saved.value = false), 2000);
    await refreshState();
  } catch (e) {
    error.value = String(e);
  }
}

async function installEnv() {
  error.value = "";
  envInstalling.value = true;
  envLines.value = [];
  store.envInstall.lines = [];
  try {
    await api.installEnv();
    await refreshState();
  } catch (e) {
    error.value = String(e);
  } finally {
    envInstalling.value = false;
  }
}

async function copyKey() {
  try {
    await navigator.clipboard.writeText(apiKey.value);
    keyCopied.value = true;
    setTimeout(() => (keyCopied.value = false), 2000);
  } catch (e) {
    error.value = String(e);
  }
}

async function open(target: string) {
  try {
    await api.openPath(target);
  } catch (e) {
    error.value = String(e);
  }
}
</script>

<template>
  <div class="settings">
    <h2>设置</h2>

    <div class="section card">
      <div class="section-title">模型服务</div>
      <div class="form-grid">
        <label class="field">
          端口
          <input type="number" v-model.number="form.port" min="1" max="65535" />
        </label>
        <label class="field">
          推理设备
          <input type="text" v-model="form.device" placeholder="cuda:0" />
        </label>
        <label class="field">
          最大输入长度（token）
          <input type="number" v-model.number="form.maxSeqLen" min="128" />
        </label>
        <label class="field">
          默认输出长度（token）
          <input type="number" v-model.number="form.maxOutputTokens" min="8" />
        </label>
        <label class="field">
          模型下载源（镜像）
          <input type="text" v-model="form.endpoint" placeholder="留空 = huggingface.co；国内填 https://hf-mirror.com" />
        </label>
        <label class="field">
          模型下载目录
          <input type="text" v-model="form.modelRoot" placeholder="留空 = 数据目录 models\；可填比如 D:/models" />
        </label>
        <label class="field">
          下载线程数（并行）
          <input type="number" v-model.number="form.downloadWorkers" min="1" max="32" />
        </label>
        <label class="field">
          Hugging Face Access Token（下载私有/受限模型时需要）
          <input type="password" v-model="hfToken" placeholder="hf_...（留空表示公开模型无需登录）" />
        </label>
        <label class="field check">
          <input type="checkbox" v-model="form.preload" />
          启动时预加载模型
        </label>
      </div>
      <div class="save-row">
        <button class="btn primary small" @click="save">保存设置</button>
        <span v-if="saved" class="saved-hint">✓ 已保存</span>
        <span v-if="error" class="error-hint">{{ error }}</span>
      </div>
    </div>

    <div class="section card">
      <div class="section-title">API 访问</div>
      <p class="hint">本地服务的访问密钥，外部兼容客户端（如 Claude Code / Codex）可用它连接。</p>
      <div class="key-row">
        <code class="key mono">{{ apiKey }}</code>
        <button class="btn small" @click="copyKey">{{ keyCopied ? "✓ 已复制" : "复制" }}</button>
      </div>
      <p class="hint mono">http://127.0.0.1:{{ form.port }}（OpenAI Responses 兼容）</p>
    </div>

    <div class="section card">
      <div class="section-title">数据与环境</div>
      <div class="env-row">
        <span class="pill">
          <span class="dot" :class="env?.depsReady ? 'ok' : 'bad'"></span>
          {{ env?.depsReady ? "运行环境就绪" : "环境未安装" }}
        </span>
        <span class="pill">
          <span class="dot" :class="env?.pythonOk ? 'ok' : 'idle'"></span>
          {{ env?.pythonOk ? "Python 就绪" : "Python 未安装" }}
        </span>
        <button class="btn small" :disabled="envInstalling" @click="installEnv">
          {{ envInstalling ? "安装中..." : (env?.depsReady ? "重新检查" : "安装环境") }}
        </button>
      </div>
      <div class="data-row">
        <span class="mono path">{{ snapshot?.dataDir }}</span>
        <button class="btn small ghost" @click="open('data')">打开目录</button>
        <button class="btn small ghost" @click="open('models')">模型目录</button>
      </div>
    </div>

    <div class="section card">
      <div class="section-title">应用更新</div>
      <div class="env-row">
        <span class="pill">当前版本 v{{ appVersion || "?" }}</span>
        <button class="btn small" @click="checkUpdate" :disabled="updateState.includes('正在')">检查更新</button>
      </div>
      <p v-if="updateState" class="hint">{{ updateState }}</p>
      <p class="hint">更新发布后可从 GitHub Releases 自动下载安装；引擎与推理库随安装包自动升级。</p>
    </div>

    <div class="section card">
      <div class="section-title">环境日志</div>
      <div v-if="envLines.length" class="env-log mono">
        <div v-for="(line, i) in envLines.slice(-300)" :key="i">{{ line }}</div>
      </div>
      <p v-else class="hint">安装环境时的输出将显示在这里。</p>
    </div>
  </div>
</template>

<style scoped>
.settings {
  height: 100%;
  overflow-y: auto;
  padding: 26px 28px 40px;
  max-width: 860px;
}
h2 {
  font-size: 20px;
  margin-bottom: 20px;
}
.section {
  padding: 18px 20px;
  margin-bottom: 16px;
}
.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 14px;
}
.field.check {
  flex-direction: row;
  align-items: center;
  gap: 8px;
  font-size: 13.5px;
  color: var(--text-2);
  align-self: end;
  padding: 9px 12px;
}
.save-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
}
.saved-hint {
  color: var(--success);
  font-size: 13px;
}
.error-hint {
  color: var(--danger);
  font-size: 13px;
}
.hint {
  color: var(--text-3);
  font-size: 12.5px;
  margin-bottom: 12px;
}
.key-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.key {
  background: var(--bg);
  border: 1px solid var(--border-2);
  border-radius: 8px;
  padding: 8px 12px;
  font-size: 12.5px;
  color: var(--text-2);
  user-select: text;
  word-break: break-all;
}
.env-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}
.data-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.path {
  font-size: 12px;
  color: var(--text-3);
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 5px 10px;
  user-select: text;
}
.env-log {
  background: var(--bg);
  border: 1px solid var(--border-2);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 11px;
  line-height: 1.7;
  color: var(--text-3);
  max-height: 200px;
  overflow-y: auto;
}
</style>
