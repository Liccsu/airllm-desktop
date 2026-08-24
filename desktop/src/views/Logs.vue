<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { refreshState, store } from "../store";

const listEl = ref<HTMLElement | null>(null);

const lines = computed(() => store.serviceLogs);

// serviceLogs 是响应式数组；push 不改变引用，必须以 length 作为 watch 依赖，
// 否则 computed 返回同一引用，watch 永不再触发，日志不自动滚动。
watch(() => store.serviceLogs.length, () => void scrollToBottom());

async function scrollToBottom() {
  await nextTick();
  if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight;
}

function clear() {
  store.serviceLogs = [];
}

async function refresh() {
  await refreshState();
}

function levelClass(text: string): string {
  if (/error|failed|traceback|错误|失败/i.test(text)) return "err";
  if (/warn|warning/i.test(text)) return "warn";
  return "";
}
</script>

<template>
  <div class="logs">
    <div class="logs-head">
      <h2>引擎日志</h2>
      <div class="actions">
        <button class="btn small ghost" @click="refresh">刷新状态</button>
        <button class="btn small" @click="clear">清空</button>
      </div>
    </div>
    <div ref="listEl" class="log-list mono">
      <div v-if="lines.length === 0" class="empty">暂无日志。启动模型服务后，引擎输出会实时显示在这里。</div>
      <div v-for="(line, i) in lines" :key="i" class="log-line" :class="levelClass(line.text)">
        <span class="ts">{{ line.ts }}</span>
        <span class="text">{{ line.text }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.logs {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 26px 28px 20px;
}
.logs-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}
h2 {
  font-size: 20px;
}
.actions {
  display: flex;
  gap: 10px;
}
.log-list {
  flex: 1;
  overflow-y: auto;
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 10px;
  padding: 14px 16px;
  font-size: 12px;
  line-height: 1.8;
}
.empty {
  color: var(--text-4);
  text-align: center;
  padding: 40px 0;
}
.log-line {
  display: flex;
  gap: 12px;
  color: var(--text-2);
  word-break: break-all;
  user-select: text;
}
.log-line .ts {
  flex: none;
  color: var(--text-4);
  font-size: 11px;
  padding-top: 2px;
}
.log-line .text {
  min-width: 0;
  white-space: pre-wrap;
}
.log-line.err .text {
  color: var(--danger);
}
.log-line.warn .text {
  color: var(--warn);
}
</style>
