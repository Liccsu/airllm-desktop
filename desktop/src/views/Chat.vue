<script setup lang="ts">
import { computed, nextTick, ref } from "vue";
import { 
 } from "../api";
import { refreshState, store } from "../store";

interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}

let nextId = 1;
const messages = ref<Message[]>([]);
const input = ref("");
const sending = ref(false);
const error = ref("");
const scrollEl = ref<HTMLElement | null>(null);

const service = computed(() => store.snapshot?.service);
const modelName = computed(
  () => service.value?.model ?? store.lastModelAlias ?? "",
);
const canSend = computed(
  () => modelName.value !== "" && !sending.value && service.value?.ready !== false,
);

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderMarkdown(text: string): string {
  const blocks: string[] = [];
  const prepared = text.replace(/```[\s\S]*?```/g, (match) => {
    blocks.push(match);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  let html = escapeHtml(prepared);
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, '<code class="inline">$1</code>');
  html = html.replace(/\n/g, "<br>");
  html = html.replace(/\u0000(\d+)\u0000/g, (_whole, index) => {
    const block = blocks[Number(index)];
    const code = block.replace(/^```[^\n]*\n?/, "").replace(/```$/, "\n").replace(/\n$/, "");
    return `<pre class="codeblock"><code>${escapeHtml(code)}</code></pre>`;
  });
  return html;
}

async function scrollToBottom() {
  await nextTick();
  if (scrollEl.value) scrollEl.value.scrollTop = scrollEl.value.scrollHeight;
}

function addMessage(role: Message["role"], content: string): Message {
  const message: Message = { id: nextId++, role, content };
  messages.value.push(message);
  void scrollToBottom();
  return message;
}

async function send() {
  const text = input.value.trim();
  if (!text || !canSend.value) return;
  input.value = "";
  error.value = "";
  addMessage("user", text);
  const assistant = addMessage("assistant", "");
  assistant.streaming = true;
  sending.value = true;

  const controller = new AbortController();
  abortController = controller;
  try {
    const history = messages.value
      .filter((m) => m.id < assistant.id);
    const payload = {
      model: modelName.value,
      input: history.map((m) => ({ role: m.role, content: m.content })),
      stream: true,
    };
    const response = await fetch(`http://127.0.0.1:${service.value?.port ?? 8000}/v1/responses`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${store.snapshot?.apiKey ?? ""}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) {
      const body = await response.text().catch(() => "");
      throw new Error(body || `HTTP ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let completed = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLines: string[] = [];
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) dataLines.push(line.slice(6));
        }
        for (const data of dataLines) {
          let parsed: any;
          try {
            parsed = JSON.parse(data);
          } catch {
            continue;
          }
          if (parsed.type === "response.output_text.delta" && typeof parsed.delta === "string") {
            assistant.content += parsed.delta;
            void scrollToBottom();
          } else if (parsed.type === "response.completed") {
            completed = true;
          } else if (parsed.type === "error") {
            throw new Error(parsed.error?.message ?? "生成失败");
          }
        }
      }
    }
    if (!completed && !assistant.content) {
      throw new Error("服务未返回内容");
    }
  } catch (e: any) {
    if (e?.name !== "AbortError") {
      error.value = String(e?.message ?? e);
    }
  } finally {
    assistant.streaming = false;
    sending.value = false;
    if (!assistant.content) {
      messages.value = messages.value.filter((m) => m.id !== assistant.id);
    }
    await refreshState();
  }
}

function stop() {
  sending.value = false;
  abortController?.abort();
}

let abortController: AbortController | null = null;

async function onKeydown(event: KeyboardEvent) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    await send();
  }
}

async function usePrompt(p: string) {
  input.value = p;
}
</script>

<template>
  <div class="chat">
    <div v-if="!service?.ready" class="not-ready card">
      <div class="not-title">模型服务未就绪</div>
      <div class="not-sub">
        {{ service?.running ? "模型正在加载（首次加载需要拆分权重，请稍候）" : "服务未启动" }}
      </div>
      <button class="btn small ghost" @click="store.view = 'library'">前往模型库</button>
    </div>
    <div ref="scrollEl" class="messages">
      <div v-if="messages.length === 0" class="empty-chat">
        <div class="chat-title">和 {{ modelName || "AI" }} 聊天</div>
        <div class="suggestions">
          <button class="btn small" @click="usePrompt('用一句话解释什么是大语言模型')">什么是大语言模型？</button>
          <button class="btn small" @click="usePrompt('写一首关于雪的小诗')">写一首关于雪的小诗</button>
          <button class="btn small" @click="usePrompt('用 Python 写一个斐波那契数列函数')">Python 示例代码</button>
        </div>
      </div>
      <div v-for="message in messages" :key="message.id" class="msg" :class="message.role">
        <div class="avatar">{{ message.role === "user" ? "你" : "AI" }}</div>
        <div class="bubble">
          <div
            v-if="message.content && message.role === 'assistant'"
            class="md"
            v-html="renderMarkdown(message.content)"
          ></div>
          <div v-else-if="message.content" class="plain">{{ message.content }}</div>
          <div v-else class="typing" :class="{ active: message.streaming }">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>
    </div>
    <div v-if="error" class="error-bar">{{ error }}</div>
    <div class="composer">
      <textarea
        v-model="input"
        rows="1"
        placeholder="输入消息，Enter 发送，Shift+Enter 换行"
        :disabled="!canSend"
        @keydown="onKeydown"
      ></textarea>
      <div class="composer-actions">
        <button v-if="sending" class="btn small" @click="stop">停止</button>
        <button class="btn small primary" :disabled="!canSend" @click="send">发送</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.chat {
  height: 100%;
  display: flex;
  flex-direction: column;
  position: relative;
}
.not-ready {
  margin: 16px 20px 0;
  padding: 14px 16px;
  display: flex;
  align-items: center;
  gap: 14px;
}
.not-title {
  font-weight: 600;
}
.not-sub {
  color: var(--text-3);
  font-size: 13px;
  flex: 1;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 28px 24px 12px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.empty-chat {
  margin: auto;
  text-align: center;
  padding-bottom: 40px;
}
.chat-title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 18px;
}
.suggestions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
}
.msg {
  display: flex;
  gap: 12px;
  max-width: 820px;
}
.msg.user {
  align-self: flex-end;
  flex-direction: row-reverse;
}
.avatar {
  width: 34px;
  height: 34px;
  flex: none;
  border-radius: 9px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: 700;
  background: var(--surface-3);
  color: var(--text-2);
}
.msg.user .avatar {
  background: var(--accent);
  color: #14171c;
}
.bubble {
  background: var(--surface);
  border: 1px solid var(--border-2);
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 14px;
  line-height: 1.75;
  min-width: 0;
  flex: 1;
}
.msg.user .bubble {
  background: var(--surface-2);
}
.md :deep(p) {
  margin: 0;
}
.plain {
  white-space: pre-wrap;
  word-break: break-word;
}
.md {
  word-break: break-word;
}
.md :deep(.codeblock) {
  background: var(--bg);
  border: 1px solid var(--border-2);
  border-radius: 8px;
  padding: 12px 14px;
  margin: 8px 0;
  overflow-x: auto;
  font-family: var(--mono);
  font-size: 12.5px;
  line-height: 1.6;
  user-select: text;
}
.md :deep(.codeblock code) {
  font-family: var(--mono);
}
.md :deep(.inline) {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.5px 6px;
  font-family: var(--mono);
  font-size: 12.5px;
}
.typing {
  display: inline-flex;
  gap: 4px;
  padding: 4px 0;
}
.typing span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-3);
  animation: blink 1.2s infinite;
}
.typing span:nth-child(2) {
  animation-delay: 0.2s;
}
.typing span:nth-child(3) {
  animation-delay: 0.4s;
}
@keyframes blink {
  0%, 80%, 100% { opacity: 0.25; }
  40% { opacity: 1; }
}
.error-bar {
  margin: 0 24px 8px;
  background: rgba(224, 108, 117, 0.1);
  border: 1px solid rgba(224, 108, 117, 0.35);
  color: var(--danger);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: 12.5px;
}
.composer {
  flex: none;
  padding: 12px 24px 18px;
}
.composer textarea {
  width: 100%;
  resize: none;
  min-height: 56px;
  max-height: 180px;
  line-height: 1.6;
}
.composer-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: 10px;
}
</style>
