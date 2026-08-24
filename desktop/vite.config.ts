import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// Tauri 期望固定端口；clearScreen 让 Rust 输出可见。
export default defineConfig({
  plugins: [vue()],
  clearScreen: false,
  server: {
        host: "127.0.0.1",
port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
});
