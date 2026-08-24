//! 引擎管理：环境安装、模型下载与 Responses 服务进程生命周期。

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

// ── 常量 ────────────────────────────────────────────────────────────────

const PYTHON_VERSION: &str = "3.11";
const TORCH_INDEX_URL: &str = "https://download.pytorch.org/whl/cu128";
const BASE_DEPENDENCIES: &[&str] = &[
    "fastapi>=0.115,<1",
    "uvicorn>=0.34,<1",
    "pydantic>=2,<3",
    "transformers>=4.49,<5.13",
    "accelerate>=1.0",
    "safetensors",
    "huggingface-hub",
    "scipy",
    "sentencepiece",
    "tqdm",
    "httpx",
];

// ── 数据结构 ────────────────────────────────────────────────────────────

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub used_mb: u64,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub size_bytes: u64,
    pub vram_gb: u64,
    pub license: String,
    pub recommended: bool,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct InstalledModel {
    pub alias: String,
    pub model_id: String,
    pub revision: String,
    pub model_dir: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct ServiceStatus {
    pub running: bool,
    pub ready: bool,
    pub port: u16,
    pub model: Option<String>,
    pub device: String,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct EnvStatus {
    pub python_ok: bool,
    pub deps_ready: bool,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct AppSnapshot {
    pub data_dir: String,
    pub catalog: Vec<CatalogEntry>,
    pub installed: Vec<InstalledModel>,
    pub env: EnvStatus,
    pub service: ServiceStatus,
    pub gpu: Option<GpuInfo>,
    pub memory: MemoryInfo,
    pub api_key: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct ServiceSettings {
    pub port: u16,
    pub device: String,
    pub max_seq_len: u32,
    pub max_output_tokens: u32,
    pub preload: bool,
    pub endpoint: String,
    pub model_root: String,
    pub download_workers: u16,
}

impl Default for ServiceSettings {
    fn default() -> Self {
        Self {
            port: 8000,
            device: "cuda:0".into(),
            max_seq_len: 512,
            max_output_tokens: 128,
            preload: true,
            endpoint: "".into(),
            model_root: "".into(),
            download_workers: 8,
        }
    }
}

// ── 引擎状态 ────────────────────────────────────────────────────────────

pub struct EngineState {
    pub data_dir: PathBuf,
    pub resources: PathBuf,
    pub uv_exe: PathBuf,
    env_checked: Mutex<Option<EnvStatus>>,
    service: Mutex<Option<ServiceRecord>>,
    installing: Mutex<bool>,
    settings: Mutex<ServiceSettings>,
}

struct ServiceRecord {
    child: Child,
}

impl EngineState {
    fn new(app: &AppHandle) -> Self {
        let base = std::env::var("LOCALAPPDATA")
            .ok()
            .map(PathBuf::from)
            .or_else(|| app.path().app_data_dir().ok());
        let data_dir = base.unwrap_or_else(|| PathBuf::from(".")).join("AirLLM");
        let resources = app
            .path()
            .resource_dir()
            .ok()
            .map(|dir| dir.join("resources"))
            .unwrap_or_else(|| PathBuf::from("resources"));
        let uv_exe = data_dir.join("bin").join("uv.exe");
        Self {
            data_dir,
            resources,
            uv_exe,
            env_checked: Mutex::new(None),
            service: Mutex::new(None),
            installing: Mutex::new(false),
            settings: Mutex::new(ServiceSettings::default()),
        }
    }

    fn venv_python_dir(&self) -> PathBuf {
        self.data_dir.join("venv").join("Scripts").join("python.exe")
    }

    fn python(&self) -> Option<PathBuf> {
        let path = self.venv_python_dir();
        path.is_file().then_some(path)
    }

    fn api_key_path(&self) -> PathBuf {
        self.data_dir.join("api_key.txt")
    }

    fn app_toml_path(&self) -> PathBuf {
        self.data_dir.join("app.toml")
    }

    fn catalog_path(&self) -> PathBuf {
        self.resources.join("catalog.json")
    }

    /// 捆绑引擎 wheel 的指纹；用于检测安装包中的引擎是否比已装版本更新。
    fn engine_fingerprint(&self) -> String {
        engine_fingerprint_for(&self.resources)
    }

    fn engine_fingerprint_path(&self) -> PathBuf {
        self.data_dir.join("engine-wheel.sha256")
    }

    fn needs_engine_update(&self) -> bool {
        let current = self.engine_fingerprint();
        if current.is_empty() {
            return false;
        }
        match std::fs::read_to_string(self.engine_fingerprint_path()) {
            // 首次启动（无记录）由引导流程负责安装，不重复触发；已有记录且变化才升级。
            Ok(recorded) => recorded.trim() != current,
            Err(_) => false,
        }
    }

    fn record_engine_fingerprint(&self) {
        let current = self.engine_fingerprint();
        if !current.is_empty() {
            let _ = std::fs::write(self.engine_fingerprint_path(), format!("{current}\n"));
        }
    }
}

// ── 事件推送 ────────────────────────────────────────────────────────────

fn emit_event(app: &AppHandle, event: &str, payload: serde_json::Value) {
    let _ = app.emit(event, payload);
}

// ── 子进程工具 ──────────────────────────────────────────────────────────

/// Windows 下隐藏子进程控制台：GUI 应用无控制台，spawn 默认会新建黑色 CMD 窗口。
#[cfg(windows)]
fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn hide_console(_command: &mut Command) {}

/// 同步运行命令，逐行回调输出，返回退出码。
fn run_process<F>(
    program: &str,
    args: &[String],
    cwd: Option<&Path>,
    env: &[(&str, &str)],
    on_line: F,
) -> std::io::Result<i32>
where
    F: Fn(String) + Send + Sync + 'static,
{
    let on_line = std::sync::Arc::new(on_line);
    let mut command = Command::new(program);
    hide_console(&mut command);
    command.args(args);
    if let Some(dir) = cwd {
        command.current_dir(dir);
    }
    for (key, value) in env {
        command.env(key, value);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child.stdout.take().expect("stdout 管道");
    let stderr = child.stderr.take().expect("stderr 管道");
    let on_line_out = on_line.clone();
    let out_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            on_line_out(line);
        }
    });
    let on_line_err = on_line.clone();
    let err_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            on_line_err(line);
        }
    });
    let status = child.wait()?;
    let _ = out_thread.join();
    let _ = err_thread.join();
    Ok(status.code().unwrap_or(-1))
}

/// 以事件流方式运行引擎 CLI 子进程：stdout 中可解析的 JSON 转为
/// ``<prefix>`` 事件，其余行与 stderr 转为 ``<prefix>-line`` 原始日志。
fn run_engine_cli(app: &AppHandle, python: &Path, args: &[String], prefix: &str) -> std::io::Result<i32> {
    let mut command = Command::new(python);
    hide_console(&mut command);
    command.args(args);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;

    let handle = app.clone();
    let prefix_owned = prefix.to_string();
    let stdout = child.stdout.take().expect("stdout 管道");
    let out_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                if value.get("event").is_some() {
                    emit_event(&handle, &prefix_owned, value.clone());
                    continue;
                }
            }
            emit_event(&handle, &format!("{}-line", prefix_owned), serde_json::json!({ "text": line }));
        }
    });
    let handle2 = app.clone();
    let prefix2 = prefix.to_string();
    let stderr = child.stderr.take().expect("stderr 管道");
    let err_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            emit_event(&handle2, &format!("{}-line", prefix2), serde_json::json!({ "text": line }));
        }
    });
    let status = child.wait()?;
    let _ = out_thread.join();
    let _ = err_thread.join();
    Ok(status.code().unwrap_or(-1))
}

// ── 系统信息 ────────────────────────────────────────────────────────────

fn detect_gpu() -> Option<GpuInfo> {
    let mut command = Command::new("nvidia-smi");
    hide_console(&mut command);
    let output = command
        .args(["--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?;
    let mut parts = line.split(',').map(str::trim);
    let name = parts.next()?.to_string();
    let total: u64 = parts.next()?.parse().ok()?;
    let used: u64 = parts.next()?.parse().ok()?;
    Some(GpuInfo { name, vram_total_mb: total, vram_used_mb: used })
}

// ── Tauri 命令 ─────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_state(state: State<'_, EngineState>) -> Result<AppSnapshot, String> {
    let catalog = read_catalog(state.catalog_path());
    let memory = detect_memory();
    let env = {
        let mut guard = state.env_checked.lock();
        if let Some(status) = guard.clone() {
            status
        } else {
            let checked = check_env_inner(&state);
            *guard = Some(checked.clone());
            checked
        }
    };
    let installed = list_installed(&state.data_dir);
    let service = detect_service(&state);
    let gpu = detect_gpu();
    let api_key = load_or_create_api_key(&state);
    Ok(AppSnapshot {
        data_dir: state.data_dir.to_string_lossy().into_owned(),
        catalog,
        installed,
        env,
        service,
        gpu,
        memory,
        api_key,
    })
}

fn read_catalog(path: PathBuf) -> Vec<CatalogEntry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let Some(models) = value.get("models").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    models
        .iter()
        .filter_map(|entry| {
            let id = entry.get("id")?.as_str()?;
            let name = entry.get("name").and_then(|v| v.as_str()).unwrap_or(id);
            Some(CatalogEntry {
                id: id.to_string(),
                name: name.to_string(),
                description: entry.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                size_bytes: entry.get("size_bytes").and_then(|v| v.as_u64()).unwrap_or(0),
                vram_gb: entry.get("vram_gb").and_then(|v| v.as_u64()).unwrap_or(0),
                license: entry.get("license").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                recommended: entry.get("recommended").and_then(|v| v.as_bool()).unwrap_or(false),
            })
        })
        .collect()
}

fn list_installed(data_dir: &Path) -> Vec<InstalledModel> {
    let manifests_dir = data_dir.join("manifests");
    let Ok(entries) = std::fs::read_dir(&manifests_dir) else {
        return Vec::new();
    };
    let mut result = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        let model_dir = value
            .get("model_dir")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if !Path::new(&model_dir).is_dir() {
            continue;
        }
        result.push(InstalledModel {
            alias: path.file_stem().and_then(|s| s.to_str()).unwrap_or_default().to_string(),
            model_id: value.get("model_id").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            revision: value.get("revision").and_then(|v| v.as_str()).unwrap_or_default().to_string(),
            model_dir,
        });
    }
    result
}

fn detect_service(state: &EngineState) -> ServiceStatus {
    let settings = state.settings.lock().clone();
    let health = http_get(settings.port, "/healthz");
    let body = health
        .as_deref()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(text).ok());
    let ready = body
        .as_ref()
        .and_then(|v| v.get("ready").and_then(|r| r.as_bool()))
        .unwrap_or(false);
    let model = body
        .as_ref()
        .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(str::to_string));
    ServiceStatus { running: body.is_some(), ready, port: settings.port, model, device: settings.device }
}

/// 极简 HTTP GET（仅 /healthz 等小响应，本地服务使用）。
/// 系统物理内存占用(MB)。
fn detect_memory() -> MemoryInfo {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};
    let system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    let total = system.total_memory(); // bytes
    let used = system.used_memory();
    MemoryInfo {
        total_mb: total / 1024 / 1024,
        used_mb: used / 1024 / 1024,
    }
}

fn http_get(port: u16, path: &str) -> Option<String> {
    let mut stream = std::net::TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(2))).ok()?;
    let request = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).ok()?;
    let mut buffer = Vec::new();
    stream.read_to_end(&mut buffer).ok()?;
    let text = String::from_utf8_lossy(&buffer).into_owned();
    text.split("\r\n\r\n").nth(1).map(str::to_string)
}

// ── 环境安装 ────────────────────────────────────────────────────────────

fn check_env_inner(state: &EngineState) -> EnvStatus {
    let Some(python) = state.python() else {
        return EnvStatus { python_ok: false, deps_ready: false };
    };
    let mut command = Command::new(python);
    hide_console(&mut command);
    let ok = command
        .args(["-c", "import torch, fastapi, transformers, airllm, airllm_responses"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    EnvStatus { python_ok: true, deps_ready: ok }
}

#[tauri::command]
pub async fn install_env(app: AppHandle, state: State<'_, EngineState>) -> Result<bool, String> {
    {
        let mut installing = state.installing.lock();
        if *installing {
            return Err("环境已在安装中".into());
        }
        *installing = true;
    }
    let data_dir = state.data_dir.clone();
    let resources = state.resources.clone();
    let uv_exe = state.uv_exe.clone();
    let venv_python = state.venv_python_dir();
    let result = tauri::async_runtime::spawn_blocking(move || {
        install_env_inner(&app, data_dir, resources, uv_exe, venv_python)
    })
    .await;
    {
        let mut installing = state.installing.lock();
        *installing = false;
        *state.env_checked.lock() = None;
    }
    if matches!(result, Ok(Ok(true))) {
        state.record_engine_fingerprint();
    }
    result.map_err(|e| e.to_string())?
}

fn install_env_inner(
    app: &AppHandle,
    data_dir: PathBuf,
    resources: PathBuf,
    uv_exe: PathBuf,
    venv_python: PathBuf,
) -> Result<bool, String> {
    // 1. 复制 uv 到数据目录（安装目录可能只读）
    let bin_dir = data_dir.join("bin");
    std::fs::create_dir_all(&bin_dir).map_err(|e| e.to_string())?;
    if !uv_exe.is_file() {
        std::fs::copy(resources.join("uv.exe"), &uv_exe).map_err(|e| e.to_string())?;
    }

    // 2. 创建 venv（uv 会自动下载 Python 3.11）
    let venv_dir = data_dir.join("venv");
    emit_event(app, "env-step", serde_json::json!({ "index": 1, "total": 4, "name": "安装 Python 运行时", "status": "running" }));
    let uv_args = vec![
        "venv".to_string(),
        venv_dir.to_string_lossy().into_owned(),
        "--python".to_string(),
        PYTHON_VERSION.to_string(),
    ];
    let app_line = app.clone();
    let code = run_process(uv_exe.to_str().unwrap(), &uv_args, None, &[], move |line| {
        emit_event(&app_line, "env-line", serde_json::json!({ "text": line }));
    })
    .map_err(|e| e.to_string())?;
    if code != 0 {
        emit_event(app, "env-step", serde_json::json!({ "index": 1, "total": 4, "name": "安装 Python 运行时", "status": "failed" }));
        return Err("Python 运行时安装失败".into());
    }
    emit_event(app, "env-step", serde_json::json!({ "index": 1, "total": 4, "name": "安装 Python 运行时", "status": "done" }));

    if !venv_python.is_file() {
        return Err("venv python 不存在".into());
    }

    // 3. 依赖（一）
    emit_event(app, "env-step", serde_json::json!({ "index": 2, "total": 4, "name": "安装依赖（一）", "status": "running" }));
    let python = venv_python.to_string_lossy().into_owned();
    let mut args = vec!["pip".to_string(), "install".to_string(), "--python".to_string(), python.clone()];
    args.extend(BASE_DEPENDENCIES.iter().map(|dep| (*dep).to_string()));
    let app_line = app.clone();
    let code = run_process(uv_exe.to_str().unwrap(), &args, None, &[], move |line| {
        emit_event(&app_line, "env-line", serde_json::json!({ "text": line }));
    })
    .map_err(|e| e.to_string())?;
    if code != 0 {
        emit_event(app, "env-step", serde_json::json!({ "index": 2, "total": 4, "name": "安装依赖（一）", "status": "failed" }));
        return Err("依赖安装失败".into());
    }
    emit_event(app, "env-step", serde_json::json!({ "index": 2, "total": 4, "name": "安装依赖（一）", "status": "done" }));

    // 4. torch（CUDA 12.8 构建，兼容 RTX 30/40/50）
    emit_event(app, "env-step", serde_json::json!({ "index": 3, "total": 4, "name": "安装 PyTorch（约 2.5GB）", "status": "running" }));
    // 允许通过 AIRLLM_TORCH_INDEX_URL 覆盖官方索引（网络受限环境可换镜像源）。
    let torch_index = std::env::var("AIRLLM_TORCH_INDEX_URL").unwrap_or_else(|_| TORCH_INDEX_URL.to_string());
    let torch_args = vec![
        "pip".to_string(),
        "install".to_string(),
        "--python".to_string(),
        python.clone(),
        "torch>=2.4".to_string(),
        "--index-url".to_string(),
        torch_index,
    ];
    let app_line = app.clone();
    let code = run_process(uv_exe.to_str().unwrap(), &torch_args, None, &[], move |line| {
        emit_event(&app_line, "env-line", serde_json::json!({ "text": line }));
    })
    .map_err(|e| e.to_string())?;
    if code != 0 {
        emit_event(app, "env-step", serde_json::json!({ "index": 3, "total": 4, "name": "安装 PyTorch（约 2.5GB）", "status": "failed" }));
        return Err("PyTorch 安装失败".into());
    }
    emit_event(app, "env-step", serde_json::json!({ "index": 3, "total": 4, "name": "安装 PyTorch（约 2.5GB）", "status": "done" }));

    install_wheels_inner(app, &resources, &uv_exe, &venv_python)?;
    Ok(true)
}

/// 安装/升级捆绑的引擎与 AirLLM wheel（步骤 4；可独立于完整环境安装执行）。
fn install_wheels_inner(
    app: &AppHandle,
    resources: &Path,
    uv_exe: &Path,
    venv_python: &Path,
) -> Result<(), String> {
    emit_event(app, "env-step", serde_json::json!({ "index": 4, "total": 4, "name": "安装引擎与 AirLLM", "status": "running" }));
    let find_wheel = |prefix: &str| -> Option<PathBuf> {
        let mut wheels: Vec<PathBuf> = std::fs::read_dir(resources)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix) && n.ends_with(".whl"))
            })
            .collect();
        wheels.sort();
        wheels.into_iter().next()
    };
    let engine_whl = find_wheel("airllm_responses_sidecar-").ok_or_else(|| "找不到 engine wheel".to_string())?;
    let airllm_whl = find_wheel("airllm-").ok_or_else(|| "找不到 airllm wheel".to_string())?;
    let wheel_args = vec![
        "pip".to_string(),
        "install".to_string(),
        "--python".to_string(),
        venv_python.to_string_lossy().into_owned(),
        "--no-deps".to_string(),
        "--reinstall".to_string(),
        engine_whl.to_string_lossy().into_owned(),
        airllm_whl.to_string_lossy().into_owned(),
    ];
    let app_line = app.clone();
    let code = run_process(uv_exe.to_str().unwrap(), &wheel_args, None, &[], move |line| {
        emit_event(&app_line, "env-line", serde_json::json!({ "text": line }));
    })
    .map_err(|e| e.to_string())?;
    if code != 0 {
        emit_event(app, "env-step", serde_json::json!({ "index": 4, "total": 4, "name": "安装引擎与 AirLLM", "status": "failed" }));
        return Err("引擎安装失败".into());
    }
    emit_event(app, "env-step", serde_json::json!({ "index": 4, "total": 4, "name": "安装引擎与 AirLLM", "status": "done" }));
    Ok(())
}

// ── 模型 ────────────────────────────────────────────────────────────────

fn deps_ready_now(state: &EngineState) -> bool {
    if let Some(status) = state.env_checked.lock().clone() {
        return status.deps_ready;
    }
    let status = check_env_inner(state);
    *state.env_checked.lock() = Some(status.clone());
    status.deps_ready
}

#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, EngineState>,
    model_id: String,
    alias: Option<String>,
) -> Result<(), String> {
    if !deps_ready_now(&state) {
        return Err("引擎环境未就绪，请先安装环境".into());
    }
    let python = state.python().ok_or_else(|| "引擎环境未就绪".to_string())?;
    let mut args = vec![
        "-m".to_string(),
        "airllm_responses.cli".to_string(),
        "download".to_string(),
        "--id".to_string(),
        model_id.clone(),
        "--data-dir".to_string(),
        state.data_dir.to_string_lossy().into_owned(),
    ];
    let alias_out = alias.clone().unwrap_or_else(|| model_id.replace('/', "-"));
    if let Some(alias) = alias {
        args.push("--alias".to_string());
        args.push(alias);
    }
    {
        let settings = state.settings.lock();
        if !settings.endpoint.is_empty() {
            args.push("--endpoint".to_string());
            args.push(settings.endpoint.clone());
        }
        if !settings.model_root.is_empty() {
            args.push("--model-root".to_string());
            args.push(settings.model_root.clone());
        }
        if settings.download_workers > 0 {
            args.push("--workers".to_string());
            args.push(settings.download_workers.to_string());
        }
    }
    let app_clone = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        run_engine_cli(&app_clone, &python, &args, "model-progress")
    })
    .await
    .map_err(|e| e.to_string())?;
    let code = result.map_err(|e| e.to_string())?;
    emit_event(
        &app,
        "model-done",
        serde_json::json!({ "success": code == 0, "modelId": model_id, "alias": alias_out }),
    );
    if code != 0 {
        return Err("模型下载失败，请查看日志".into());
    }
    Ok(())
}

#[tauri::command]
pub async fn remove_model(
    app: AppHandle,
    state: State<'_, EngineState>,
    alias: String,
) -> Result<(), String> {
    let python = state.python().ok_or_else(|| "引擎环境未就绪".to_string())?;
    let mut args = vec![
        "-m".to_string(),
        "airllm_responses.cli".to_string(),
        "remove".to_string(),
        "--alias".to_string(),
        alias,
        "--data-dir".to_string(),
        state.data_dir.to_string_lossy().into_owned(),
    ];
    {
        let settings = state.settings.lock();
        if !settings.model_root.is_empty() {
            args.push("--model-root".to_string());
            args.push(settings.model_root.clone());
        }
    }
    let app_clone = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        run_engine_cli(&app_clone, &python, &args, "model-remove")
    })
    .await
    .map_err(|e| e.to_string())?;
    let code = result.map_err(|e| e.to_string())?;
    if code != 0 {
        return Err("删除模型失败".into());
    }
    Ok(())
}

// ── 服务 ────────────────────────────────────────────────────────────────

#[tauri::command]
pub async fn import_model(
    app: AppHandle,
    state: State<'_, EngineState>,
    dir: String,
    alias: String,
) -> Result<(), String> {
    let python = state.python().ok_or_else(|| "引擎环境未就绪".to_string())?;
    let mut args = vec![
        "-m".to_string(),
        "airllm_responses.cli".to_string(),
        "install-local".to_string(),
        "--dir".to_string(),
        dir,
        "--alias".to_string(),
        alias,
        "--data-dir".to_string(),
        state.data_dir.to_string_lossy().into_owned(),
    ];
    {
        let settings = state.settings.lock();
        if !settings.model_root.is_empty() {
            args.push("--model-root".to_string());
            args.push(settings.model_root.clone());
        }
    }
    let app_clone = app.clone();
    let result = tauri::async_runtime::spawn_blocking(move || {
        run_engine_cli(&app_clone, &python, &args, "model-progress")
    })
    .await
    .map_err(|e| e.to_string())?;
    let code = result.map_err(|e| e.to_string())?;
    if code != 0 {
        return Err("导入模型失败（请确认目录包含 config.json 与权重文件）".into());
    }
    Ok(())
}

#[tauri::command]
pub async fn start_service(
    app: AppHandle,
    state: State<'_, EngineState>,
    alias: String,
    settings: ServiceSettings,
) -> Result<(), String> {
    let python = state.python().ok_or_else(|| "引擎环境未就绪".to_string())?;
    if !list_installed(&state.data_dir).iter().any(|m| m.alias == alias) {
        return Err(format!("模型 {alias} 尚未安装"));
    }
    stop_service_inner(&state);
    {
        let mut guard = state.settings.lock();
        *guard = settings.clone();
    }
    let config_path = write_app_toml(&state, &alias, &settings)?;
    let api_key = load_or_create_api_key(&state);

    let mut command = Command::new(&python);
    hide_console(&mut command);
    command
        .args(["-m", "airllm_responses.cli", "serve", "--config"])
        .arg(&config_path)
        .env("AIRLLM_API_KEY", &api_key)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut record = ServiceRecord { child: command.spawn().map_err(|e| e.to_string())? };

    let handle = app.clone();
    let stdout = record.child.stdout.take().expect("stdout");
    let out_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                if value.get("event").is_some() {
                    emit_event(&handle, "service-event", value.clone());
                    continue;
                }
            }
            emit_event(&handle, "service-line", serde_json::json!({ "text": line }));
        }
    });
    let handle2 = app.clone();
    let stderr = record.child.stderr.take().expect("stderr");
    let err_thread = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            emit_event(&handle2, "service-line", serde_json::json!({ "text": line }));
        }
    });

    {
        let mut service = state.service.lock();
        *service = Some(record);
    }
    emit_event(&app, "service-start", serde_json::json!({ "model": alias, "port": settings.port }));
    let _ = out_thread;
    let _ = err_thread;
    Ok(())
}

#[tauri::command]
pub async fn stop_service(state: State<'_, EngineState>) -> Result<(), String> {
    stop_service_inner(&state);
    Ok(())
}

fn stop_service_inner(state: &EngineState) {
    let mut guard = state.service.lock();
    if let Some(mut record) = guard.take() {
        let _ = record.child.kill();
        let _ = record.child.wait();
    }
}

#[tauri::command]
pub async fn update_settings(state: State<'_, EngineState>, settings: ServiceSettings) -> Result<(), String> {
    {
        let mut guard = state.settings.lock();
        *guard = settings;
    }
    Ok(())
}

fn write_app_toml(state: &EngineState, alias: &str, settings: &ServiceSettings) -> Result<PathBuf, String> {
    std::fs::create_dir_all(&state.data_dir).map_err(|e| e.to_string())?;
    let win_path = |path: PathBuf| path.to_string_lossy().replace('\\', "/");
    let model_base = if settings.model_root.is_empty() {
        state.data_dir.join("models")
    } else {
        PathBuf::from(&settings.model_root)
    };
    let toml = format!(
        "[server]\nhost = \"127.0.0.1\"\nport = {port}\nmax_concurrent_requests = 1\npreload = {preload}\napi_key_file = \"{api_key_file}\"\n\n[model]\nname = \"{alias}\"\nendpoint = \"{endpoint}\"\nroot = \"{model_root}\"\ndir = \"{model_dir}\"\nshard_dir = \"{shard_dir}\"\nmanifest = \"{manifest}\"\n\n[engine]\ndevice = \"{device}\"\nmax_seq_len = {max_seq_len}\nmax_output_tokens = {max_output_tokens}\ndownload_workers = {download_workers}\n",
        port = settings.port,
        preload = settings.preload,
        endpoint = settings.endpoint,
        model_root = settings.model_root,
        api_key_file = win_path(state.api_key_path()),
        model_dir = win_path(model_base.join(alias)),
        shard_dir = win_path(state.data_dir.join("shards").join(alias)),
        manifest = win_path(state.data_dir.join("manifests").join(format!("{alias}.json"))),
        device = settings.device,
        max_seq_len = settings.max_seq_len,
        max_output_tokens = settings.max_output_tokens,
        download_workers = settings.download_workers,
    );
    let path = state.app_toml_path();
    std::fs::write(&path, toml).map_err(|e| e.to_string())?;
    Ok(path)
}

// ── API key ─────────────────────────────────────────────────────────────

fn load_or_create_api_key(state: &EngineState) -> String {
    let path = state.api_key_path();
    if let Ok(text) = std::fs::read_to_string(&path) {
        let text = text.trim().to_string();
        if !text.is_empty() {
            return text;
        }
    }
    let key = new_api_key();
    let _ = std::fs::create_dir_all(&state.data_dir);
    let _ = std::fs::write(&path, format!("{key}\n"));
    key
}

fn new_api_key() -> String {
    // 本地 API key，非安全用途；基于种子与 xorshift 的确定性序列。
    let seed: u64 = (std::process::id() as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        );
    let mut value = seed ^ 0xdeadbeefcafef00d;
    let mut bytes = [0u8; 20];
    for byte in bytes.iter_mut() {
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *byte = (value & 0xff) as u8;
    }
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[tauri::command]
pub async fn get_api_key(state: State<'_, EngineState>) -> Result<String, String> {
    Ok(load_or_create_api_key(&state))
}

#[tauri::command]
pub async fn open_path(app: AppHandle, state: State<'_, EngineState>, target: String) -> Result<(), String> {
    let path = match target.as_str() {
        "data" => state.data_dir.clone(),
        "models" => state.data_dir.join("models"),
        "shards" => state.data_dir.join("shards"),
        "logs" => state.data_dir.join("logs"),
        other => state.data_dir.join(other),
    };
    std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_path(path.to_string_lossy().into_owned(), None::<&str>)
        .map_err(|e| e.to_string())
}

// ── 设置 ────────────────────────────────────────────────────────────────

pub fn setup(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let state = EngineState::new(app);
    std::fs::create_dir_all(&state.data_dir)?;
    app.manage(state);
    maybe_auto_update_engine(app.clone());
    Ok(())
}

/// 启动时探测：安装包中的引擎 wheel 与已安装版本不同时，后台自动升级
/// （复用安装步骤 4，不重新下载 Python/PyTorch）。
fn maybe_auto_update_engine(app: AppHandle) {
    let state = app.state::<EngineState>();
    let data_dir = state.data_dir.clone();
    let resources = state.resources.clone();
    let uv_exe = state.uv_exe.clone();
    let venv_python = state.venv_python_dir();
    let update = state.needs_engine_update();
    if !update {
        return;
    }
    std::thread::spawn(move || {
        if install_wheels_inner(&app, &resources, &uv_exe, &venv_python).is_ok() {
            // 仅成功后记录指纹；失败保留旧记录，下次启动再重试。
            let fingerprint = engine_fingerprint_for(&resources);
            let path = data_dir.join("engine-wheel.sha256");
            if !fingerprint.is_empty() {
                let _ = std::fs::write(&path, format!("{fingerprint}\n"));
            }
            let _ = app.emit("engine-updated", serde_json::json!({ "updated": true }));
        }
    });
}

/// 独立于 State 的指纹计算（供后台线程使用）。
fn engine_fingerprint_for(resources: &Path) -> String {
    fn sha256_hex(data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }
    let mut wheels: Vec<PathBuf> = std::fs::read_dir(resources)
        .ok()
        .map(|entries| entries.filter_map(|e| e.ok()).map(|e| e.path()).collect::<Vec<PathBuf>>())
        .unwrap_or_default();
    wheels.sort();
    for path in wheels {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("airllm_responses_sidecar-") && name.ends_with(".whl") {
                if let Ok(data) = std::fs::read(&path) {
                    return sha256_hex(&data);
                }
            }
        }
    }
    String::new()
}
