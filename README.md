# AirLLM Desktop

在本地运行大模型的桌面应用：一键安装、图形化引导、模型下载与聊天，开箱即用。

参考 [FreeToken](https://github.com/FlashML-org/FreeToken) 的桌面应用形态：Tauri 桌面外壳 + 图形化新用户向导 +
模型目录 + 聊天界面；底层引擎为 AirLLM 内存节俭推理（分层流式加载，可运行大于显存的模型）。

## 系统要求

- Windows 10/11（x64，自带 WebView2 运行时）
- NVIDIA GPU（CUDA 12.8 驱动；RTX 30/40/50 系列均可）
- 磁盘空间：安装包约 14MB；首次安装运行环境约需 6GB（Python 3.11 + PyTorch CUDA + 依赖）
- 模型另需磁盘空间（最小模型约 1GB）

## 从零部署（用户视角）

1. 到 [GitHub Releases](https://github.com/Liccsu/airllm-desktop/releases/latest)
   下载最新的 `AirLLM_<版本>_x64-setup.exe`，运行完成安装。
   国内网络可改用加速地址下载（gh-proxy 代理，与官方文件一致）：
   `https://gh-proxy.org/https://github.com/Liccsu/airllm-desktop/releases/latest/download/AirLLM_<版本>_x64-setup.exe`
2. 首次启动进入欢迎引导：
   - 检查显卡（NVIDIA GPU 与显存）；
   - 一键安装运行环境（Python 3.11、PyTorch CUDA 版、引擎依赖，全程进度展示，无命令行）；
   - 选择模型源（官方 huggingface.co / 国内镜像 https://hf-mirror.com / 自定义地址）与模型下载目录；
   - 从模型库选择模型（内置推荐目录），下载即有进度条；
   - 启动模型服务，进入聊天界面。
3. 日常使用：左侧导航提供聊天、模型库（下载/启动/删除/导入本地模型）、设置
   （端口、设备、长度、预加载、模型源、模型下载目录、下载线程数）、引擎日志；
   左下角实时显示显存与内存占用。

## 模型源

模型默认从 [huggingface.co](https://huggingface.co) 下载。国内网络可在引导中选择镜像源
（https://hf-mirror.com）或在设置页填入任意 Hugging Face 镜像地址（如公司内部代理）；
该设置会用于后续所有模型下载，配置持久化。模型下载目录与下载线程数（默认 8 并行）同样可在设置页调整，
模型目录也可在引导时指定到其他磁盘。

## 导入本地模型

模型库支持导入本地目录（含 `config.json`、tokenizer 与权重文件）：以目录链接方式导入（Windows
junction / POSIX symlink），**不复制**模型文件，模型数据与源目录保持同步；删除别名只断开链接，
源目录不受影响。

## 首次安装运行环境时网络受限

PyTorch CUDA 版默认从官方索引 `https://download.pytorch.org/whl/cu128` 下载（约 2.5GB）。
网络受限环境可在启动应用前设置环境变量切换镜像源：

```powershell
# PyTorch CUDA 版索引（国内一般无需切换；如遇网络问题可换镜像）
$env:AIRLLM_TORCH_INDEX_URL = "<镜像的 cu128 索引地址>"
# Python 运行时下载镜像（可选；uv 官方默认从 GitHub 下载 python-build-standalone）
$env:UV_PYTHON_INSTALL_MIRROR = "<python-build-standalone 镜像地址>"
```

引擎其余 Python 依赖默认从**中科大 PyPI 镜像**（https://mirrors.ustc.edu.cn/pypi/web/simple/）安装；
airllm 上游源码克隆默认走 gh-proxy 加速（失败自动回退官方源）。

## 升级

- **引擎/airllm 升级**：重新安装最新版安装包后，应用启动时会自动检测并后台升级引擎与推理库
  （仅替换 Python wheel，不重下 Python/PyTorch；失败会在下次启动重试）。
- **桌面应用升级**：设置页可检查更新（更新清单先经 gh-proxy 加速获取，失败自动回退官方）；
  更新发布后，应用会自动下载新版安装包并提示安装。
- 升级不会影响模型、分片与对话数据（均位于用户数据目录 `%LOCALAPPDATA%\AirLLM`）。

## 引擎使用（可选，通常无需手工操作）

```bash
airllm-engine serve --config %LOCALAPPDATA%\AirLLM\app.toml   # 启动服务
airllm-engine download --id Qwen/Qwen2.5-0.5B-Instruct        # 下载模型（默认官方源）
airllm-engine catalog                                          # 内置模型目录
airllm-engine download --id <hf-id> --endpoint https://hf-mirror.com --alias <名称>  # 镜像源下载
airllm-engine install-local --dir <snapshot-dir> --alias <名称>  # 链接式导入本地模型
```

服务兼容 OpenAI Responses API：`http://127.0.0.1:8000/v1/responses`，
API key 见设置页，或数据目录 `api_key.txt`。模型清单兼容旧 manifest 来源。

## 从源码构建安装包（开发者）

前置：Rust 工具链、Node.js 18+、Python 3.11+、uv；Windows 10/11（NSIS 打包目标为 Windows）。

```powershell
# 1. 获取仓库（本仓库不含 airllm 上游；构建脚本会在缺失时自动拉取）
# 国内加速（gh-proxy）：
git clone https://gh-proxy.org/https://github.com/Liccsu/airllm-desktop.git
# 官方地址：git clone https://github.com/Liccsu/airllm-desktop.git
cd airllm-desktop

# 2. 生成捆绑资源（engine/airllm wheel + uv.exe + catalog.json）
python tools/build_bundle.py

# 3. 构建前端与 NSIS 安装包
cd desktop
npm install
npm run tauri build      # 产物：src-tauri/target/release/bundle/nsis/AirLLM_*.x64-setup.exe
```

`airllm/` 为上游 [AirLLM](https://github.com/lyogavin/airllm)（仅本地构建依赖，不作为仓库内容提交）。

## 测试

```bash
cd airllm-responses-sidecar
python -m unittest discover -s tests -p "test_*.py"
```

## 发布新版本（维护者）

打 tag 即触发 GitHub Actions 自动构建并发布到 Releases（含 updater 更新清单）：

```bash
git tag v0.2.0
git push origin v0.2.0
```

需要仓库 Secrets：`TAURI_SIGNING_PRIVATE_KEY`（updater 签名私钥，生成方式：
`npx tauri signer generate -w ~/.tauri/airllm.key` 后填入内容）、
`TAURI_SIGNING_PRIVATE_KEY_PASSWORD`。公钥已在仓库配置中，用户侧无需任何密钥。

## 目录结构（运行时数据）

```
%LOCALAPPDATA%\AirLLM\
  app.toml          # 引擎配置（桌面应用写入）
  api_key.txt       # 本地 API key
  venv\             # Python 运行环境
  bin\uv.exe        # 运行时安装器
  engine-wheel.sha256  # 引擎版本指纹（用于自动升级）
  models\<alias>\   # 模型（huggingface snapshot 或本地链接）
  shards\<alias>\   # AirLLM 分层分片
  manifests\<alias>.json  # 审核清单（无 secret）
  cache\hub\        # huggingface 下载缓存
```

## 隐私

模型与对话数据全部保存在本机；除安装、下载与引擎运行时必要的联网外不上传任何数据。
