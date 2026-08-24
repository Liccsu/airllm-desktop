# AirLLM Responses Sidecar（AirLLM 引擎）

为 AirLLM 提供本地 OpenAI Responses API 服务与模型管理。桌面应用（`../desktop`）通过本引擎完成
环境准备、模型下载与对话；本目录是引擎的完整源码与测试。

- `POST /v1/responses`（非流式 JSON + Responses 风格 SSE 流式）
- 单模型、单进程、串行生成
- 模型、分片和运行数据与源码分离，全部位于用户数据目录
- 提供 `airllm-engine` 命令行（桌面应用也调用它）

## 命令

```bash
airllm-engine serve [--config <app.toml>]       # 启动服务
airllm-engine download --id <hf-repo-id> [--alias <name>]  # 下载并审核模型
airllm-engine install-local --dir <snapshot-dir> --alias <name>  # 安装本地模型
airllm-engine catalog [--json]                  # 内置模型目录
airllm-engine status [--base-url http://127.0.0.1:8000] [--json]
```

`serve` 就绪前会校验审核清单（`manifests/<alias>.json`），模型目录与分片目录位于
数据目录 `models/<alias>`、`shards/<alias>`。API key 从环境变量 `AIRLLM_API_KEY`
或配置文件 `api_key_file`（默认数据目录 `api_key.txt`）读取，**不**接受命令行参数。

## 配置

优先级：命令行参数 > `app.toml` > 环境变量（`AIRLLM_*`，兼容旧脚本）> 默认值。
完整示例见 [config/app.example.toml](config/app.example.toml)。

数据目录布局：

```text
<data-dir>/
  app.toml               # 引擎配置（桌面应用写入）
  api_key.txt            # 本地 API key
  models/<alias>/        # 模型（huggingface snapshot）
  shards/<alias>/        # AirLLM 分层分片
  manifests/<alias>.json # 审核清单（无 secret）
  cache/hub/      # 下载缓存
```

## 服务 API

`Authorization: Bearer <api-key>`，模型名为 `[model] name`（默认 `airllm-local`）。

| 端点 | 说明 |
|---|---|
| `GET /healthz` | 就绪状态（模型加载完成后 `ready=true`） |
| `GET /v1/models` | 模型列表（服务就绪前 503） |
| `POST /v1/responses` | 对话生成；`stream=true` 时输出 SSE 事件 |

请求字段支持 `model`、`input`（字符串或消息数组）、`instructions`、`stream`、
`max_output_tokens`、`temperature`、`top_p`、`store=false`。不支持 tools、多模态、
批量输入与持久化。

| HTTP | 含义 |
|---|---|
| 401 | API key 缺失或错误 |
| 400 | 请求字段、模型别名或输入无效 |
| 503 | manifest、模型目录、分片目录或模型状态未就绪 |
| 500 | 已加载模型生成失败 |

## 模型下载

`download` 使用 huggingface_hub 下载到 staging，校验（config.json、tokenizer、权重文件）后
原子发布到模型根目录（默认 `models/<alias>`，可用 `--model-root` 或配置 `[model] root` 指定），
并写入无 secret 审核清单（来源标注 huggingface）。revision 为空时使用仓库默认分支并从缓存
固定实际 commit；下载进度通过 stdout JSON 事件输出（`download.started` / `download.files` /
`download.progress` / `download.done`）。镜像源通过 `--endpoint` 传入（如 https://hf-mirror.com，
镜像源自动禁用 xet 后端以保证兼容）。

凭据可经环境变量转发（`--token-env`，默认 `HF_TOKEN`），不写入配置。

`install-local` 以目录链接（Windows junction / POSIX symlink）导入本地模型，不复制文件；
`remove` 删除别名时仅断开链接，源目录保持不变。

## 测试

```bash
python -m unittest discover -s tests -p "test_*.py"
```

测试全部离线可跑（不下载模型、不依赖 GPU）。服务端到端验证由桌面应用引导流程覆盖。
