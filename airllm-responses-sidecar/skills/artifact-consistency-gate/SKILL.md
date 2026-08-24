---
name: artifact-consistency-gate
description: >
  交付前强制检查 README、部署脚本、配置和验证声明的一致性。适用于创建或修改部署文档、声称通用/可配置/可复现、简化安装流程、完成交付或发布说明时。检查本机路径泄漏、参数与默认值漂移、secret/test key、脚本边界和全新临时目录 dry-run；任一门禁失败都不得声称完成。
---

# Artifact Consistency Gate

这是交付门禁，不是提醒清单。必须执行检查并保留结果；不能只阅读本文件后声称通过。

## 触发条件

在以下任一情况出现时加载或明确调用本 skill：

- 新建、修改或审核 README、部署教程、安装脚本、启动脚本、配置示例。
- 文档声称“通用”“可分发”“可配置”“可复现”“三步部署”“开箱即用”。
- 修改安装路径、运行数据路径、上游源码路径、模型参数、监听参数或凭据注入方式。
- 准备输出“完成”“已验证”“可部署”“已交付”等结论。
- 用户明确要求 `artifact-consistency-gate`。

skill 只有在被自动加载或明确调用时有效。未加载时不能声称本门禁已执行。

## 强制门禁

### 1. 建立声明清单

从目标 README 和脚本提取可验证声明，至少记录：

- 支持的操作系统、Python 版本和运行时类型。
- 默认安装目录和默认运行数据目录。
- 可配置参数及其默认值。
- 哪一步会安装可选依赖。
- 哪一步会下载模型。
- 凭据从哪里读取。
- 哪些路径必须是绝对路径。
- 哪些能力明确不支持。

每条声明必须绑定到脚本、配置或命令的实际证据。无法绑定的句子不得作为完成结论。

### 2. 扫描通用性和本机泄漏

对 README、部署脚本、配置示例和生成的命令执行扫描。默认阻止以下内容出现在声称通用的示例中：

```regex
(?m)^\s*[A-Za-z]:\\
(?i)/Users/
(?i)/home/
```

实际用户名、仓库绝对路径、真实运行目录、测试目录和临时目录也属于泄漏，即使没有命中默认正则。将当前工作区、用户目录和运行目录加入运行时生成的禁止集合，不要把这些值写进 skill 或通用文档。

允许的路径示例只有：

- `<install-root>`、`<runtime-root>`、`<upstream-root>` 等明确占位符。
- `$PSScriptRoot`、`$env:LOCALAPPDATA`、`$env:USERPROFILE` 等运行时表达式。
- 使用 `Path`、`Join-Path`、`Resolve-Path` 等根据参数计算的路径。

命中禁止集合时，阻止“通用”“可分发”“已完成”等结论；不要只替换一句说明而保留代码块中的泄漏。

### 3. 扫描凭据和测试残留

阻止以下内容出现在 README、脚本、manifest、配置或日志示例中：

- 真实 token、API key、Bearer token、密码或私钥。
- 测试 key、测试模型目录、临时服务地址或本机 smoke-test 值。
- 将 secret 作为脚本参数或命令行参数传递。
- 文档声称“只从环境变量读取”，脚本却存在 `-ApiKey`、`-Token` 等参数。

示例只能使用 `<api-key>`、`<token>` 等占位符，并要求脚本从环境变量读取。检查正文、代码围栏、注释、默认值和错误消息。

### 4. 核对参数和默认值

对每个部署脚本分别取得参数定义：

```powershell
Get-Help .\scripts\setup.ps1 -Full
Get-Help .\scripts\download-model.ps1 -Full
Get-Help .\scripts\start.ps1 -Full
```

逐项比较 README、脚本和配置：

- README 中出现的参数必须真实存在。
- 脚本必需参数必须出现在 README 的最短流程中。
- 同名参数的默认值必须一致。
- `InstallRoot`、`RuntimeRoot`、`UpstreamRoot` 等路径参数必须在 setup/download/start 之间保持同一语义。
- 模型别名、manifest、模型目录和分片目录必须使用同一 alias 规则。
- README 不得继续描述已经删除的参数、旧脚本名或旧安装模式。

参数漂移即失败，不用“用户可以自行调整”掩盖不一致。

### 5. 检查脚本边界

每个 PowerShell 脚本必须：

- 以 Windows PowerShell 5.1 可读的 UTF-8 BOM 或纯 ASCII 保存。
- 通过 `Parser::ParseFile`，且 `$errors.Count` 为零。
- 对用户输入的 alias、路径和 revision 做边界检查。
- 将 manifest、模型目录、分片目录和上游目录规范化为绝对路径。
- 在访问 Python、manifest、模型或启动服务之前拒绝非法 alias，例如空值、`.`、`..` 和路径分隔符。
- 不把 secret 写入进程参数。
- 使用明确的非零退出码报告失败。

### 6. 执行全新临时目录 dry-run

使用标准 CPython 3.11，不能用其他 Python 版本、已有虚拟环境、旧 runtime 或 embeddable fallback 代替。使用全新、非源码目录的临时路径：

```powershell
$InstallRoot = Join-Path ([System.IO.Path]::GetTempPath()) "artifact-gate-install"
$RuntimeRoot = Join-Path ([System.IO.Path]::GetTempPath()) "artifact-gate-runtime"

.\scripts\setup.ps1 `
  -PythonExe "<python-3.11>\python.exe" `
  -InstallRoot $InstallRoot `
  -RuntimeRoot $RuntimeRoot
```

验证：

- setup 成功退出。
- `<InstallRoot>\venv\Scripts\python.exe` 存在。
- `<RuntimeRoot>` 及其模型、分片、cache、日志父目录存在。
- setup 没有创建模型 snapshot，也没有触发下载。
- 用新进程运行 `download-model.ps1 -?` 和 `start.ps1 -?`。
- 用非法 alias（至少 `..`）分别执行 download/start，确认在访问 Python、manifest、模型或服务前以非零状态拒绝。

如果当前环境没有标准 CPython 3.11，状态是“未验证”或“阻塞”，不能用 3.13、旧 venv、静态 Parser 或契约测试替代真实 dry-run。

### 7. 证据和交付结论

输出固定格式：

```text
通用路径扫描: PASS/FAIL
凭据与测试残留扫描: PASS/FAIL
README/脚本参数一致性: PASS/FAIL
PowerShell 解析: PASS/FAIL
非法输入边界: PASS/FAIL
全新目录 dry-run: PASS/FAIL/UNVERIFIED
交付结论: ALLOWED/BLOCKED
```

以下任一状态都必须是 `BLOCKED`：

- 任意扫描命中本机路径、用户名、工作区路径或真实 runtime。
- 参数、默认值或脚本名不一致。
- secret 出现在命令行、代码或示例默认值。
- Parser 失败。
- 非法输入未在副作用前拒绝。
- 全新目录 dry-run 未执行、失败或环境不足。

## 反残留规则

修订时重新扫描全文，不允许被删除的方案、错误中间决策、旧路径、旧参数或“修正说明”残留在最终 README、脚本、注释、metadata 或发布说明中。最终产物只描述当前有效行为；保留必要的迁移、弃用或边界说明，但不要记录内部编辑过程。

## 最低完成标准

只有在所有门禁通过并有命令输出作为证据时，才能使用“已完成”“可分发”“部署已验证”等结论。否则必须明确指出失败门禁、实际阻塞点和已经完成的可达范围。
