"""airllm-engine 命令行入口。

桌面应用直接以参数启动本命令并逐行解析 stdout 事件；用户也可手动使用::

    airllm-engine serve [--config <app.toml>] [--model <alias>]
    airllm-engine download --id <model-id> [--revision <rev>] [--alias <name>]
    airllm-engine install-local --dir <snapshot-dir> --alias <name>
    airllm-engine catalog [--json]
    airllm-engine status [--base-url <url>] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .app_config import (
    AppConfig,
    AppModel,
    TOML_NAME,
    default_data_dir,
    load_api_key,
    load_app_config,
    to_settings,
)
from .catalog import load_catalog
from .config import ConfigError, Settings
from .downloader import (
    DownloadError,
    download_catalog_model,
    install_from_local,
    installed_models,
    remove_model_alias,
    resolve_revision,
)
from .events import emit, emit_error

DEFAULT_TOKEN_ENV = "HF_TOKEN"


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="app.toml 的绝对路径")
    parser.add_argument("--data-dir", type=Path, default=None, help="应用数据目录（默认见 default_data_dir）")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airllm-engine",
        description="AirLLM 本地引擎：服务、模型安装与运行时管理",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="启动 Responses API 服务")
    _add_common_options(serve)
    serve.add_argument("--model", default=None, help="模型别名（默认使用配置中的模型）")
    serve.add_argument("--host", default=None, help="监听地址")
    serve.add_argument("--port", type=int, default=None, help="监听端口")
    serve.add_argument("--device", default=None, help="推理设备（如 cuda:0，默认 cuda:0）")
    serve.add_argument("--no-preload", action="store_true", help="启动时不加载模型")

    download = subparsers.add_parser("download", help="下载并审核模型")
    _add_common_options(download)
    download.add_argument("--id", required=True, help="模型仓库 id（huggingface，如 Qwen/Qwen2.5-0.5B-Instruct）")
    download.add_argument("--revision", default=None, help="revision；为空时使用默认分支并固定实际 commit")
    download.add_argument("--alias", default=None, help="本地别名（默认使用模型 id）")
    download.add_argument("--endpoint", default=None, help="镜像源地址（如 https://hf-mirror.com）；为空使用官方源")
    download.add_argument("--model-root", default=None, help="模型根目录（默认 <data-dir>/models）")
    download.add_argument("--workers", type=int, default=None, help="下载并行线程数（默认 8；网络受限可调小为 1）")
    download.add_argument("--token-env", default=DEFAULT_TOKEN_ENV, help="读取凭据的环境变量名")
    download.add_argument("--no-progress-events", action="store_true", help="仅输出最终结果")

    local = subparsers.add_parser("install-local", help="从本地 snapshot 目录安装模型")
    _add_common_options(local)
    local.add_argument("--dir", required=True, help="本地 Transformers snapshot 目录")
    local.add_argument("--alias", required=True, help="本地别名")
    local.add_argument("--model-root", default=None, help="模型根目录（默认 <data-dir>/models）")

    remove = subparsers.add_parser("remove", help="删除模型别名（本地链接仅断开链接，不删源）")
    _add_common_options(remove)
    remove.add_argument("--alias", required=True, help="要删除的模型别名")
    remove.add_argument("--model-root", default=None, help="模型根目录（默认 <data-dir>/models）")

    catalog_parser = subparsers.add_parser("catalog", help="列出内置模型目录")
    catalog_parser.add_argument("--json", action="store_true", help="以 JSON 输出")

    status = subparsers.add_parser("status", help="查询运行中的服务")
    status.add_argument("--base-url", default="http://127.0.0.1:8000", help="服务地址")
    status.add_argument("--json", action="store_true", help="以 JSON 输出")

    return parser


def _resolve_config(args: argparse.Namespace) -> AppConfig:
    config_path = args.config
    if config_path is None and getattr(args, "data_dir", None) is not None:
        # 桌面应用只传 --data-dir：自动读取数据目录中的 app.toml（镜像源设置等）。
        candidate = Path(args.data_dir) / TOML_NAME
        if candidate.is_file():
            config_path = candidate
    return load_app_config(
        config_path=config_path,
        data_dir=args.data_dir,
    )


# ── serve ────────────────────────────────────────────────────────────────

def _model_for_serve(config: AppConfig, alias: str | None) -> AppModel:
    """按别名对齐数据目录布局；未显式配置时从 manifests/models/shards 推断。"""

    name = alias or config.model.name
    return config.model if alias is None else AppModel(
        name=name,
        id=config.model.id,
        revision=config.model.revision,
        dir=config.model.dir or config.model_dir_for(name),
        shard_dir=config.model.shard_dir or config.shard_dir_for(name),
        manifest=config.model.manifest or config.manifest_path_for(name),
    )


def cmd_serve(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    config = AppConfig(
        data_dir=config.data_dir,
        server=config.server,
        model=_model_for_serve(config, args.model),
        engine=config.engine,
    )
    server_cfg = config.server
    if args.host:
        server_cfg = _replace(server_cfg, host=args.host)
    if args.port:
        server_cfg = _replace(server_cfg, port=args.port)
    engine_cfg = config.engine
    if args.device:
        engine_cfg = _replace(engine_cfg, device=args.device)
    if args.no_preload:
        server_cfg = _replace(server_cfg, preload=False)
    config = AppConfig(data_dir=config.data_dir, server=server_cfg, model=config.model, engine=engine_cfg)

    try:
        api_key = load_api_key(config)
    except ConfigError as exc:
        emit_error("serve.failed", str(exc))
        print(str(exc), file=sys.stderr)
        return 2

    # 设备自动降级：配置为 cuda 但当前环境 CUDA 不可用时，回退到 cpu 并提示。
    if config.engine.device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                config = AppConfig(
                    data_dir=config.data_dir,
                    server=config.server,
                    model=config.model,
                    engine=_replace(config.engine, device="cpu"),
                )
                emit(
                    "serve.warning",
                    message="未检测到可用 CUDA，已自动使用 CPU 推理（速度较慢）",
                )
        except ImportError:
            pass

    settings = to_settings(config, api_key)

    # 保证服务超时校验到模型清单；无效路径直接失败优于启动一个永远 503 的进程。
    from .server import create_app

    emit(
        "serve.started",
        host=settings.host,
        port=settings.port,
        model=settings.model_name,
        device=settings.device,
        preload=settings.preload,
        data_dir=str(config.data_dir),
    )

    import uvicorn

    application = create_app(runtime=None, settings=settings)

    stopping = False

    def _on_signal(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        if not stopping:
            stopping = True
            emit("serve.stopping")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    uvicorn.run(
        application,
        host=settings.host,
        port=settings.port,
        log_level="warning",
        access_log=False,
    )
    emit("serve.stopped")
    return 0


def _replace(instance: Any, **changes: Any) -> Any:
    import dataclasses

    return dataclasses.replace(instance, **changes)


# ── download / install-local / catalog / status ─────────────────────────

def cmd_download(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    token = os.environ.get(args.token_env, "") or None
    endpoint = args.endpoint or config.model.endpoint
    model_root = args.model_root or None
    events_enabled = not args.no_progress_events
    try:
        result = download_catalog_model(
            model_id=args.id,
            alias=args.alias,
            data_dir=config.data_dir,
            token=token,
            revision=args.revision,
            endpoint=endpoint,
            model_root=Path(model_root) if model_root else None,
            workers=args.workers,
            progress=_download_progress if events_enabled else None,
        )
    except DownloadError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "model_id": args.id,
                "alias": result.model_dir.name,
                "revision": result.revision,
                "model_dir": str(result.model_dir),
                "manifest": str(result.manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


_last_progress_at: float = 0.0


def _download_progress(file: str, done: int, total: int, stage: str) -> None:
    """转发下载进度；同文件高频块事件节流到每秒 2 次。"""

    global _last_progress_at
    import time

    now = time.monotonic()
    if stage not in {"started", "done"} and now - _last_progress_at < 0.5:
        return
    _last_progress_at = now
    emit(
        "download.progress",
        file=file,
        done_bytes=done,
        total_bytes=total,
        stage=stage,
    )


def cmd_install_local(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    try:
        result = install_from_local(
            args.dir,
            args.alias,
            config.data_dir,
            model_root=Path(args.model_root) if getattr(args, "model_root", None) else None,
        )
    except DownloadError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "alias": result.model_dir.name,
                "revision": result.revision,
                "model_dir": str(result.model_dir),
                "manifest": str(result.manifest),
            },
            ensure_ascii=False,
        )
    )
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    from .catalog import CatalogError

    try:
        catalog = load_catalog()
    except CatalogError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(catalog, ensure_ascii=False))
        return 0
    for entry in catalog["models"]:
        size_gb = (entry.get("size_bytes") or 0) / 1e9
        marker = "*" if entry.get("recommended") else " "
        print(f"{marker} {entry['id']:<32} {size_gb:6.1f}GB  ~{entry.get('vram_gb')}GB VRAM  {entry.get('license', '')}")
    return 0


def cmd_remove(args: argparse.Namespace) -> int:
    config = _resolve_config(args)
    try:
        remove_model_alias(
            args.alias,
            config.data_dir,
            model_root=Path(args.model_root) if getattr(args, "model_root", None) else None,
        )
    except OSError as exc:
        print(f"删除模型失败: {exc}", file=sys.stderr)
        return 1
    emit("model.removed", alias=args.alias)
    print(json.dumps({"removed": args.alias}, ensure_ascii=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    import urllib.error
    import urllib.request

    summary: dict[str, Any] = {"base_url": args.base_url, "healthy": False}
    try:
        with urllib.request.urlopen(f"{args.base_url}/healthz", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        summary.update(payload)
        summary["healthy"] = bool(payload.get("ready"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        summary["error"] = str(exc)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
        return 0 if summary["healthy"] else 1
    print(f"base_url: {summary['base_url']}")
    print(f"ready    : {summary.get('ready', False)}")
    print(f"model    : {summary.get('model', '')}")
    return 0 if summary["healthy"] else 1


# ── 入口 ────────────────────────────────────────────────────────────────

_COMMANDS = {
    "serve": cmd_serve,
    "download": cmd_download,
    "install-local": cmd_install_local,
    "remove": cmd_remove,
    "catalog": cmd_catalog,
    "status": cmd_status,
}


def main(argv: Sequence[str] | None = None) -> int:
    # 管道输出（桌面端接管 stdout/stderr）时 Windows 默认 cp1252 会因中文崩溃，
    # 统一强制 UTF-8 并用替换符兜底。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
