"""引擎应用配置：app.toml、数据目录布局与兼容的环境变量覆盖。

配置优先级：命令行参数 > app.toml > 环境变量(``AIRLLM_*``，与旧脚本兼容) > 默认值。
数据目录中保存模型、分片、清单、下载缓存与 API key。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .config import ConfigError, Settings, _absolute_path, _positive_int

TOML_NAME = "app.toml"
API_KEY_FILE = "api_key.txt"


def default_data_dir() -> Path:
    """返回当前平台的应用数据目录。"""

    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", "")) or (Path.home() / "AppData" / "Local")
        return base / "AirLLM"
    return Path.home() / ".local" / "share" / "airllm"


@dataclass(frozen=True)
class AppServer:
    host: str = "127.0.0.1"
    port: int = 8000
    max_concurrent_requests: int = 1
    preload: bool = False
    api_key_file: Path | None = None


@dataclass(frozen=True)
class AppModel:
    """一个已安装模型的别名、位置、来源与下载源。"""

    name: str = "airllm-local"
    id: str | None = None
    revision: str | None = None
    dir: Path | None = None
    shard_dir: Path | None = None
    manifest: Path | None = None
    endpoint: str | None = None
    root: str | None = None


@dataclass(frozen=True)
class AppEngine:
    device: str = "cuda:0"
    max_seq_len: int = 512
    max_output_tokens: int = 128


@dataclass(frozen=True)
class AppConfig:
    """引擎进程的完整配置。"""

    data_dir: Path
    server: AppServer = field(default_factory=AppServer)
    model: AppModel = field(default_factory=AppModel)
    engine: AppEngine = field(default_factory=AppEngine)

    @property
    def config_path(self) -> Path:
        return self.data_dir / TOML_NAME

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def shards_dir(self) -> Path:
        return self.data_dir / "shards"

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def hub_cache_dir(self) -> Path:
        return self.cache_dir / "hub"

    @property
    def api_key_path(self) -> Path:
        return self.data_dir / API_KEY_FILE

    def manifest_path_for(self, model_name: str) -> Path:
        return self.manifests_dir / f"{model_name}.json"

    def model_dir_for(self, model_name: str) -> Path:
        return self.models_dir / model_name

    @property
    def model_root(self) -> Path:
        """模型下载根目录；配置了 [model] root 时使用之。"""
        configured = self.model.root
        if configured:
            try:
                return _absolute_path(configured.replace("${DATA_DIR}", str(self.data_dir)), "model.root")
            except ConfigError:
                pass
        return self.models_dir

    def shard_dir_for(self, model_name: str) -> Path:
        return self.shards_dir / model_name


def _toml_section(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = payload.get(name)
    return section if isinstance(section, Mapping) else {}


def _toml_int(payload: Mapping[str, object], name: str, default: int) -> int | None:
    value = payload.get(name)
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _toml_bool(payload: Mapping[str, object], name: str, default: bool) -> bool | None:
    value = payload.get(name)
    if not isinstance(value, bool):
        return None
    return value


def load_app_config(
    config_path: str | os.PathLike[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """加载应用配置；不存在或无效的 toml 回退到默认值。"""

    env = os.environ if environ is None else environ
    resolved_data = _absolute_path(data_dir, "data_dir") if data_dir else default_data_dir()

    payload: dict[str, object] = {}
    if config_path is not None:
        path = _absolute_path(config_path, "config_path")
        if path.is_file():
            try:
                with path.open("rb") as stream:
                    payload = tomllib.load(stream)
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ConfigError(f"无法读取配置文件: {path}") from exc

    server_raw = _toml_section(payload, "server")
    model_raw = _toml_section(payload, "model")
    engine_raw = _toml_section(payload, "engine")

    default_server = AppServer()
    default_model = AppModel()
    default_engine = AppEngine()

    # 配置根覆盖：DATA_DIR 类路径与环境变量(兼容旧脚本)。
    env_data = env.get("AIRLLM_DATA_DIR", "").strip()
    resolved_data = _absolute_path(env_data, "AIRLLM_DATA_DIR") if env_data else resolved_data

    def env_bool(name: str, default: bool) -> bool:
        value = env.get(name)
        if value is None:
            return default
        normalized = value.strip().lower()
        return normalized in {"1", "true", "yes", "on"} if normalized else default

    env_port = env.get("AIRLLM_PORT", "").strip()
    port = _positive_int(env, "AIRLLM_PORT", default_server.port) if env_port else None

    toml_preload = _toml_bool(server_raw, "preload", default_server.preload)
    preload = env_bool(
        "AIRLLM_PRELOAD",
        default_server.preload if toml_preload is None else toml_preload,
    )

    env_concurrency = env.get("AIRLLM_MAX_CONCURRENT_REQUESTS", "").strip()
    concurrency_value = env_concurrency if env_concurrency else str(
        server_raw.get("max_concurrent_requests", default_server.max_concurrent_requests)
    )

    server = AppServer(
        host=env.get("AIRLLM_HOST", "").strip() or str(server_raw.get("host") or default_server.host),
        port=port or (_toml_int(server_raw, "port", default_server.port) or default_server.port),
        max_concurrent_requests=int(concurrency_value),
        preload=preload,
        api_key_file=_optional_path_value(env, server_raw, "api_key_file", resolved_data),
    )

    def model_path(key: str) -> Path | None:
        env_value = env.get(f"AIRLLM_{key.upper()}")
        if env_value and env_value.strip():
            try:
                return _absolute_path(env_value.strip(), f"AIRLLM_{key.upper()}")
            except ConfigError:
                return None
        raw_value = model_raw.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            return None
        try:
            return _absolute_path(raw_value.replace("${DATA_DIR}", str(resolved_data)), key)
        except ConfigError:
            return None

    model = AppModel(
        name=env.get("AIRLLM_MODEL_NAME", "").strip() or str(model_raw.get("name") or default_model.name),
        id=env.get("AIRLLM_MODEL_ID", "").strip() or (str(model_raw["id"]) if isinstance(model_raw.get("id"), str) else default_model.id),
        revision=env.get("AIRLLM_MODEL_REVISION", "").strip() or (str(model_raw["revision"]) if isinstance(model_raw.get("revision"), str) else default_model.revision),
        endpoint=env.get("AIRLLM_MODEL_ENDPOINT", "").strip()
        or (str(model_raw["endpoint"]) if isinstance(model_raw.get("endpoint"), str) else default_model.endpoint),
        root=env.get("AIRLLM_MODEL_ROOT", "").strip()
        or (str(model_raw["root"]) if isinstance(model_raw.get("root"), str) else default_model.root),
        dir=model_path("dir"),
        shard_dir=model_path("shard_dir"),
        manifest=model_path("manifest"),
    )

    engine = AppEngine(
        device=env.get("AIRLLM_DEVICE", "").strip() or str(engine_raw.get("device") or default_engine.device),
        max_seq_len=int(env.get("AIRLLM_MAX_SEQ_LEN", str(engine_raw.get("max_seq_len", default_engine.max_seq_len)))),
        max_output_tokens=int(env.get("AIRLLM_MAX_OUTPUT_TOKENS", str(engine_raw.get("max_output_tokens", default_engine.max_output_tokens)))),
    )

    return AppConfig(data_dir=resolved_data, server=server, model=model, engine=engine)


def _optional_path_value(environ: Mapping[str, str], raw: Mapping[str, object], key: str, data_dir: Path | None = None) -> Path | None:
    env_value = environ.get(f"AIRLLM_{key.upper()}")
    if env_value and env_value.strip():
        try:
            return _absolute_path(env_value.strip(), f"AIRLLM_{key.upper()}")
        except ConfigError:
            return None
    raw_value = raw.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        return _absolute_path(raw_value.replace("${DATA_DIR}", str(data_dir)) if data_dir is not None else raw_value, key)
    except ConfigError:
        return None


def load_api_key(config: AppConfig, environ: Mapping[str, str] | None = None) -> str:
    """解析服务 API key：环境变量优先，其次 api_key_file。"""

    env = os.environ if environ is None else environ
    from_env = env.get("AIRLLM_API_KEY", "").strip()
    if from_env:
        return from_env
    key_path = config.server.api_key_file or config.api_key_path
    try:
        if key_path is not None and key_path.is_file():
            value = key_path.read_text(encoding="utf-8").strip()
            if value:
                return value
    except OSError:
        pass
    raise ConfigError("API key 不可用：请设置 AIRLLM_API_KEY 或 api_key_file")


# 保持与 config.Settings 的桥接，避免重复实现校验。
def to_settings(config: AppConfig, api_key: str) -> Settings:
    """将应用配置转换为服务运行时使用的 Settings。"""

    return Settings(
        manifest_path=config.model.manifest,
        model_dir=config.model.dir,
        shard_dir=config.model.shard_dir,
        model_name=config.model.name,
        api_key=api_key,
        device=config.engine.device,
        max_seq_len=config.engine.max_seq_len,
        max_output_tokens=config.engine.max_output_tokens,
        max_concurrent_requests=config.server.max_concurrent_requests,
        host=config.server.host,
        port=config.server.port,
        preload=config.server.preload,
    )


__all__ = [
    "API_KEY_FILE",
    "TOML_NAME",
    "AppConfig",
    "AppEngine",
    "AppModel",
    "AppServer",
    "default_data_dir",
    "load_api_key",
    "load_app_config",
    "to_settings",
]
