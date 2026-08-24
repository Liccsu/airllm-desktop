"""Sidecar 服务配置与已批准模型清单。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """环境配置或模型清单无效。"""


class ManifestError(ConfigError):
    """模型清单不是可供服务使用的审核记录。"""


def _absolute_path(value: str | os.PathLike[str], name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ConfigError(f"{name} 必须是绝对路径")
    return path

def validate_absolute_path(value: str | os.PathLike[str] | Path, name: str) -> Path:
    """校验并返回绝对路径。"""

    return _absolute_path(value, name)


def _optional_path(environ: Mapping[str, str], name: str) -> Path | None:
    value = environ.get(name)
    return None if value is None or not value.strip() else _absolute_path(value, name)


def _positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name, str(default))
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是正整数") from exc
    if number <= 0:
        raise ConfigError(f"{name} 必须是正整数")
    return number


def _boolean(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} 必须是 true 或 false")


@dataclass(frozen=True)
class ApprovedManifest:
    """通过下载审核后写入的、无 secret 模型记录。"""

    source: str
    model_id: str
    revision: str
    model_dir: Path
    approved: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "model_id": self.model_id,
            "revision": self.revision,
            "model_dir": str(self.model_dir),
            "approved": self.approved,
        }


def load_approved_manifest(path: str | os.PathLike[str] | Path) -> ApprovedManifest | None:
    """读取审核清单；不存在的清单返回 ``None``，无效清单抛出 ManifestError。"""

    manifest_path = _absolute_path(path, "AIRLLM_MODEL_MANIFEST")
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"无法读取模型清单: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("模型清单必须是 JSON 对象")
    required = {"source", "model_id", "revision", "model_dir", "approved"}
    if set(payload) != required:
        raise ManifestError("模型清单字段必须恰好为 source、model_id、revision、model_dir、approved")
    if payload["source"] not in {"huggingface", "modelscope", "local"} or payload["approved"] is not True:
        raise ManifestError("模型清单来源未知或未获批准")
    for field in ("model_id", "revision", "model_dir"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ManifestError(f"模型清单字段 {field} 无效")
    model_dir = _absolute_path(payload["model_dir"], "model_dir")
    if not model_dir.is_dir():
        raise ManifestError("模型清单中的 model_dir 不存在或不是目录")
    return ApprovedManifest(
        source=payload["source"],
        model_id=payload["model_id"],
        revision=payload["revision"],
        model_dir=model_dir,
    )


@dataclass(frozen=True)
class Settings:
    """服务进程使用的环境配置。"""

    manifest_path: Path | None
    model_dir: Path | None
    shard_dir: Path | None
    model_name: str
    api_key: str
    device: str
    max_seq_len: int
    max_output_tokens: int
    max_concurrent_requests: int
    host: str
    port: int
    preload: bool

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        api_key = env.get("AIRLLM_API_KEY", "").strip()
        if not api_key:
            raise ConfigError("AIRLLM_API_KEY 必填")
        concurrency = _positive_int(env, "AIRLLM_MAX_CONCURRENT_REQUESTS", 1)
        if concurrency != 1:
            raise ConfigError("AIRLLM_MAX_CONCURRENT_REQUESTS 固定为 1")
        port = _positive_int(env, "AIRLLM_PORT", 8000)
        if port > 65535:
            raise ConfigError("AIRLLM_PORT 超出范围")
        return cls(
            manifest_path=_optional_path(env, "AIRLLM_MODEL_MANIFEST"),
            model_dir=_optional_path(env, "AIRLLM_MODEL_DIR"),
            shard_dir=_optional_path(env, "AIRLLM_SHARD_DIR"),
            model_name=env.get("AIRLLM_MODEL_NAME", "airllm-local").strip() or "airllm-local",
            api_key=api_key,
            device=env.get("AIRLLM_DEVICE", "cuda:0").strip() or "cuda:0",
            max_seq_len=_positive_int(env, "AIRLLM_MAX_SEQ_LEN", 512),
            max_output_tokens=_positive_int(env, "AIRLLM_MAX_OUTPUT_TOKENS", 128),
            max_concurrent_requests=concurrency,
            host=env.get("AIRLLM_HOST", "127.0.0.1").strip() or "127.0.0.1",
            port=port,
            preload=_boolean(env, "AIRLLM_PRELOAD", False),
        )

    @property
    def approved_manifest(self) -> ApprovedManifest | None:
        if self.manifest_path is None:
            return None
        try:
            return load_approved_manifest(self.manifest_path)
        except ManifestError:
            return None

    @property
    def ready(self) -> bool:
        return self.approved_manifest is not None

    def load_manifest(self) -> ApprovedManifest | None:
        """返回当前审核记录，供服务启动路径使用。"""

        if self.manifest_path is None:
            return None
        return load_approved_manifest(self.manifest_path)


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """从环境变量创建服务配置。"""

    return Settings.from_env(environ)


__all__ = [
    "ApprovedManifest",
    "ConfigError",
    "ManifestError",
    "Settings",
    "load_approved_manifest",
    "load_settings",
    "validate_absolute_path",
]
