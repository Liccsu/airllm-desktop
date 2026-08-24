"""模型预检与审核清单读写。

模型下载由 downloader 模块负责（huggingface_hub）。本模块只处理：
- snapshot 完整性校验（config/tokenizer/权重）
- 无 secret 审核清单的原子写入与读取
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import ApprovedManifest, ConfigError, ManifestError, _absolute_path

_ALLOWED_SOURCES = ("huggingface", "modelscope", "local")


class ProvisionError(ValueError):
    """模型参数或 snapshot 内容无效。"""


_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
)
_WEIGHT_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "pytorch_model.bin",
)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvisionError(f"{name} 必填")
    return value.strip()

def validate_provision_paths(
    cache_dir: str | os.PathLike[str] | Path,
    local_dir: str | os.PathLike[str] | Path,
    manifest: str | os.PathLike[str] | Path | None = None,
) -> tuple[Path, Path, Path | None]:
    """校验下载、snapshot 和清单路径均为绝对路径。"""

    cache_path = _absolute_path(cache_dir, "cache_dir")
    local_path = _absolute_path(local_dir, "local_dir")
    manifest_path = None if manifest is None else _absolute_path(manifest, "manifest")
    return cache_path, local_path, manifest_path


def _contains_file(root: Path, names: tuple[str, ...]) -> bool:
    return any(path.is_file() for name in names for path in (root / name, *root.rglob(name)))


def validate_snapshot(local_dir: str | os.PathLike[str] | Path) -> Path:
    """检查 Transformers snapshot 的配置、tokenizer 和权重类别。"""

    root = _absolute_path(local_dir, "local_dir")
    if not root.is_dir():
        raise ProvisionError(f"snapshot 目录不存在: {root}")
    if not (root / "config.json").is_file():
        raise ProvisionError("snapshot 缺少 config.json")
    if not _contains_file(root, _TOKENIZER_FILES):
        raise ProvisionError("snapshot 缺少 tokenizer 文件")
    if not _contains_file(root, _WEIGHT_FILES):
        raise ProvisionError("snapshot 缺少支持的模型权重文件")
    return root


def write_approved_manifest(
    manifest: str | os.PathLike[str] | Path,
    model_id: str,
    revision: str,
    model_dir: str | os.PathLike[str] | Path,
    source: str = "huggingface",
) -> Path:
    """原子写入不含 token 的审核记录。"""

    manifest_path = _absolute_path(manifest, "manifest")
    model_id = _required_text(model_id, "model_id")
    revision = _required_text(revision, "revision")
    source = _required_text(source, "source")
    model_path = validate_snapshot(model_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = ApprovedManifest(
        source=source,
        model_id=model_id,
        revision=revision,
        model_dir=model_path,
        approved=True,
    ).as_dict()
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return manifest_path


__all__ = [
    "ProvisionError",
    "_ALLOWED_SOURCES",
    "_TOKENIZER_FILES",
    "_WEIGHT_FILES",
    "validate_provision_paths",
    "validate_snapshot",
    "write_approved_manifest",
]
