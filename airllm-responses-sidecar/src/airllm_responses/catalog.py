"""内置模型目录和新用户选择模型用的元数据。"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any


class CatalogError(ValueError):
    """模型目录数据无效。"""


def load_catalog() -> dict[str, Any]:
    """读取打包在引擎内的模型目录。"""

    try:
        text = resources.files("airllm_responses").joinpath("catalog.json").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CatalogError("无法读取内置模型目录") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CatalogError("模型目录不是合法 JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise CatalogError("模型目录格式无效")
    return payload


def find_model(catalog: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """按模型 id 在目录中查找条目；找不到返回 ``None``。"""

    for entry in catalog["models"]:
        if entry.get("id") == model_id:
            return entry
    return None


def model_size_bytes(entry: dict[str, Any]) -> int | None:
    """目录条目声明的下载体积；未知返回 ``None``。"""

    value = entry.get("size_bytes")
    return value if isinstance(value, int) and value > 0 else None
