"""模型下载、预检与审核发布（基于 huggingface_hub，支持镜像源与链接导入）。

下载流程（对用户完全透明，桌面应用展示进度）：

1. 解析模型 id 与 revision（为空时使用仓库默认分支，成功后从缓存固定实际 commit）；
2. 通过 ``list_repo_tree`` 获取文件清单（用于总大小估算，失败不阻塞下载）；
3. 使用 ``snapshot_download`` 下载到 staging 目录，进度通过 ``tqdm_class`` 逐文件上报；
4. 校验 snapshot 内容，以 staging 目录原子发布到 ``models/<alias>``；
5. 写入无 secret 的审核清单 ``manifests/<alias>.json``。

导入：

- ``install_from_local`` 以目录链接（Windows junction / POSIX symlink）指向本地目录，
  不复制模型文件；删除别名时断开链接本身，源目录保持不动。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from pathlib import Path

from .catalog import CatalogError, find_model, load_catalog
from .config import ConfigError, _absolute_path
from .events import emit, emit_error
from .provision import (
    ProvisionError,
    validate_snapshot,
    write_approved_manifest,
)

ProgressCallback = Callable[[str, int, int, str], None]


class DownloadError(ProvisionError):
    """模型下载或发布失败。"""


@dataclass(frozen=True)
class DownloadResult:
    model_dir: Path
    manifest: Path
    revision: str
    source: str = "huggingface"


def resolve_revision(revision: str | None) -> str | None:
    """解析 revision：留空表示使用仓库默认分支。"""

    if revision is None:
        return None
    return revision.strip() or None


def _hub_cache_meta(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / ("models--" + repo_id.replace("/", "--"))


def _cached_commit(cache_dir: Path, repo_id: str, revision: str | None) -> str | None:
    """读取 huggingface_hub 缓存中的实际 commit hash。"""

    meta = _hub_cache_meta(cache_dir, repo_id)
    refs_dir = meta / "refs"
    try:
        if refs_dir.is_dir():
            for ref_file in sorted(refs_dir.iterdir()):
                value = ref_file.read_text(encoding="utf-8").strip()
                if value:
                    return value
    except OSError:
        pass
    return None


def _model_files(model_id: str, revision: str | None, endpoint: str | None) -> list[dict[str, Any]] | None:
    """查询仓库文件清单（含大小）；任何失败返回 ``None``（未知大小，不阻塞下载）。"""

    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    try:
        api = HfApi(endpoint=endpoint or None)
        entries = list(api.list_repo_tree(model_id, repo_type="model", revision=revision, recursive=True))
    except Exception:
        return None
    result: list[dict[str, Any]] = []
    for entry in entries:
        kind = getattr(entry, "type", None) or entry.get("type") if isinstance(entry, dict) else getattr(entry, "type", None)
        path = getattr(entry, "path", None) or (entry.get("path") if isinstance(entry, dict) else None)
        if not path or str(kind) not in {"file", "lfs"}:
            continue
        size = getattr(entry, "size", None) or (entry.get("size") if isinstance(entry, dict) else None) or 0
        result.append({"path": str(path), "size": int(size or 0)})
    return result or None


def _download_with_progress(
    model_id: str,
    revision: str | None,
    cache_dir: Path,
    staging_dir: Path,
    token: str | None,
    endpoint: str | None,
    progress: ProgressCallback,
    workers: int | None = None,
) -> None:
    """使用 snapshot_download 下载到 staging，逐文件回调进度。"""

    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    file_progress: dict[str, tuple[int, int]] = {}

    class _ProgressTqdm(tqdm):
        def __init__(self, **kwargs: Any) -> None:
            # huggingface_hub 传入 desc(文件名)、total、unit 等；全部透传。
            total = kwargs.get("total")
            desc = str(kwargs.get("desc") or "")
            file_progress[desc] = (0, int(total or 0))
            progress(desc, 0, int(total or 0), "started")
            super().__init__(**kwargs)

        def update(self, n: int = 1) -> None:
            super().update(n)
            file = str(getattr(self, "desc", "") or "")
            done = int(self.n)
            total = int(self.total or 0)
            file_progress[file] = (done, total)
            progress(file, done, total, "progress")

        def close(self) -> None:
            file = str(getattr(self, "desc", "") or "")
            done, total = file_progress.get(file, (int(self.n), int(self.total or 0)))
            super().close()
            progress(file, done, total, "done")

    try:
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            revision=revision,
            cache_dir=str(cache_dir),
            local_dir=str(staging_dir),
            token=token or None,
            endpoint=endpoint or None,
            tqdm_class=_ProgressTqdm,
            max_workers=workers,
        )
    except Exception as exc:
        raise DownloadError(f"模型下载失败: {type(exc).__name__}") from exc


def _sizes_match(staging: Path, files: list[dict[str, Any]] | None) -> bool:
    """快照内每个文件的大小与仓库清单一致（含 LFS 大文件）。"""

    if not files:
        return False
    for entry in files:
        path = staging / entry["path"]
        expected = int(entry["size"] or 0)
        if expected <= 0:
            continue
        try:
            if not path.is_file() or path.stat().st_size != expected:
                return False
        except OSError:
            return False
    return True


def download_catalog_model(
    model_id: str,
    alias: str | None,
    data_dir: Path,
    token: str | None,
    catalog: Mapping[str, Any] | None = None,
    revision: str | None = None,
    endpoint: str | None = None,
    model_root: Path | None = None,
    workers: int | None = None,
    progress: ProgressCallback | None = None,
) -> DownloadResult:
    """下载模型并审核发布到应用数据目录。

    ``revision`` 为空时使用仓库默认分支并把实际 commit hash 写入清单；
    ``endpoint`` 为镜像源地址（如 https://hf-mirror.com），为空时使用官方源。
    """

    resolved_revision = resolve_revision(revision)
    if endpoint is not None and endpoint.strip() not in ("", "https://huggingface.co"):
        # 镜像源关闭 xet 后端(重建请求会打到官方 CAS 返回 401),强制 HTTP 分片下载。
        # 须在首次 import huggingface_hub 之前设置:该常量在模块加载时固化。
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    emit("download.started", model_id=model_id, revision=resolved_revision, endpoint=endpoint)

    # 进度事件发送累计已下载字节(跨文件)与整体大小；文件列表不可用时整体大小未知。
    _completed: dict[str, int] = {"bytes": 0}

    def on_progress(file: str, done: int, total: int, stage: str) -> None:
        if stage == "done":
            _completed["bytes"] += total or done or 0
        current = done if stage == "progress" else 0
        emit(
            "download.progress",
            file=file,
            done_bytes=_completed["bytes"] + current,
            total_bytes=total_bytes,
            stage=stage,
        )
        if progress is not None:
            progress(file, done, total, stage)

    notify = on_progress

    if alias is None:
        alias = model_id.replace("/", "-")

    cache_dir = data_dir / "cache" / "hub"
    final_dir = (model_root if model_root is not None else data_dir / "models") / alias
    manifest_path = data_dir / "manifests" / f"{alias}.json"

    if final_dir.exists():
        if manifest_path.is_file():
            emit("download.skipped", model_id=model_id, alias=alias, reason="exists")
            return DownloadResult(final_dir, manifest_path, resolved_revision or "default")
        raise DownloadError(f"模型目录已存在但缺少清单: {final_dir}")

    files = _model_files(model_id, resolved_revision, endpoint)
    total_bytes = sum(int(f["size"] or 0) for f in files) if files else None
    emit("download.files", model_id=model_id, files_count=len(files) if files else 0, total_bytes=total_bytes)

    final_dir.parent.mkdir(parents=True, exist_ok=True)
    # 清理该别名先前失败/取消留下的 staging 残留。
    for stale in final_dir.parent.glob(f".{final_dir.name}.staging-*"):
        shutil.rmtree(stale, ignore_errors=True)
    staging = final_dir.parent / f".{final_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=False)
        _download_with_progress(
            model_id, resolved_revision, cache_dir, staging, token, endpoint, notify, workers
        )
        validate_snapshot(staging)
        staging.replace(final_dir)
    except Exception as exc:
        # 镜像源不支持受限仓库（403/503）时，自动降级到官方源重试一次。
        if endpoint is not None and endpoint.strip() not in ("", "https://huggingface.co"):
            lowered = str(exc).lower()
            if "403" in lowered or "503" in lowered or "gated" in lowered or "forbidden" in lowered:
                shutil.rmtree(staging, ignore_errors=True)
                emit(
                    "download.warning",
                    message="镜像源不支持该模型（受限仓库或鉴权失败），自动切换官方源重试",
                    model_id=model_id,
                    alias=alias,
                )
                try:
                    result = download_catalog_model(
                        model_id=model_id,
                        alias=alias,
                        data_dir=data_dir,
                        token=token,
                        catalog=catalog,
                        revision=revision,
                        endpoint=None,  # 官方源
                        model_root=model_root,
                        workers=workers,
                        progress=progress,
                    )
                    return result
                except DownloadError as retry_exc:
                    raise DownloadError(
                        f"镜像源与官方源均下载失败：{retry_exc}"
                    ) from exc
        # 兜底：hub 收尾/校验失败（如断流、镜像 401）但快照文件与清单大小一致时，
        # 视为下载成功继续发布，让用户对网络抖动镜像更鲁棒。
        snap_ok = False
        if isinstance(exc, DownloadError):
            try:
                validate_snapshot(staging)
                snap_ok = _sizes_match(staging, files)
            except Exception:
                snap_ok = False
        if snap_ok:
            emit(
                "download.warning",
                message="下载收尾校验失败，已按完整快照继续",
                model_id=model_id,
                alias=alias,
            )
        else:
            shutil.rmtree(staging, ignore_errors=True)
            message = str(exc) if isinstance(exc, DownloadError) else f"模型下载或校验失败: {type(exc).__name__}"
            emit_error("download.failed", message, model_id=model_id, alias=alias)
            raise DownloadError(message) from exc

    actual_revision = _cached_commit(cache_dir, model_id, resolved_revision) or resolved_revision or "default"
    try:
        write_approved_manifest(
            manifest=manifest_path,
            model_id=model_id,
            revision=actual_revision,
            model_dir=final_dir,
            source="huggingface",
        )
    except Exception:
        shutil.rmtree(final_dir, ignore_errors=True)
        raise

    emit(
        "download.done",
        model_id=model_id,
        alias=alias,
        revision=actual_revision,
        model_dir=str(final_dir),
        manifest=str(manifest_path),
    )
    return DownloadResult(final_dir, manifest_path, actual_revision)


def _link_directory(source: Path, destination: Path) -> None:
    """以目录链接指向 source（Windows junction / POSIX symlink），不复制数据。"""

    try:
        os.symlink(source, destination, target_is_directory=True)
        return
    except OSError:
        if os.name == "nt":
            # Windows 无符号链接权限时使用 junction（mklink /J 不需特权）。
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
                capture_output=True,
            )
            if result.returncode != 0:
                raise DownloadError("创建目录链接失败（mklink /J）")
            return
        raise


def install_from_local(
    source_dir: str | os.PathLike[str] | Path,
    alias: str,
    data_dir: Path,
    model_root: Path | None = None,
    progress: ProgressCallback | None = None,
) -> DownloadResult:
    """以链接形式安装本地模型（不复制文件；删除别名仅断开链接）。"""

    source = _absolute_path(source_dir, "source_dir")
    try:
        validate_snapshot(source)
    except ProvisionError as exc:
        raise DownloadError(f"源目录不是完整的模型 snapshot: {exc}") from exc
    final_dir = (model_root if model_root is not None else data_dir / "models") / alias
    # 在创建链接前判断路径是否相同（junction 创建后 resolve 会跟随链接，不能再用于比较）。
    same_location = os.path.normcase(os.path.abspath(final_dir)) == os.path.normcase(os.path.abspath(source))
    if same_location:
        # 模型已在模型下载目录的正确位置：无需创建链接，直接注册。
        emit(
            "download.warning",
            message="模型已在模型下载目录中，直接注册",
            alias=alias,
        )
    else:
        if final_dir.exists():
            raise DownloadError(f"模型目录已存在: {final_dir}")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            _link_directory(source, final_dir)
        except Exception as exc:
            raise DownloadError("创建模型目录链接失败") from exc
        # 校验链接后的读取路径，写失败时断开链接。
        try:
            validate_snapshot(final_dir)
        except Exception as exc:
            _unlink_directory(final_dir)
            raise DownloadError("链接后的模型 snapshot 无效") from exc
    managed = not same_location
    manifest_path = data_dir / "manifests" / f"{alias}.json"
    write_approved_manifest(
        manifest=manifest_path,
        model_id=f"local:{source}",
        revision="local",
        model_dir=final_dir,
        source="local",
        managed=managed,
    )
    emit("download.done", model_id=f"local:{source}", alias=alias, model_dir=str(final_dir), manifest=str(manifest_path))
    return DownloadResult(final_dir, manifest_path, "local", source="local")


def _unlink_directory(path: Path) -> None:
    """断开目录链接（不删除源）；失败时回退完整删除。"""

    try:
        path.rmdir()
        return
    except OSError:
        pass
    shutil.rmtree(path, ignore_errors=True)


def remove_model_alias(alias: str, data_dir: Path, model_root: Path | None = None) -> None:
    """删除模型别名：断开链接（local 源）或删除数据目录，并移除清单与分片。

    模型目录由应用管理（下载/链接）时删除；外部目录（managed=False）只删清单与分片。
    """

    import json as _json

    manifest_path = data_dir / "manifests" / f"{alias}.json"
    managed = True
    try:
        payload = _json.loads(manifest_path.read_text(encoding="utf-8"))
        managed = payload.get("managed", True) is True
    except Exception:
        pass
    model_dir = (model_root if model_root is not None else data_dir / "models") / alias
    if managed and model_dir.exists():
        _unlink_directory(model_dir)
    for path in (
        data_dir / "shards" / alias,
        manifest_path,
    ):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink(missing_ok=True)


def installed_models(data_dir: Path) -> list[dict[str, Any]]:
    """列出数据目录中已完成发布的模型。"""

    manifests_dir = data_dir / "manifests"
    results: list[dict[str, Any]] = []
    if not manifests_dir.is_dir():
        return results
    for manifest_file in sorted(manifests_dir.glob("*.json")):
        try:
            payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        model_dir = Path(str(payload.get("model_dir")))
        results.append(
            {
                "alias": manifest_file.stem,
                "model_id": payload.get("model_id"),
                "revision": payload.get("revision"),
                "model_dir": str(model_dir),
                "installed": model_dir.exists(),
                "source": payload.get("source"),
            }
        )
    return results


def catalog_revision(model_id: str, catalog: Mapping[str, Any] | None = None) -> str | None:
    """从目录条目读取推荐的 revision（目录可能不指定）。"""

    catalog = load_catalog() if catalog is None else catalog
    entry = find_model(catalog, model_id)
    if entry is None:
        return None
    value = entry.get("revision")
    return value if isinstance(value, str) and value.strip() else None


__all__ = [
    "DownloadError",
    "DownloadResult",
    "ProgressCallback",
    "catalog_revision",
    "download_catalog_model",
    "install_from_local",
    "installed_models",
    "remove_model_alias",
    "resolve_revision",
]
