"""构建桌面应用捆绑资源：engine wheel、airllm wheel 与 uv 运行时。

用法::

    python tools/build_bundle.py [--output desktop/src-tauri/resources]

产物由 Tauri 安装器捆绑，首次启动时由引导流程安装到数据目录。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDECAR_DIR = REPO_ROOT / "airllm-responses-sidecar"
AIRLLM_SRC = REPO_ROOT / "airllm" / "air_llm"
DEFAULT_OUTPUT = REPO_ROOT / "desktop" / "src-tauri" / "resources"


def run(command: list[str], cwd: Path) -> None:
    print(f"> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def build_engine_wheel(out_dir: Path) -> Path:
    """用 uv 构建 sidecar wheel 到目标目录。"""

    run(["uv", "build", "--wheel", "--out-dir", str(out_dir)], cwd=SIDECAR_DIR)
    for residual in (SIDECAR_DIR / "build", SIDECAR_DIR / "src" / "airllm_responses.egg-info"):
        if residual.exists():
            shutil.rmtree(residual, ignore_errors=True)
    wheels = sorted(out_dir.glob("airllm_responses_sidecar-*.whl"))
    if not wheels:
        raise RuntimeError("engine wheel 构建失败")
    return wheels[0]


def ensure_airllm_source() -> None:
    """确保上游 AirLLM 源码存在；缺失时自动克隆（构建依赖，不作为仓库内容提交）。"""

    if (AIRLLM_SRC / "airllm" / "__init__.py").is_file():
        return
    if AIRLLM_SRC.parent.exists() and any(AIRLLM_SRC.parent.iterdir()):
        raise RuntimeError(
            f"AirLLM 源码目录非空但缺少 airllm 包: {AIRLLM_SRC}；"
            "请清理后重试或手动放置上游源码"
        )
    AIRLLM_SRC.parent.mkdir(parents=True, exist_ok=True)
    # CI（GitHub 运行器）网络直连官方源最快；本地开发优先 gh-proxy 国内加速。
    ghproxy_url = "https://gh-proxy.org/https://github.com/lyogavin/airllm.git"
    official_url = "https://github.com/lyogavin/airllm.git"
    candidates = (
        (official_url, ghproxy_url)
        if os.environ.get("CI") == "true"
        else (ghproxy_url, official_url)
    )
    for clone_url in candidates:
        try:
            run(
                ["git", "clone", "--depth", "1", clone_url, str(AIRLLM_SRC.parent)],
                cwd=REPO_ROOT,
            )
            return
        except subprocess.CalledProcessError:
            import shutil

            shutil.rmtree(AIRLLM_SRC.parent, ignore_errors=True)
    raise RuntimeError(f"AirLLM 源码克隆失败（已尝试 gh-proxy 与官方源）: {AIRLLM_SRC.parent}")


def build_airllm_wheel(out_dir: Path) -> Path:
    """把 air_llm 源码目录打为 airllm wheel（包名 airllm）。"""

    ensure_airllm_source()
    if not (AIRLLM_SRC / "airllm" / "__init__.py").is_file():
        raise RuntimeError(f"找不到 AirLLM 源码: {AIRLLM_SRC}")
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        air_llm_copy = tmp / "air_llm"
        shutil.copytree(
            AIRLLM_SRC,
            air_llm_copy,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "*.egg-info"),
        )
        pyproject = air_llm_copy / "pyproject.toml"
        pyproject.write_text(
            "\n".join(
                [
                    "[build-system]",
                    'requires = ["setuptools>=69"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                    "[project]",
                    'name = "airllm"',
                    'version = "1.0.0"',
                    'description = "AirLLM 内存节俭本地推理库（桌面捆绑版）"',
                    'requires-python = ">=3.11,<3.12"',
                    'dependencies = []',
                    "",
                    "[tool.setuptools]",
                    'packages = ["airllm", "airllm.persist"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )
        run(["uv", "build", "--wheel", "--out-dir", str(out_dir)], cwd=air_llm_copy)
    wheels = sorted(out_dir.glob("airllm-*.whl"))
    if not wheels:
        raise RuntimeError("airllm wheel 构建失败")
    return wheels[0]


def find_uv_exe() -> Path:
    """定位本机 uv.exe 作为捆绑的运行时安装器。"""

    from shutil import which

    located = which("uv")
    if located:
        return Path(located)
    raise RuntimeError("未找到 uv.exe；请先安装 uv（https://docs.astral.sh/uv/）")


def build_bundle(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in output.iterdir():
        if path.is_file():
            path.unlink()

    build_engine_wheel(output)
    build_airllm_wheel(output)
    shutil.copy2(find_uv_exe(), output / "uv.exe")
    shutil.copy2(SIDECAR_DIR / "src" / "airllm_responses" / "catalog.json", output / "catalog.json")
    gitignore = output / ".gitignore"
    if gitignore.exists():
        gitignore.unlink()
    print(
        "\n捆绑资源:",
        *sorted(p.name for p in output.iterdir()),
        sep="\n  - ",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建桌面应用捆绑资源")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    build_bundle(args.output.resolve())
    return 0


if __name__ == "__main__":
    import sys

    # Windows CI 默认编码可能不含中文；统一使用 UTF-8 输出，避免打印崩溃。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    raise SystemExit(main())
