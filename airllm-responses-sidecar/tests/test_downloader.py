"""模型下载器离线可测部分：布局、清单、链接导入、目录与已存在分支。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from airllm_responses.downloader import (
    DownloadError,
    catalog_revision,
    download_catalog_model,
    install_from_local,
    installed_models,
    remove_model_alias,
    resolve_revision,
)
from airllm_responses.provision import validate_snapshot, write_approved_manifest


class DownloaderTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "model.safetensors").write_bytes(b"weights")

    def test_resolve_revision_defaults_to_none(self) -> None:
        self.assertIsNone(resolve_revision(None))
        self.assertIsNone(resolve_revision(""))
        self.assertEqual(resolve_revision("abc123"), "abc123")

    def test_installed_models_lists_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            self.assertEqual(installed_models(data_dir), [])
            model_dir = data_dir / "models" / "demo"
            self._snapshot(model_dir)
            write_approved_manifest(
                manifest=data_dir / "manifests" / "demo.json",
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                revision="abcdef",
                model_dir=model_dir,
            )
            result = installed_models(data_dir)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["alias"], "demo")
            self.assertEqual(result[0]["model_id"], "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(result[0]["revision"], "abcdef")
            self.assertTrue(result[0]["installed"])

    def test_download_skips_when_already_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            model_dir = data_dir / "models" / "demo"
            self._snapshot(model_dir)
            write_approved_manifest(
                manifest=data_dir / "manifests" / "demo.json",
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                revision="abcdef",
                model_dir=model_dir,
            )
            result = download_catalog_model(
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                alias="demo",
                data_dir=data_dir,
                token=None,
            )
            self.assertTrue(result.model_dir.is_dir())
            self.assertEqual(result.source, "huggingface")
            self.assertEqual(result.revision, "default")

    def test_download_rejects_dir_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            model_dir = data_dir / "models" / "demo"
            self._snapshot(model_dir)
            with self.assertRaises(DownloadError):
                download_catalog_model(
                    model_id="Qwen/Qwen2.5-0.5B-Instruct",
                    alias="demo",
                    data_dir=data_dir,
                    token=None,
                )

    def test_install_from_local_links_and_remove_unlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            source = data_dir / "source-model"
            self._snapshot(source)
            marker = source / "marker.txt"
            marker.write_text("keep", encoding="utf-8")

            result = install_from_local(source, "linked", data_dir)
            self.assertEqual(result.source, "local")
            linked = data_dir / "models" / "linked"
            self.assertTrue(linked.is_dir())
            # 链接模式:不复制,能读源文件。
            self.assertTrue((linked / "marker.txt").exists())
            self.assertEqual((linked / "marker.txt").read_text(encoding="utf-8"), "keep")
            self.assertTrue(marker.is_file())

            # 删除别名只断开链接,源目录与文件保持。
            remove_model_alias("linked", data_dir)
            self.assertFalse(linked.exists())
            self.assertTrue(source.is_dir())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(data_dir.joinpath("manifests").glob("linked.json")), [])

    def test_install_from_local_rejects_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            source = data_dir / "incomplete"
            source.mkdir()
            with self.assertRaises(DownloadError):
                install_from_local(source, "bad", data_dir)

    def test_catalog_revision_lookup(self) -> None:
        self.assertIsNone(catalog_revision("Qwen/Qwen2.5-0.5B-Instruct"))
        self.assertIsNone(catalog_revision("does/not-exist"))

    def test_validate_snapshot_checks_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(Exception):
                validate_snapshot(root)
            self._snapshot(root)
            self.assertEqual(validate_snapshot(root), root)


if __name__ == "__main__":
    unittest.main()
