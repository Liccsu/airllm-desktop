"""模型预检、审核清单与校验工具测试（离线）。"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from airllm_responses.config import ConfigError, ManifestError, Settings, load_approved_manifest
from airllm_responses.provision import (
    ProvisionError,
    validate_provision_paths,
    validate_snapshot,
    write_approved_manifest,
)


class ProvisionTests(unittest.TestCase):
    def _snapshot(self, root: Path, *, weights: bool = True) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        if weights:
            (root / "model.safetensors").write_bytes(b"weights")

    def test_settings_reads_environment_and_rejects_relative_paths(self) -> None:
        settings = Settings.from_env({
            "AIRLLM_API_KEY": "key",
            "AIRLLM_MODEL_MANIFEST": "C:/runtime/manifest.json",
            "AIRLLM_MODEL_DIR": "C:/runtime/models/demo",
            "AIRLLM_SHARD_DIR": "C:/runtime/shards/demo",
            "AIRLLM_MAX_SEQ_LEN": "1024",
            "AIRLLM_MAX_OUTPUT_TOKENS": "16",
            "AIRLLM_PRELOAD": "true",
        })
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.max_seq_len, 1024)
        self.assertTrue(settings.preload)
        with self.assertRaises(ConfigError):
            Settings.from_env({"AIRLLM_API_KEY": "key", "AIRLLM_MODEL_DIR": "models/demo"})

    def test_paths_must_be_absolute(self) -> None:
        with self.assertRaises(ValueError):
            validate_provision_paths(Path("/tmp/cache"), Path("models/demo"))

    def test_snapshot_requires_config_tokenizer_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ProvisionError):
                validate_snapshot(root)
            self._snapshot(root, weights=False)
            with self.assertRaises(ProvisionError):
                validate_snapshot(root)
            self._snapshot(root)
            self.assertEqual(validate_snapshot(root), root)

    def test_manifest_roundtrip_and_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            self._snapshot(model_dir)
            manifest = root / "m.json"
            write_approved_manifest(
                manifest=manifest,
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                revision="abc123",
                model_dir=model_dir,
                source="huggingface",
            )
            loaded = load_approved_manifest(manifest)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.source, "huggingface")
            self.assertEqual(loaded.model_id, "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(loaded.model_dir, model_dir)
            # 旧格式（modelscope）仍然可读：白名单兼容。
            raw = json.loads(manifest.read_text(encoding="utf-8"))
            raw["source"] = "modelscope"
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_approved_manifest(manifest).source, "modelscope")
            # 未知来源拒绝。
            raw["source"] = "mystery"
            manifest.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ManifestError):
                load_approved_manifest(manifest)

    def test_manifest_missing_or_unavailable_model_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "m.json"
            self.assertIsNone(load_approved_manifest(manifest))
            model_dir = root / "gone"
            manifest.write_text(
                json.dumps({
                    "source": "huggingface",
                    "model_id": "demo/model",
                    "revision": "r1",
                    "model_dir": str(model_dir),
                    "approved": True,
                }),
                encoding="utf-8",
            )
            with self.assertRaises(ManifestError):
                load_approved_manifest(manifest)

    def test_manifest_contains_no_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            self._snapshot(model_dir)
            manifest = root / "m.json"
            write_approved_manifest(
                manifest=manifest,
                model_id="demo/model",
                revision="r1",
                model_dir=model_dir,
            )
            text = manifest.read_text(encoding="utf-8")
            self.assertNotIn("token", text.lower().replace("tokenizer", ""))


if __name__ == "__main__":
    unittest.main()
