"""应用配置加载、API key 解析与 Settings 桥接测试。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from airllm_responses.app_config import (
    AppConfig,
    AppServer,
    default_data_dir,
    load_api_key,
    load_app_config,
    to_settings,
)
from airllm_responses.config import ConfigError, Settings


class AppConfigTests(unittest.TestCase):
    def test_legacy_environment_shapes_settings(self) -> None:
        """旧脚本设置的环境变量仍然生效（兼容迁移）。"""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "m.json"
            manifest.write_text("{}", encoding="utf-8")
            config = load_app_config(
                data_dir=root / "data",
                environ={
                    "AIRLLM_API_KEY": "key",
                    "AIRLLM_MODEL_ID": "Qwen/Qwen2.5-0.5B-Instruct",
                    "AIRLLM_MODEL_NAME": "local",
                    "AIRLLM_HOST": "0.0.0.0",
                    "AIRLLM_PORT": "9000",
                    "AIRLLM_DEVICE": "cuda:0",
                    "AIRLLM_MAX_SEQ_LEN": "1024",
                    "AIRLLM_MAX_OUTPUT_TOKENS": "32",
                    "AIRLLM_PRELOAD": "true",
                },
            )
            self.assertEqual(config.server.host, "0.0.0.0")
            self.assertEqual(config.server.port, 9000)
            self.assertTrue(config.server.preload)
            self.assertEqual(config.engine.device, "cuda:0")
            self.assertEqual(config.engine.max_seq_len, 1024)
            self.assertEqual(config.model.id, "Qwen/Qwen2.5-0.5B-Instruct")
            self.assertEqual(config.model.name, "local")
            settings = to_settings(config, "key")
            self.assertIsInstance(settings, Settings)
            self.assertEqual(settings.host, "0.0.0.0")
            self.assertEqual(settings.port, 9000)

    def test_toml_sections_and_path_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (root / "app.toml").write_text(
                "\n".join(
                    [
                        "[server]",
                        'host = "127.0.0.1"',
                        "port = 8123",
                        "preload = true",
                        'api_key_file = "${DATA_DIR}/key.txt"',
                        "[model]",
                        'name = "chat"',
                        'id = "Qwen/Qwen3-1.7B"',
                        'dir = "${DATA_DIR}/models/chat"',
                        "[engine]",
                        'device = "cuda:1"',
                        "max_seq_len = 2048",
                    ]
                ),
                encoding="utf-8",
            )
            config = load_app_config(config_path=root / "app.toml", data_dir=data_root)
            self.assertEqual(config.server.port, 8123)
            self.assertTrue(config.server.preload)
            # 无环境变量时路径以 DATA_DIR 为根展开。
            self.assertEqual(config.server.api_key_file, data_root / "key.txt")
            self.assertEqual(config.model.dir, data_root / "models" / "chat")
            self.assertEqual(config.engine.device, "cuda:1")
            self.assertEqual(config.engine.max_seq_len, 2048)

    def test_api_key_file_and_environment_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key_file = root / "api_key.txt"
            key_file.write_text("file-key\n", encoding="utf-8")
            config = AppConfig(
                data_dir=root,
                server=AppServer(api_key_file=key_file),
            )
            self.assertEqual(load_api_key(config, environ={}), "file-key")
            self.assertEqual(
                load_api_key(config, environ={"AIRLLM_API_KEY": "env-key"}),
                "env-key",
            )

    def test_missing_api_key_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_app_config(data_dir=Path(temporary))
            with self.assertRaises(ConfigError):
                load_api_key(config, environ={})

    def test_default_data_dir_is_absolute(self) -> None:
        path = default_data_dir()
        self.assertTrue(path.is_absolute())


if __name__ == "__main__":
    unittest.main()
