"""CLI 命令离线可测部分：目录、状态与下载参数转发。"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from airllm_responses import cli
from airllm_responses.downloader import DownloadResult


class CliTests(unittest.TestCase):
    def test_catalog_json_output(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["catalog", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertGreaterEqual(len(payload["models"]), 1)
        self.assertIn("Qwen/Qwen2.5-0.5B-Instruct", [m["id"] for m in payload["models"]])

    def test_status_unreachable_returns_error(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["status", "--base-url", "http://127.0.0.1:1", "--json"])
        self.assertEqual(code, 1)
        payload = json.loads(buffer.getvalue())
        self.assertFalse(payload["healthy"])

    def test_download_forwards_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            result = DownloadResult(
                model_dir=data_dir / "models" / "demo",
                manifest=data_dir / "manifests" / "demo.json",
                revision="abc",
            )
            with patch.object(cli, "download_catalog_model", return_value=result) as mocked:
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    code = cli.main([
                        "download",
                        "--id", "Qwen/Qwen2.5-0.5B-Instruct",
                        "--alias", "demo",
                        "--data-dir", str(data_dir),
                        "--no-progress-events",
                    ])
            self.assertEqual(code, 0)
            self.assertEqual(
                mocked.call_args.kwargs["model_id"], "Qwen/Qwen2.5-0.5B-Instruct"
            )
            self.assertEqual(mocked.call_args.kwargs["alias"], "demo")
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload["alias"], "demo")
            self.assertEqual(payload["revision"], "abc")


if __name__ == "__main__":
    unittest.main()
