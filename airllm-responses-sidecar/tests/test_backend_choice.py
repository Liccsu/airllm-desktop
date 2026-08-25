"""choose_backend 的选择逻辑离线测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from airllm_responses.model_runtime import choose_backend


class ChooseBackendTests(unittest.TestCase):
    def test_small_model_on_cuda_uses_direct(self) -> None:
        """模型小于可用显存时选择全量加载后端。"""

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "tiny"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with open(model_dir / "model.safetensors", "wb") as fh:
                fh.write(b"\x00" * 1024)  # 1KB 模型

            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(total_memory=16 * 1024**3),  # 16GB
            ), mock.patch("airllm_responses.model_runtime.DirectBackend") as direct, mock.patch(
                "airllm_responses.model_runtime.AirLLMBackend"
            ) as airllm:
                choose_backend(str(model_dir), "cuda:0", str(model_dir / "shards"), 512)
                direct.assert_called_once()
                airllm.assert_not_called()

    def test_large_model_uses_airllm(self) -> None:
        """模型超过显存容量时退回 AirLLM 流式加载。"""

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "big"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with open(model_dir / "model.safetensors", "wb") as fh:
                fh.seek(10 * 1024**3 - 1)  # ~10GB
                fh.write(b"\x00")

            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(total_memory=16 * 1024**3),
            ), mock.patch("airllm_responses.model_runtime.DirectBackend") as direct, mock.patch(
                "airllm_responses.model_runtime.AirLLMBackend"
            ) as airllm:
                choose_backend(str(model_dir), "cuda:0", str(model_dir / "shards"), 512)
                airllm.assert_called_once()
                direct.assert_not_called()

    def test_no_cuda_uses_airllm(self) -> None:
        """无 CUDA 时始终使用 AirLLM 后端。"""

        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary) / "tiny"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            with mock.patch("torch.cuda.is_available", return_value=False), mock.patch(
                "airllm_responses.model_runtime.DirectBackend"
            ) as direct, mock.patch("airllm_responses.model_runtime.AirLLMBackend") as airllm:
                choose_backend(str(model_dir), "cuda:0", str(model_dir / "shards"), 512)
                airllm.assert_called_once()
                direct.assert_not_called()


if __name__ == "__main__":
    unittest.main()
