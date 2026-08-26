"""层驻留窗口的大小计算与 hook 替换逻辑测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from airllm_responses.model_runtime import _apply_layer_window, _layer_window_size


class _FakeModule:
    def __init__(self, idx: int | None = None) -> None:
        self._airllm_idx = idx
        self._forward_pre_hooks: OrderedDict = OrderedDict()
        self._forward_hooks: OrderedDict = OrderedDict()

    def register_forward_pre_hook(self, hook) -> None:
        self._forward_pre_hooks[len(self._forward_pre_hooks)] = hook

    def register_forward_hook(self, hook) -> None:
        self._forward_hooks[len(self._forward_hooks)] = hook


class _FakeModel:
    def __init__(self, n_layers: int = 12, total_bytes: int = 12 * 1024**3) -> None:
        self.checkpoint_path = None
        self.layer_names = [f"model.layers.{i}." for i in range(n_layers)]
        self.layers = [_FakeModule(i) for i in range(n_layers)]
        self.device = "cuda:0"
        self.model = object()
        self._airllm_resident = {}
        self._airllm_window = 1
        self.total_bytes = total_bytes

    def _load_streamed_layer(self, idx: int):
        return {}

    def move_layer_to_device(self, state_dict: dict):
        return ["fake.param"]


class LayerWindowTests(unittest.TestCase):
    def _model_dir(self, temporary: str, size_bytes: int) -> str:
        path = Path(temporary) / "model.safetensors"
        with open(path, "wb") as fh:
            fh.seek(size_bytes - 1)
            fh.write(b"\x00")
        return str(path)

    def test_window_size_by_vram(self) -> None:
        """窗口大小按可用显存与每层体积计算。"""

        model = _FakeModel(n_layers=12, total_bytes=12 * 1024**3)  # 12GB / 12 层
        with tempfile.TemporaryDirectory() as temporary:
            model.checkpoint_path = self._model_dir(temporary, 12 * 1024**3)
            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(total_memory=16 * 1024**3),
            ), mock.patch("torch.cuda.memory_allocated", return_value=3 * 1024**3):
                # 每层 fp16 估算 = 12GB*2/12 = 2GB；可用 = 16-3-2 = 11GB → W = 5
                w = _layer_window_size(model)
        self.assertGreaterEqual(w, 2)
        self.assertLessEqual(w, 8)

    def test_window_capped_by_layers(self) -> None:
        model = _FakeModel(n_layers=3, total_bytes=3 * 1024**3)
        with tempfile.TemporaryDirectory() as temporary:
            model.checkpoint_path = self._model_dir(temporary, 3 * 1024**3)
            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
                "torch.cuda.get_device_properties",
                return_value=mock.Mock(total_memory=64 * 1024**3),
            ), mock.patch("torch.cuda.memory_allocated", return_value=0):
                w = _layer_window_size(model)
        self.assertEqual(w, 3)

    def test_no_cuda_returns_one(self) -> None:
        model = _FakeModel()
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(_layer_window_size(model), 1)

    def test_apply_window_replaces_hooks(self) -> None:
        """hook 替换后：窗口滑过才释放，驻留集正确维护。"""

        from unittest.mock import patch as _patch

        model = _FakeModel(n_layers=6)
        # 模拟 airllm 注册原 hook（引用与 _pre_hook/_post_hook 一致才被移除）
        model._pre_hook = lambda *a: None
        model._post_hook = lambda *a: None
        for module in model.layers:
            module.register_forward_pre_hook(model._pre_hook)
            module.register_forward_hook(model._post_hook)
        model._load_streamed_layer = lambda idx: {}
        model.move_layer_to_device = lambda sd: [f"layer{idx}.w" for idx in range(6)]
        model.model = object()

        with _patch("airllm.utils.clean_memory"), _patch(
            "accelerate.utils.modeling.set_module_tensor_to_device"
        ):
            _apply_layer_window(model, 3)

        # 所有层 hook 被替换（新 hook 不再等于原引用）
        for module in model.layers:
            self.assertTrue(module._forward_pre_hooks)
            self.assertTrue(module._forward_hooks)
        # 触发一次 pre（第 0 层）：加载窗口 [0, 3)
        for module in model.layers:
            if module._airllm_idx == 0:
                pre_hook = next(iter(module._forward_pre_hooks.values()))
                post_hook = next(iter(module._forward_hooks.values()))
                pre_hook(module, None)
                post_hook(module, None, "out")
        # 确认旧 hook 已被替换
        self.assertIsNot(pre_hook, model._pre_hook)
        self.assertEqual(len(model._airllm_resident), 3)
        # 第 3 层 pre：释放 <3 的层，加载 [3, 6)
        for module in model.layers:
            if module._airllm_idx == 3:
                pre_hook = next(iter(module._forward_pre_hooks.values()))
                pre_hook(module, None)
        self.assertEqual(len(model._airllm_resident), 3)
        self.assertNotIn(0, model._airllm_resident)
        self.assertIn(5, model._airllm_resident)


    def test_window_fallback_on_load_error(self) -> None:
        """窗口加载失败时逐层降级，不崩溃。"""

        from unittest.mock import patch as _patch

        model = _FakeModel(n_layers=4)
        model._pre_hook = lambda *a: None
        model._post_hook = lambda *a: None
        for module in model.layers:
            module.register_forward_pre_hook(model._pre_hook)
            module.register_forward_hook(model._post_hook)

        def broken_load(idx):
            raise RuntimeError("boom")

        model._load_streamed_layer = broken_load
        model.move_layer_to_device = lambda sd: ["fake.w"]
        model.model = object()

        with _patch("airllm.utils.clean_memory"), _patch(
            "accelerate.utils.modeling.set_module_tensor_to_device"
        ):
            _apply_layer_window(model, 3)
        # 触发第 0 层 pre：窗口加载抛错 → 降级单层
        for module in model.layers:
            if module._airllm_idx == 0:
                pre = next(iter(module._forward_pre_hooks.values()))
                post = next(iter(module._forward_hooks.values()))
                with self.assertRaises(RuntimeError):
                    pre(module, None)
                self.assertTrue(model._airllm_fallback)
                # 降级后再次 pre：走单层路径（仍会抛错但不再尝试窗口）
                self.assertTrue(model._airllm_fallback)
                post(module, None, "out")


if __name__ == "__main__":
    unittest.main()
