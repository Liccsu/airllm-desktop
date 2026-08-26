"""AirLLM 模型适配器和串行生成运行时。"""

from __future__ import annotations

import collections.abc
import dataclasses
import queue
import threading
import typing


@dataclasses.dataclass(frozen=True)
class GenerationOptions:
    max_output_tokens: int
    temperature: float | None
    top_p: float | None


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    text: str
    input_tokens: int
    output_tokens: int
    finish_reason: str


@dataclasses.dataclass(frozen=True)
class StreamPiece:
    text: str | None = None
    result: GenerationResult | None = None
    error: Exception | None = None


class TextBackend(typing.Protocol):
    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        """生成完整文本响应。"""

    def stream(
        self,
        prompt: str,
        options: GenerationOptions,
        cancel: threading.Event,
    ) -> collections.abc.Iterator[StreamPiece]:
        """生成文本片段，并最终恰好给出一个结果或错误。"""


class AirLLMBackend:
    """生产后端；只有明确请求构造时才导入 AirLLM。"""

    def __init__(
        self,
        model_dir: str,
        device: str,
        shard_dir: str,
        max_seq_len: int,
    ) -> None:
        from airllm import AutoModel

        # airllm 未收录 Qwen3.5 MoE 架构，会退回 generic 类；其权重为
        # model.language_model.layers.N 结构，generic 的 model.layers 前缀
        # 会错误匹配导致拆分崩溃。复用 Qwen3.5 专用类的正确层名映射。
        # 容错处理：上游内部结构变化时跳过，不阻塞模型加载。
        try:
            from airllm import auto_model as _airllm_auto

            _airllm_auto.ARCH_OVERRIDES.setdefault(
                "Qwen3_5MoeForConditionalGeneration", "AirLLMQwen3_5"
            )
        except (ImportError, AttributeError):
            pass

        self.model = AutoModel.from_pretrained(
            model_dir,
            device=device,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=shard_dir,
            hf_token=None,
            delete_original=False,
        )
        self.tokenizer = self.model.tokenizer
        # 层驻留窗口：用显存余量批量驻留多层，减少逐层磁盘换载。
        try:
            window = _layer_window_size(self.model)
            if window > 1:
                _apply_layer_window(self.model, window)
                print(
                    f"[serve] 已启用层驻留窗口: 同时驻留 {window} 层",
                    flush=True,
                )
        except Exception:
            pass
        self._generation_thread: threading.Thread | None = None

    def _tokenize(self, prompt: str) -> tuple[object, int]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=getattr(self.model, "max_seq_len", 512),
        )
        input_ids = encoded["input_ids"]
        return encoded, int(input_ids.shape[-1])

    @staticmethod
    def _generation_kwargs(options: GenerationOptions) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": options.max_output_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        sampling = options.temperature is not None and options.temperature > 0
        kwargs["do_sample"] = sampling
        if sampling:
            kwargs["temperature"] = options.temperature
            if options.top_p is not None:
                kwargs["top_p"] = options.top_p
        return kwargs

    def _decode_result(self, output: object, input_tokens: int) -> GenerationResult:
        sequences = output.sequences
        sequence = sequences[0]
        generated = sequence[input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(generated.shape[-1]),
            finish_reason="stop",
        )

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        encoded, input_tokens = self._tokenize(prompt)
        model_device = getattr(self.model, "device", None)
        model_inputs = {
            key: value.to(model_device) if model_device is not None and hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        output = self.model.generate(**model_inputs, **self._generation_kwargs(options))
        return self._decode_result(output, input_tokens)

    def stream(
        self,
        prompt: str,
        options: GenerationOptions,
        cancel: threading.Event,
    ) -> collections.abc.Iterator[StreamPiece]:
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        class CancelCriteria(StoppingCriteria):
            def __call__(self, input_ids: object, scores: object, **kwargs: object) -> bool:
                return cancel.is_set()

        encoded, input_tokens = self._tokenize(prompt)
        model_device = getattr(self.model, "device", None)
        model_inputs = {
            key: value.to(model_device) if model_device is not None and hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.25,
        )
        terminal: queue.Queue[StreamPiece] = queue.Queue(maxsize=1)
        finished = threading.Event()

        def run_generation() -> None:
            try:
                output = self.model.generate(
                    **model_inputs,
                    **self._generation_kwargs(options),
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([CancelCriteria()]),
                )
                terminal.put(StreamPiece(result=self._decode_result(output, input_tokens)))
            except Exception as exc:
                terminal.put(StreamPiece(error=exc))
            finally:
                finished.set()
                streamer.end()

        thread = threading.Thread(target=run_generation, name="airllm-generate", daemon=True)
        self._generation_thread = thread
        thread.start()
        try:
            while True:
                try:
                    piece = next(streamer)
                except queue.Empty:
                    if finished.is_set():
                        break
                    continue
                except StopIteration:
                    break
                if piece:
                    yield StreamPiece(text=piece)
            terminal_piece = terminal.get(timeout=1.0)
            yield terminal_piece
        finally:
            cancel.set()
            self._generation_thread = None


class DirectBackend:
    """全量加载后端：模型小于可用显存时一次性加载到 GPU。

    AirLLM 的逐层流式加载对 0.5B 这类小模型是过度设计（每 token 都要重新读取
    全部层分片），导致 CPU 满载、GPU 空闲、速度远低于应有的水平。
    模型能完整放进显存时改用 transformers 直接加载，推理速度提升一个数量级。
    """

    def __init__(self, model_dir: str, device: str, max_seq_len: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_seq_len = max_seq_len
        self.device = device if device.startswith("cuda") else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._generation_thread: threading.Thread | None = None

    def _tokenize(self, prompt: str) -> tuple[object, int]:
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = encoded["input_ids"]
        return encoded, int(input_ids.shape[-1])

    @staticmethod
    def _generation_kwargs(options: GenerationOptions) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": options.max_output_tokens,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        sampling = options.temperature is not None and options.temperature > 0
        kwargs["do_sample"] = sampling
        if sampling:
            kwargs["temperature"] = options.temperature
            if options.top_p is not None:
                kwargs["top_p"] = options.top_p
        return kwargs

    def _decode_result(self, output: object, input_tokens: int) -> GenerationResult:
        sequences = output.sequences
        sequence = sequences[0]
        generated = sequence[input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=int(generated.shape[-1]),
            finish_reason="stop",
        )

    def _model_inputs(self, prompt: str) -> tuple[dict[str, object], int]:
        encoded, input_tokens = self._tokenize(prompt)
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        return inputs, input_tokens

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        model_inputs, input_tokens = self._model_inputs(prompt)
        output = self.model.generate(**model_inputs, **self._generation_kwargs(options))
        return self._decode_result(output, input_tokens)

    def stream(
        self,
        prompt: str,
        options: GenerationOptions,
        cancel: threading.Event,
    ) -> collections.abc.Iterator[StreamPiece]:
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        class CancelCriteria(StoppingCriteria):
            def __call__(self, input_ids: object, scores: object, **kwargs: object) -> bool:
                return cancel.is_set()

        model_inputs, input_tokens = self._model_inputs(prompt)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=0.25,
        )
        terminal: queue.Queue[StreamPiece] = queue.Queue(maxsize=1)
        finished = threading.Event()

        def run_generation() -> None:
            try:
                output = self.model.generate(
                    **model_inputs,
                    **self._generation_kwargs(options),
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([CancelCriteria()]),
                )
                terminal.put(StreamPiece(result=self._decode_result(output, input_tokens)))
            except Exception as exc:
                terminal.put(StreamPiece(error=exc))
            finally:
                finished.set()
                streamer.end()

        thread = threading.Thread(target=run_generation, name="airllm-generate", daemon=True)
        self._generation_thread = thread
        thread.start()
        try:
            while True:
                try:
                    piece = next(streamer)
                except queue.Empty:
                    if finished.is_set():
                        break
                    continue
                except StopIteration:
                    break
                if piece:
                    yield StreamPiece(text=piece)
            terminal_piece = terminal.get(timeout=1.0)
            yield terminal_piece
        finally:
            cancel.set()
            self._generation_thread = None


# ── 层驻留窗口 ────────────────────────────────────────────────────────────
#
# airllm 默认每层计算完立即释放，每生成一个 token 都要从磁盘重新读全部层，
# GPU 大部分时间空转。这里把 pre/post hook 替换为“窗口驻留”版本：
# 一次加载 W 层驻留显存，窗口滑过才释放，磁盘换层次数降低到 1/W。

_WINDOW_MAX = 8


def _layer_window_size(model: object) -> int:
    """按可用显存与每层权重体积估算驻留窗口大小 W。"""

    import os
    from pathlib import Path

    try:
        import torch

        if not torch.cuda.is_available():
            return 1
        device = getattr(model, "device", "cuda:0")
        dev = torch.device(device) if isinstance(device, str) else device
        props = torch.cuda.get_device_properties(dev.index)
        vram_total = props.total_memory
        vram_used = torch.cuda.memory_allocated(dev.index)
    except Exception:
        return 1
    checkpoint = getattr(model, "checkpoint_path", None)
    n_layers = len(getattr(model, "layer_names", None) or [])
    if not checkpoint or n_layers <= 0:
        return 1
    root = Path(checkpoint).parent
    total_bytes = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total_bytes += os.path.getsize(path)
                except OSError:
                    pass
    if total_bytes <= 0:
        return 1
    # 保守按反量化后 fp16 体积估算（存储可能是 INT8/FP8，乘以 2）。
    per_layer = total_bytes * 2.0 / n_layers
    avail = vram_total - vram_used - 2 * 1024**3
    window = int(avail / per_layer)
    return max(1, min(window, _WINDOW_MAX, n_layers))


def _apply_layer_window(model: object, window: int) -> None:
    """替换 airllm 的层 pre/post hook 为窗口驻留版本。"""

    from accelerate.utils.modeling import set_module_tensor_to_device
    from airllm.utils import clean_memory

    model._airllm_resident = {}
    model._airllm_window = max(1, window)
    model._airllm_fallback = False

    def window_pre(module: object, args: object) -> None:
        idx = module._airllm_idx  # type: ignore[attr-defined]
        resident = model._airllm_resident  # type: ignore[attr-defined]
        if getattr(model, "_airllm_fallback", False):
            # 降级模式：单层加载、用后即释放（airllm 原行为）。
            state_dict = model._load_streamed_layer(idx)  # type: ignore[attr-defined]
            module._airllm_moved = model.move_layer_to_device(state_dict)  # type: ignore[attr-defined]
            return
        if idx in resident:
            return
        try:
            # 释放窗口滑过的层。
            for i in list(resident):
                if i < idx:
                    for param_name in resident[i]:
                        set_module_tensor_to_device(model.model, param_name, "meta")  # type: ignore[attr-defined]
                    del resident[i]
            # 批量加载 [idx, idx+W) 驻留。
            end = min(idx + model._airllm_window, len(model.layers))  # type: ignore[attr-defined]
            for i in range(idx, end):
                if i in resident:
                    continue
                state_dict = model._load_streamed_layer(i)  # type: ignore[attr-defined]
                moved = model.move_layer_to_device(state_dict)  # type: ignore[attr-defined]
                resident[i] = moved
            clean_memory()
        except Exception:
            # 窗口加载失败（显存不足等）时逐层降级，不让整个推理崩溃。
            model._airllm_fallback = True
            model._airllm_resident.clear()
            state_dict = model._load_streamed_layer(idx)  # type: ignore[attr-defined]
            module._airllm_moved = model.move_layer_to_device(state_dict)  # type: ignore[attr-defined]
            print(
                "[serve] 层驻留窗口失效，已回退逐层加载模式",
                flush=True,
            )

    def window_post(module: object, args: object, output: object) -> object:
        if getattr(model, "_airllm_fallback", False):
            # 降级模式：本层用后即释放。
            for param_name in getattr(module, "_airllm_moved", []):
                set_module_tensor_to_device(model.model, param_name, "meta")  # type: ignore[attr-defined]
            module._airllm_moved = []  # type: ignore[attr-defined]
            clean_memory()
        # 窗口模式：不释放，层在窗口滑过前保持驻留。
        return output

    for module in model.layers:  # type: ignore[attr-defined]
        midx = getattr(module, "_airllm_idx", None)
        if midx is None:
            continue
        for hooks in (module._forward_pre_hooks, module._forward_hooks):  # type: ignore[attr-defined]
            stale = [k for k, v in hooks.items() if v == model._pre_hook or v == model._post_hook]  # type: ignore[attr-defined]
            for k in stale:
                del hooks[k]
        module.register_forward_pre_hook(window_pre)  # type: ignore[attr-defined]
        module.register_forward_hook(window_post)  # type: ignore[attr-defined]


def choose_backend(
    model_dir: str,
    device: str,
    shard_dir: str,
    max_seq_len: int,
) -> AirLLMBackend | DirectBackend:
    """按模型大小与可用显存选择后端：放得下就全量加载，否则退回 AirLLM 流式。"""

    import os
    from pathlib import Path

    if device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            total_bytes = 0
            root = Path(model_dir)
            if root.is_dir():
                for path in root.rglob("*"):
                    if path.is_file():
                        try:
                            total_bytes += os.path.getsize(path)
                        except OSError:
                            pass
            try:
                device_index = int(device.split(":")[-1]) if ":" in device else 0
                vram = torch.cuda.get_device_properties(device_index).total_memory
            except Exception:
                vram = 0
            if 0 < total_bytes < vram * 0.6:
                return DirectBackend(model_dir, device, max_seq_len)
    return AirLLMBackend(model_dir, device, shard_dir, max_seq_len)


class ModelRuntime:
    """Process-level single-model runtime with one serialized generation slot."""

    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend
        self._generation_lock = threading.Lock()

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        with self._generation_lock:
            return self.backend.generate(prompt, options)

    def stream(
        self,
        prompt: str,
        options: GenerationOptions,
        cancel: threading.Event,
    ) -> collections.abc.Iterator[StreamPiece]:
        acquired = self._generation_lock.acquire()
        try:
            yield from self.backend.stream(prompt, options, cancel)
        finally:
            if acquired:
                self._generation_lock.release()
