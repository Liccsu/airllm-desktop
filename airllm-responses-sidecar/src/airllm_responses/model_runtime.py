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

        self.model = AutoModel.from_pretrained(
            model_dir,
            device=device,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=shard_dir,
            hf_token=None,
            delete_original=False,
        )
        self.tokenizer = self.model.tokenizer
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
