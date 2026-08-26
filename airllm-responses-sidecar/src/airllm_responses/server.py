"""本地 AirLLM Responses profile 的 FastAPI 应用。"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import ConfigError, Settings, load_approved_manifest, load_settings
from .model_runtime import (
    AirLLMBackend,
    DirectBackend,
    GenerationResult,
    ModelRuntime,
    StreamPiece,
    TextBackend,
    choose_backend,
)
from .provision import ProvisionError, validate_snapshot
from .protocol import (
    ProtocolError,
    build_prompt,
    build_response,
    error_body,
    generation_options,
    normalize_messages,
    sse_record,
    validate_request,
)


class _NotReadyError(RuntimeError):
    pass


class _RuntimeSlot:
    def __init__(self, runtime: TextBackend | ModelRuntime | None, settings: Settings | None) -> None:
        self.settings = settings
        self._runtime = runtime if isinstance(runtime, ModelRuntime) else (ModelRuntime(runtime) if runtime is not None else None)
        self._load_lock = threading.Lock()
        self._load_error: Exception | None = None

    @property
    def runtime(self) -> ModelRuntime | None:
        return self._runtime

    @property
    def ready(self) -> bool:
        return self._runtime is not None

    @property
    def load_error(self) -> Exception | None:
        return self._load_error

    def _validated_manifest(self):
        if self.settings is None:
            raise _NotReadyError("服务配置不可用")
        if self.settings.manifest_path is None or self.settings.model_dir is None or self.settings.shard_dir is None:
            raise _NotReadyError("模型清单、模型目录和外部分片目录必须配置")
        try:
            manifest = load_approved_manifest(self.settings.manifest_path)
        except ConfigError as exc:
            raise _NotReadyError("模型清单无效") from exc
        if manifest is None:
            raise _NotReadyError("没有已批准的模型清单")
        if manifest.model_dir != self.settings.model_dir:
            raise _NotReadyError("模型清单路径与 AIRLLM_MODEL_DIR 不一致")
        try:
            validate_snapshot(manifest.model_dir)
        except ProvisionError as exc:
            raise _NotReadyError("模型 snapshot 不完整") from exc
        return manifest

    def validate_snapshot(self) -> None:
        """健康检查使用的无副作用 snapshot 预检。"""
        self._validated_manifest()

    def _load_production(self) -> ModelRuntime:
        manifest = self._validated_manifest()
        # 模型能完整放进显存时全量加载（远快于逐层流式），否则退回 AirLLM 流式。
        backend = choose_backend(
            model_dir=str(manifest.model_dir),
            device=self.settings.device,
            shard_dir=str(self.settings.shard_dir),
            max_seq_len=self.settings.max_seq_len,
        )
        mode = "全量加载（GPU）" if isinstance(backend, DirectBackend) else "逐层流式加载（AirLLM）"
        print(f"[serve] 模型加载模式: {mode}", flush=True)
        return ModelRuntime(backend)

    def ensure_runtime(self) -> ModelRuntime:
        if self._runtime is not None:
            return self._runtime
        with self._load_lock:
            if self._runtime is None:
                try:
                    self._runtime = self._load_production()
                    self._load_error = None
                except Exception as exc:
                    self._load_error = exc
                    if isinstance(exc, _NotReadyError):
                        raise
                    raise _NotReadyError(f"模型加载失败: {type(exc).__name__}") from exc
        return self._runtime


def _settings_for_app(settings: Settings | None) -> Settings | None:
    if settings is not None:
        return settings
    try:
        return load_settings()
    except ConfigError:
        return None


def _auth_error() -> JSONResponse:
    return JSONResponse(
        error_body("authentication_error", "invalid or missing bearer token"),
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _authorized(request: Request, settings: Settings | None) -> bool:
    if settings is None or not settings.api_key:
        return False
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    return separator == " " and scheme.lower() == "bearer" and token == settings.api_key


def _not_ready(slot: _RuntimeSlot) -> JSONResponse:
    model = slot.settings.model_name if slot.settings is not None else "airllm-local"
    return JSONResponse({"model": model, "ready": False}, status_code=503)


def _base_stream_response(model: str, response_id: str, created_at: int, item_id: str, status: str) -> dict[str, Any]:
    response = build_response(
        model,
        result=GenerationResult("", 0, 0, "stop"),
        response_id=response_id,
        created_at=created_at,
        item_id=item_id,
    )
    response["status"] = status
    return response


def _stream_response(
    slot: _RuntimeSlot,
    model: str,
    prompt: str,
    options: Any,
) -> Iterator[str]:
    import time
    import uuid

    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created_at = int(time.time())
    queued = _base_stream_response(model, response_id, created_at, item_id, "queued")
    yield sse_record("response.created", {"type": "response.created", "response": queued})
    in_progress = dict(queued)
    in_progress["status"] = "in_progress"
    yield sse_record("response.in_progress", {"type": "response.in_progress", "response": in_progress})
    item = {"id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}
    yield sse_record("response.output_item.added", {"type": "response.output_item.added", "item": item, "output_index": 0})
    part = {"type": "output_text", "text": "", "annotations": []}
    yield sse_record("response.content_part.added", {"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "content_index": 0, "part": part})

    cancel = threading.Event()
    pieces: list[str] = []
    result = None
    try:
        for stream_piece in slot.ensure_runtime().stream(prompt, options, cancel):
            if stream_piece.text is not None:
                pieces.append(stream_piece.text)
                yield sse_record(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": stream_piece.text,
                    },
                )
            elif stream_piece.error is not None:
                yield sse_record(
                    "error",
                    {"type": "error", "error": {"type": "server_error", "message": "generation failed"}},
                )
                return
            elif stream_piece.result is not None:
                result = stream_piece.result
        if result is None:
            yield sse_record(
                "error",
                {"type": "error", "error": {"type": "server_error", "message": "generation produced no result"}},
            )
            return
        text = "".join(pieces)
        if not pieces:
            text = result.text
        result = dataclasses.replace(result, text=text)
        yield sse_record("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": text})
        yield sse_record("response.content_part.done", {"type": "response.content_part.done", "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}})
        done_item = {"id": item_id, "type": "message", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}
        yield sse_record("response.output_item.done", {"type": "response.output_item.done", "item": done_item, "output_index": 0})
        completed = build_response(model, result, response_id, created_at, item_id)
        yield sse_record("response.completed", {"type": "response.completed", "response": completed})
    except _NotReadyError:
        yield sse_record("error", {"type": "error", "error": {"type": "server_error", "message": "model is not ready"}})
    except Exception:
        yield sse_record("error", {"type": "error", "error": {"type": "server_error", "message": "generation failed"}})
    finally:
        cancel.set()


def create_app(runtime: TextBackend | ModelRuntime | None = None, settings: Settings | None = None) -> FastAPI:
    resolved_settings = _settings_for_app(settings)
    slot = _RuntimeSlot(runtime, resolved_settings)
    application = FastAPI(title="AirLLM Responses Sidecar")

    # 桌面端 WebView 与本地服务跨源（dev 为 http://127.0.0.1:1420，打包后为 tauri://localhost），
    # 必须放行 CORS，否则前端 fetch 会被拦截并报 Failed to fetch。
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if resolved_settings is not None and resolved_settings.preload and runtime is None:
        try:
            slot.ensure_runtime()
        except _NotReadyError:
            # 预加载失败原因输出到 stderr（桌面端会作为服务日志展示），
            # 带出底层异常详情，便于定位是目录、依赖还是架构兼容问题。
            import sys as _sys
            import traceback as _traceback
            detail = slot.load_error
            print(
                "[serve] 模型预加载失败：模型未就绪，请检查模型目录、设备(CUDA)与依赖是否可用",
                file=_sys.stderr,
                flush=True,
            )
            if detail is not None:
                _traceback.print_exception(type(detail), detail, detail.__traceback__, file=_sys.stderr)
            else:
                print("[serve]（无底层异常详情）", file=_sys.stderr, flush=True)

    @application.get("/healthz")
    async def healthz() -> JSONResponse:
        if not slot.ready:
            try:
                slot.validate_snapshot()
            except _NotReadyError:
                return _not_ready(slot)
            return _not_ready(slot)
        return JSONResponse({"model": resolved_settings.model_name if resolved_settings else "airllm-local", "ready": True})

    @application.get("/v1/models")
    async def models() -> JSONResponse:
        if not slot.ready:
            return _not_ready(slot)
        name = resolved_settings.model_name if resolved_settings else "airllm-local"
        return JSONResponse({"object": "list", "data": [{"id": name, "object": "model", "owned_by": "airllm"}]})

    @application.post("/v1/responses")
    async def responses(request: Request) -> Response:
        if not _authorized(request, resolved_settings):
            return _auth_error()
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ProtocolError("invalid_request", "request body must be an object")
            payload = dict(payload)
            if "max_output_tokens" not in payload and resolved_settings is not None:
                payload["max_output_tokens"] = resolved_settings.max_output_tokens
            request_model = validate_request(payload)
        except ProtocolError as exc:
            return JSONResponse(error_body(exc.error_type, exc.message), status_code=400)
        except Exception:
            return JSONResponse(error_body("invalid_request", "request body must be valid JSON"), status_code=400)

        model_name = resolved_settings.model_name if resolved_settings else "airllm-local"
        if request_model.model != model_name:
            return JSONResponse(error_body("invalid_request", "model does not match configured model"), status_code=400)
        try:
            active_runtime = slot.ensure_runtime()
        except _NotReadyError:
            return _not_ready(slot)
        messages = normalize_messages(request_model.input, request_model.instructions)
        tokenizer = getattr(getattr(active_runtime, "backend", None), "tokenizer", None)
        prompt = build_prompt(messages, tokenizer)
        options = generation_options(request_model)
        if request_model.stream:
            return StreamingResponse(
                _stream_response(slot, model_name, prompt, options),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        try:
            result = await asyncio.to_thread(active_runtime.generate, prompt, options)
        except Exception:
            return JSONResponse(error_body("server_error", "generation failed"), status_code=500)
        return JSONResponse(build_response(model_name, result))

    return application


try:
    app = create_app()
except Exception:
    app = create_app(runtime=None, settings=None)

__all__ = ["app", "create_app"]
