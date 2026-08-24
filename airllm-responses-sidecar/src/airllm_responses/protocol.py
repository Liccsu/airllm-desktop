"""Pydantic 请求校验和 OpenAI Responses 形状对象。"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .model_runtime import GenerationOptions, GenerationResult


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    stream: bool = False
    max_output_tokens: int = Field(default=128, ge=1)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    store: bool = False

    @field_validator("store")
    @classmethod
    def only_false_store(cls, value: bool) -> bool:
        if value:
            raise ValueError("store must be false")
        return value


class ProtocolError(Exception):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


_UNSUPPORTED_FIELDS = {
    "tools",
    "tool_choice",
    "previous_response_id",
    "background",
    "stop",
    "audio",
    "modalities",
}
_ALLOWED_FIELDS = {
    "model",
    "input",
    "instructions",
    "stream",
    "max_output_tokens",
    "temperature",
    "top_p",
    "store",
}


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        raise ProtocolError("unsupported_parameter", "message content must be text")
    parts: list[str] = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "input_text" or not isinstance(part.get("text"), str):
            raise ProtocolError("unsupported_parameter", "only input_text content parts are supported")
        parts.append(part["text"])
    return "".join(parts)


def normalize_messages(value: str | Sequence[Mapping[str, Any]], instructions: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if instructions is not None:
        messages.append({"role": "system", "content": instructions})
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return messages
    if not isinstance(value, list) or not value:
        raise ProtocolError("invalid_request", "input must be a string or non-empty message array")
    for message in value:
        if not isinstance(message, Mapping):
            raise ProtocolError("unsupported_parameter", "input messages must be objects")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ProtocolError("unsupported_parameter", "only system, user, and assistant roles are supported")
        messages.append({"role": role, "content": _text_content(message.get("content"))})
    return messages


def validate_request(payload: Mapping[str, Any]) -> ResponseRequest:
    unknown = set(payload) - _ALLOWED_FIELDS
    unsupported = unknown & _UNSUPPORTED_FIELDS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ProtocolError("unsupported_parameter", f"unsupported parameter: {names}")
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ProtocolError("unsupported_parameter", f"unsupported parameter: {names}")
    try:
        request = ResponseRequest.model_validate(payload)
    except ValidationError as exc:
        detail = exc.errors()[0].get("msg", "invalid request")
        raise ProtocolError("invalid_request", detail) from exc
    normalize_messages(request.input, request.instructions)
    return request


def build_prompt(messages: Sequence[Mapping[str, str]], tokenizer: Any = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
            if isinstance(rendered, str):
                return rendered
        except (AttributeError, TypeError, ValueError, NotImplementedError):
            pass
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generation_options(request: ResponseRequest) -> GenerationOptions:
    return GenerationOptions(
        max_output_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )


def _output_item(text: str, item_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def build_response(
    model: str,
    result: GenerationResult,
    response_id: str | None = None,
    created_at: int | None = None,
    item_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id or f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "status": "completed",
        "model": model,
        "output": [_output_item(result.text, item_id or f"msg_{uuid.uuid4().hex}")],
        "usage": {
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
        "error": None,
    }


def error_body(error_type: str, message: str) -> dict[str, Any]:
    return {"error": {"type": error_type, "message": message}}


def sse_record(event_type: str, data: Mapping[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"


def stream_records(
    model: str,
    prompt: str,
    options: GenerationOptions,
    pieces: Sequence[str],
    result: GenerationResult,
    response_id: str | None = None,
    created_at: int | None = None,
    item_id: str | None = None,
) -> list[str]:
    """为测试和非网络调用构造确定性的事件记录。"""
    response_id = response_id or f"resp_{uuid.uuid4().hex}"
    created_at = created_at if created_at is not None else int(time.time())
    item_id = item_id or f"msg_{uuid.uuid4().hex}"
    partial = build_response(model, GenerationResult("", 0, 0, "stop"), response_id, created_at, item_id)
    partial["status"] = "queued"
    records = [sse_record("response.created", {"type": "response.created", "response": partial})]
    partial["status"] = "in_progress"
    records.append(sse_record("response.in_progress", {"type": "response.in_progress", "response": partial}))
    item = _output_item("", item_id)
    item["status"] = "in_progress"
    records.append(sse_record("response.output_item.added", {"type": "response.output_item.added", "item": item, "output_index": 0}))
    part = {"type": "output_text", "text": "", "annotations": []}
    records.append(sse_record("response.content_part.added", {"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "content_index": 0, "part": part}))
    text = ""
    for piece in pieces:
        text += piece
        records.append(sse_record("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": piece}))
    records.append(sse_record("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "output_index": 0, "content_index": 0, "text": text}))
    records.append(sse_record("response.content_part.done", {"type": "response.content_part.done", "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}}))
    done_item = _output_item(text, item_id)
    records.append(sse_record("response.output_item.done", {"type": "response.output_item.done", "item": done_item, "output_index": 0}))
    completed = build_response(model, result, response_id, created_at, item_id)
    records.append(sse_record("response.completed", {"type": "response.completed", "response": completed}))
    return records
