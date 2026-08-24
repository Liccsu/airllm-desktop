from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from fastapi.testclient import TestClient

from airllm_responses.config import Settings
from airllm_responses.model_runtime import GenerationOptions, GenerationResult, StreamPiece
from airllm_responses.server import _RuntimeSlot, _stream_response, create_app



class FakeBackend:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.cancel: threading.Event | None = None
        self.prompts: list[str] = []

    def generate(self, prompt: str, options: GenerationOptions) -> GenerationResult:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return GenerationResult("AB", 2, 2, "stop")

    def stream(self, prompt: str, options: GenerationOptions, cancel: threading.Event):
        self.prompts.append(prompt)
        self.cancel = cancel
        if self.error:
            yield StreamPiece(error=self.error)
            return
        yield StreamPiece(text="A")
        yield StreamPiece(text="B")
        yield StreamPiece(result=GenerationResult("AB", 2, 2, "stop"))


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings(
            manifest_path=None,
            model_dir=None,
            shard_dir=None,
            model_name="airllm-local",
            api_key="test-key",
            device="cpu",
            max_seq_len=512,
            max_output_tokens=128,
            max_concurrent_requests=1,
            host="127.0.0.1",
            port=8000,
            preload=False,
        )

    def test_auth_model_and_not_ready(self) -> None:
        backend = FakeBackend()
        client = TestClient(create_app(backend, self.settings))
        payload = {"model": "airllm-local", "input": "hello"}
        self.assertEqual(client.post("/v1/responses", json=payload).status_code, 401)
        self.assertEqual(client.post("/v1/responses", json=payload, headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(client.post("/v1/responses", json={**payload, "model": "other"}, headers={"Authorization": "Bearer test-key"}).status_code, 400)

        not_ready_client = TestClient(create_app(None, self.settings))
        self.assertEqual(not_ready_client.get("/healthz").status_code, 503)
        self.assertEqual(not_ready_client.get("/v1/models").status_code, 503)

    def test_incomplete_manifest_snapshot_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "source": "modelscope",
                "model_id": "demo/model",
                "revision": "immutable-r1",
                "model_dir": str(model_dir),
                "approved": True,
            }), encoding="utf-8")
            settings = dataclasses.replace(
                self.settings,
                manifest_path=manifest,
                model_dir=model_dir,
                shard_dir=root / "shards",
            )
            with patch("airllm_responses.server.AirLLMBackend") as backend:
                response = TestClient(create_app(None, settings)).get("/healthz")
            self.assertEqual(response.status_code, 503)
            backend.assert_not_called()
    def test_invalid_manifest_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir = root / "model"
            model_dir.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "source": "huggingface",
                "model_id": "demo/model",
                "revision": "immutable-r1",
                "model_dir": str(model_dir),
                "approved": True,
            }), encoding="utf-8")
            settings = dataclasses.replace(
                self.settings,
                manifest_path=manifest,
                model_dir=model_dir,
                shard_dir=root / "shards",
            )
            response = TestClient(create_app(None, settings)).get("/healthz")
            self.assertEqual(response.status_code, 503)

    def test_non_stream_json_and_unsupported_parameter(self) -> None:
        backend = FakeBackend()
        client = TestClient(create_app(backend, self.settings))
        headers = {"Authorization": "Bearer test-key"}
        response = client.post("/v1/responses", json={"model": "airllm-local", "input": "hello"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["output"][0]["content"][0]["text"], "AB")
        self.assertEqual(body["usage"]["total_tokens"], 4)
        rejected = client.post("/v1/responses", json={"model": "airllm-local", "input": "hello", "tools": []}, headers=headers)
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual(rejected.json()["error"]["type"], "unsupported_parameter")

    def test_stream_content_type_order_and_delta(self) -> None:
        backend = FakeBackend()
        client = TestClient(create_app(backend, self.settings))
        with client.stream(
            "POST",
            "/v1/responses",
            json={"model": "airllm-local", "input": "hello", "stream": True},
            headers={"Authorization": "Bearer test-key"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            text = "".join(response.iter_text())
        events = [line.removeprefix("event: ") for line in text.splitlines() if line.startswith("event: ")]
        self.assertEqual(events, [
            "response.created", "response.in_progress", "response.output_item.added",
            "response.content_part.added", "response.output_text.delta", "response.output_text.delta",
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
            "response.completed",
        ])
        payloads = [json.loads(text.split("data: ", 1)[1].split("\n", 1)[0]) for text in text.split("\n\n") if text]
        completed = payloads[-1]["response"]
        self.assertEqual(completed["output"][0]["content"][0]["text"], "AB")

    def test_backend_error_returns_http_500_and_stream_error(self) -> None:
        client = TestClient(create_app(FakeBackend(RuntimeError("boom")), self.settings))
        headers = {"Authorization": "Bearer test-key"}
        response = client.post("/v1/responses", json={"model": "airllm-local", "input": "hello"}, headers=headers)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["type"], "server_error")
        with client.stream("POST", "/v1/responses", json={"model": "airllm-local", "input": "hello", "stream": True}, headers=headers) as streamed:
            text = "".join(streamed.iter_text())
        self.assertIn("event: error", text)

    def test_stream_generator_close_sets_cancel(self) -> None:
        backend = FakeBackend()
        generator = _stream_response(
            __import__("airllm_responses.server", fromlist=["_RuntimeSlot"])._RuntimeSlot(backend, self.settings),
            "airllm-local",
            "user: hello",
            GenerationOptions(8, 0.0, None),
        )
        next(generator)
        next(generator)
        next(generator)
        next(generator)
        next(generator)
        generator.close()
        self.assertIsNotNone(backend.cancel)
        self.assertTrue(backend.cancel.is_set())
    def test_default_path_does_not_download(self) -> None:
        def fail_download(*args: object, **kwargs: object) -> None:
            raise AssertionError("download must not be called")

        fake_huggingface = types.ModuleType("huggingface_hub")
        fake_huggingface.snapshot_download = fail_download  # type: ignore[attr-defined]
        with patch.dict(os.environ, {"AIRLLM_API_KEY": "test-key"}, clear=False), patch.dict(
            sys.modules,
            {"huggingface_hub": fake_huggingface},
        ):
            client = TestClient(create_app(None))
            self.assertEqual(client.get("/healthz").status_code, 503)


if __name__ == "__main__":
    unittest.main()
