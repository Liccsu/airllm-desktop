from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import json
import threading
import unittest

from airllm_responses.model_runtime import GenerationResult
from airllm_responses.protocol import (
    ProtocolError,
    build_prompt,
    build_response,
    normalize_messages,
    stream_records,
    validate_request,
)


class ProtocolTests(unittest.TestCase):
    def test_normalizes_string_and_message_parts(self) -> None:
        self.assertEqual(
            normalize_messages(
                [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
                "be concise",
            ),
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(build_prompt([{"role": "user", "content": "hello"}]), "user: hello\nassistant:")

    def test_rejects_unsupported_fields_and_multimodal_content(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "unsupported parameter"):
            validate_request({"model": "airllm-local", "input": "hi", "tools": []})
        with self.assertRaisesRegex(ProtocolError, "input_text"):
            normalize_messages([{"role": "user", "content": [{"type": "input_image", "image_url": "x"}]}])

    def test_response_shape_and_usage(self) -> None:
        response = build_response("airllm-local", GenerationResult("AB", 3, 2, "stop"))
        self.assertEqual(response["object"], "response")
        self.assertEqual(response["status"], "completed")
        self.assertEqual(response["output"][0]["content"][0]["text"], "AB")
        self.assertEqual(response["usage"], {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5})
        self.assertIsNone(response["error"])

    def test_sse_event_order_delta_and_completed_usage(self) -> None:
        result = GenerationResult("AB", 4, 2, "stop")
        records = stream_records("airllm-local", "", object(), ["A", "B"], result)  # type: ignore[arg-type]
        events = [record.splitlines()[0].removeprefix("event: ") for record in records]
        self.assertEqual(
            events,
            [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ],
        )
        deltas = []
        completed = None
        for record in records:
            event, data = record.strip().split("\ndata: ", 1)
            payload = json.loads(data)
            if event == "event: response.output_text.delta":
                deltas.append(payload["delta"])
            if event == "event: response.completed":
                completed = payload["response"]
        self.assertEqual("".join(deltas), "AB")
        self.assertEqual(completed["output"][0]["content"][0]["text"], "AB")
        self.assertEqual(completed["usage"]["total_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
