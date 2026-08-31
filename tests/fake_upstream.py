#!/usr/bin/env python3
"""Fake upstream providers for testing CG without touching real APIs.

Modes (chosen by URL prefix, so one server covers all cases):
  /rich/v1     OpenAI-style, /models includes context_window
  /bare/v1     OpenAI-style, /models has no context field, no metadata
  /sibling/v1  bare listing but /models/{id} exposes max_model_len
  /errctx/v1   bare listing; oversized max_tokens returns the limit in an error
  /anthropic/v1  Anthropic-native (x-api-key, /messages, SSE event stream)
  /flaky/v1    OpenAI-style; key 'good' works, others 429
  /deadkey/v1  OpenAI-style; key 'good' works, others 401
  /thinker/v1  reasons inline in content (<think>), drops images silently
  /seer/v1     real vision: an image adds input tokens
  /faker/v1    HTTP 200 carrying a "top up your account" notice
  /emptyreason/v1  reasoning key present but blank (the false-positive trap)
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RICH_MODELS = [
    {"id": "rich-a", "object": "model", "context_window": 512000},
    {"id": "rich-b", "object": "model", "context_length": 262144},
]
BARE_MODELS = [{"id": "bare-a", "object": "model"}, {"id": "bare-b", "object": "model"}]


def _has_image(body):
    """True when any message part carries an image (either API shape)."""
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                return True
    return False


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits = []

    def log_message(self, *a):
        pass

    def _json(self, status, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _mode(self):
        parts = self.path.strip("/").split("/")
        return parts[0] if parts else ""

    def _bearer(self):
        h = self.headers.get("Authorization") or ""
        return h[7:] if h.startswith("Bearer ") else None

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ---------- GET ----------

    def do_GET(self):
        mode = self._mode()
        path = self.path
        # real providers only serve the versioned path; refuse the unversioned
        # one so URL-recovery behaviour is actually exercised
        if "/v1/" not in path and not path.endswith("/v1/models"):
            return self._json(404, {"error": "use /v1"})
        if path.endswith("/models"):
            if mode == "rich":
                return self._json(200, {"object": "list", "data": RICH_MODELS})
            if mode == "anthropic":
                if not self.headers.get("x-api-key"):
                    return self._json(401, {"error": "x-api-key required"})
                return self._json(200, {"data": [{"id": "claude-fake", "type": "model",
                                                  "max_context_length": 200000}]})
            if mode in ("bare", "sibling", "errctx", "flaky", "deadkey",
                        "thinker", "faker", "seer", "emptyreason", "err200",
                        "cheapseer"):
                if not self._bearer():
                    return self._json(401, {"error": "no key"})
                return self._json(200, {"object": "list", "data": BARE_MODELS})
            return self._json(404, {"error": "unknown mode"})

        # /sibling/v1/models/bare-a
        if mode == "sibling" and "/models/" in path:
            mid = path.rsplit("/", 1)[-1]
            return self._json(200, {"id": mid, "max_model_len": 96000})
        if mode == "bare" and "/models/" in path:
            return self._json(404, {"error": "no such endpoint"})
        return self._json(404, {"error": "not found"})

    # ---------- POST ----------

    def do_POST(self):
        mode = self._mode()
        body = self._body()
        Fake.hits.append({"mode": mode, "key": self._bearer() or self.headers.get("x-api-key"),
                          "model": body.get("model"), "stream": bool(body.get("stream")),
                          "body": body})

        if mode == "errctx":
            if int(body.get("max_tokens") or 0) > 1_000_000:
                return self._json(400, {"error": {"message":
                    "This model's maximum context length is 131072 tokens, however you requested more."}})
            return self._chat(body, "errctx ok")

        if mode == "thinker":
            # Reasoning trace inline in content (raw vLLM shape), and images
            # are accepted but silently dropped: prompt_tokens never moves.
            return self._json(200, {
                "id": "chatcmpl-think", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant",
                    "content": "<think>17*23 = 391, let me verify</think>391"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
            })

        if mode == "seer":
            # Genuine vision: an image costs extra input tokens.
            extra = 512 if _has_image(body) else 0
            return self._json(200, {
                "id": "chatcmpl-seer", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "a red square"}}],
                "usage": {"prompt_tokens": 20 + extra, "completion_tokens": 3,
                          "total_tokens": 23 + extra},
            })

        if mode == "cheapseer":
            # Real vision but billed cheaply (b.ai/Qwen charge ~60 tokens for a
            # small image): the delta is modest, still well above the floor.
            extra = 80 if _has_image(body) else 0
            return self._json(200, {
                "id": "chatcmpl-seer", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "a red square"}}],
                "usage": {"prompt_tokens": 20 + extra, "completion_tokens": 3,
                          "total_tokens": 23 + extra},
            })

        if mode == "faker":
            # HTTP 200 carrying a billing notice instead of a completion.
            return self._json(200, {
                "id": "chatcmpl-fake-999", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant",
                    "content": ("Sorry, to prevent abuse of free resources, accounts that "
                                "have not been recharged can only try 10 times.")}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        if mode == "emptyreason":
            # The trap: a reasoning key that is present but blank.
            return self._json(200, {
                "id": "chatcmpl-er", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "391", "reasoning": ""}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            })

        if mode == "flaky":
            if self._bearer() != "good":
                return self._json(429, {"error": {"message": "rate limited"}}, {"Retry-After": "60"})
            return self._chat(body, "flaky ok")

        if mode == "deadkey":
            if self._bearer() != "good":
                return self._json(401, {"error": {"message": "invalid api key"}})
            return self._chat(body, "deadkey ok")

        if mode == "anthropic":
            if not self.headers.get("x-api-key"):
                return self._json(401, {"error": "x-api-key required"})
            if body.get("stream"):
                return self._anthropic_stream(body)
            return self._json(200, {
                "id": "msg_fake", "type": "message", "role": "assistant",
                "model": body.get("model"), "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "hello from anthropic"}],
                "usage": {"input_tokens": 11, "output_tokens": 4},
            })

        if mode == "err200":
            # Listed in /models but every chat call fails INSIDE a 200:
            # the exact false positive that made a dead model look available.
            return self._json(200, {
                "id": "chatcmpl-err200", "object": "chat.completion",
                "created": int(time.time()), "model": body.get("model"),
                "error": {"message": "model %s is not available" % body.get("model")},
                "choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                         "total_tokens": 0},
            })

        return self._chat(body, "%s ok" % mode)

    def _chat(self, body, text):
        if body.get("stream"):
            return self._openai_stream(body, text)
        return self._json(200, {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        })

    def _sse_head(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _openai_stream(self, body, text):
        self._sse_head()
        for piece in text.split():
            chunk = {"id": "c", "object": "chat.completion.chunk", "model": body.get("model"),
                     "choices": [{"index": 0, "delta": {"content": piece + " "}, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def _anthropic_stream(self, body):
        self._sse_head()
        events = [
            ("message_start", {"type": "message_start", "message": {"id": "msg_1", "model": body.get("model")}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "streamed "}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "text_delta", "text": "anthropic"}}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
            ("message_stop", {"type": "message_stop"}),
        ]
        for name, ev in events:
            self.wfile.write(("event: %s\n" % name).encode())
            self.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")


def start(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Fake)
    srv.daemon_threads = True
    return srv


if __name__ == "__main__":
    s = start(int(sys.argv[1]) if len(sys.argv) > 1 else 20199)
    print("fake upstream on http://127.0.0.1:%d" % s.server_address[1])
    s.serve_forever()
