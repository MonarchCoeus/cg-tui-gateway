"""OpenAI <-> Anthropic translation for providers whose native API is Anthropic.

Only the egress direction is implemented: CG always speaks OpenAI to its
clients, and translates outbound when the upstream is Anthropic-native.
"""

import base64
import json
import re
import time
import uuid

# data:image/png;base64,AAAA...
_DATA_URL = re.compile(r"^data:([^;,]+);base64,(.*)$", re.S)


def _text_of(content):
    """OpenAI content can be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and part.get("type") in (None, "text"):
                chunks.append(part.get("text") or "")
        return "".join(chunks)
    return ""


def _blocks_of(content):
    """OpenAI content -> Anthropic content blocks, keeping images.

    Dropping image parts turned a vision request into a silent text-only one,
    which reads as the model ignoring the picture. Images are forwarded as
    Anthropic image blocks; a URL that isn't an inline data: URL can't be
    forwarded (Anthropic takes base64 or its own file ids), so it degrades to
    a visible text note rather than vanishing.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    blocks = []
    for part in content:
        if isinstance(part, str):
            if part:
                blocks.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype in (None, "text"):
            text = part.get("text") or ""
            if text:
                blocks.append({"type": "text", "text": text})
        elif ptype == "image_url":
            url = (part.get("image_url") or {})
            url = url.get("url") if isinstance(url, dict) else url
            block = _image_block(url)
            blocks.append(block)
        elif ptype == "image" and isinstance(part.get("source"), dict):
            blocks.append(part)  # already Anthropic-shaped
    return blocks


def _image_block(url):
    if not isinstance(url, str) or not url:
        return {"type": "text", "text": "[image omitted: no url]"}
    m = _DATA_URL.match(url.strip())
    if m:
        media, data = m.group(1), m.group(2).strip()
        try:  # reject a corrupt payload here rather than upstream
            base64.b64decode(data, validate=True)
        except Exception:
            return {"type": "text", "text": "[image omitted: undecodable data url]"}
        return {"type": "image",
                "source": {"type": "base64", "media_type": media, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _merge_content(existing, incoming):
    """Concatenate two message contents, promoting to block form as needed."""
    if isinstance(existing, str) and isinstance(incoming, str):
        return existing + "\n\n" + incoming
    a = [{"type": "text", "text": existing}] if isinstance(existing, str) else list(existing)
    b = [{"type": "text", "text": incoming}] if isinstance(incoming, str) else list(incoming)
    return a + b


def openai_to_anthropic(body):
    """Build an Anthropic /v1/messages request from an OpenAI chat request."""
    systems, messages = [], []
    for msg in body.get("messages") or []:
        role = msg.get("role")
        if role == "system":
            text = _text_of(msg.get("content"))
            if text:
                systems.append(text)
            continue
        content = _blocks_of(msg.get("content"))
        if role == "tool":
            role = "user"
        if role not in ("user", "assistant"):
            role = "user"
        # Anthropic rejects consecutive same-role turns; merge them.
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] = _merge_content(messages[-1]["content"], content)
        else:
            messages.append({"role": role, "content": content})

    if not messages:
        messages = [{"role": "user", "content": ""}]

    # max_tokens is mandatory upstream; OpenAI clients often omit it. An
    # explicit 0 or a non-numeric value is treated as "unset", not as zero.
    want = body.get("max_tokens")
    if want in (None, 0, ""):
        want = body.get("max_completion_tokens")
    try:
        max_tokens = int(want)
    except (TypeError, ValueError):
        max_tokens = 0
    out = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": max_tokens if max_tokens > 0 else 4096,
    }
    if systems:
        out["system"] = "\n\n".join(systems)
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("stop", "stop_sequences")):
        if body.get(src) is not None:
            val = body[src]
            if dst == "stop_sequences" and isinstance(val, str):
                val = [val]
            out[dst] = val
    if body.get("stream"):
        out["stream"] = True
    # forward reasoning_effort as an Anthropic thinking block so the
    # hierarchy (low/medium/high/xhigh/max) applies on anthropic-native
    # upstreams too — not just the openai passthrough path
    effort = body.get("reasoning_effort")
    if effort:
        budgets = {"low": 1024, "medium": 2048, "high": 4096, "xhigh": 8192, "max": 16384}
        budget = budgets.get(str(effort).lower(), 2048)
        out["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif body.get("thinking") and isinstance(body.get("thinking"), dict):
        # pass through an already-anthropic-shaped thinking config
        out["thinking"] = body["thinking"]
    return out


_STOP_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def anthropic_to_openai(payload, model):
    """Convert a non-streaming Anthropic response to OpenAI chat shape."""
    text = ""
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += block.get("text") or ""
    usage = payload.get("usage") or {}
    return {
        "id": payload.get("id") or "chatcmpl-%s" % uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": _STOP_MAP.get(payload.get("stop_reason"), "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


class StreamTranslator:
    """Feed Anthropic SSE lines in, get OpenAI SSE chunks out."""

    def __init__(self, model):
        self.model = model
        self.id = "chatcmpl-%s" % uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.sent_role = False
        self.done = False

    def _chunk(self, delta, finish=None):
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def _emit(self, obj):
        return b"data: " + json.dumps(obj).encode() + b"\n\n"

    def feed(self, line):
        """line: one decoded SSE line. Returns bytes to forward (may be b'')."""
        line = line.strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            return b""
        if not line.startswith("data:"):
            return b""
        raw = line[5:].strip()
        if raw == "[DONE]":
            return self.finish()
        try:
            ev = json.loads(raw)
        except ValueError:
            return b""

        etype = ev.get("type")
        out = b""
        if etype == "message_start" and not self.sent_role:
            self.sent_role = True
            out += self._emit(self._chunk({"role": "assistant", "content": ""}))
        elif etype == "content_block_delta":
            delta = ev.get("delta") or {}
            text = delta.get("text") or delta.get("partial_json") or ""
            if text:
                if not self.sent_role:
                    self.sent_role = True
                    out += self._emit(self._chunk({"role": "assistant", "content": ""}))
                out += self._emit(self._chunk({"content": text}))
        elif etype == "message_delta":
            reason = (ev.get("delta") or {}).get("stop_reason")
            if reason:
                out += self._emit(self._chunk({}, _STOP_MAP.get(reason, "stop")))
        elif etype == "message_stop":
            out += self.finish()
        return out

    def finish(self):
        if self.done:
            return b""
        self.done = True
        return b"data: [DONE]\n\n"
