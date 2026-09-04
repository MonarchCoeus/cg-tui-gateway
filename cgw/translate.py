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
    Anthropic image blocks: inline data: URLs as base64, remote URLs with a
    url source (supported upstream); anything else degrades to a visible
    text note rather than vanishing.
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
    a = [] if existing == "" else ([{"type": "text", "text": existing}] if isinstance(existing, str) else list(existing))
    b = [] if incoming == "" else ([{"type": "text", "text": incoming}] if isinstance(incoming, str) else list(incoming))
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
        if role == "assistant":
            # prior tool calls must stay structured or the loop breaks:
            # free text rides in content blocks, calls as tool_use blocks
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                if not fn.get("name"):
                    continue
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except ValueError:
                    args = {}
                content = _merge_content(
                    content, [{"type": "tool_use",
                               "id": tc.get("id") or "call_0",
                               "name": fn["name"], "input": args}])
        if role == "tool":
            # tool results are first-class blocks upstream, not user text
            content = [{"type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id") or "",
                        "content": _text_of(msg.get("content"))}]
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
    # tool definitions ride along or agentic clients go text-only upstream
    atools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        f = t.get("function") if t.get("type") == "function" else t
        if not isinstance(f, dict) or not f.get("name"):
            continue
        at = {"name": f["name"]}
        if f.get("description") is not None:
            at["description"] = f["description"]
        if isinstance(f.get("parameters"), dict):
            at["input_schema"] = f["parameters"]
        atools.append(at)
    if atools:
        out["tools"] = atools
    choice = body.get("tool_choice")
    if choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        name = (choice.get("function") or {}).get("name")
        if name:
            out["tool_choice"] = {"type": "tool", "name": name}
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
    tool_calls = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text += block.get("text") or ""
        elif block.get("type") == "tool_use" and block.get("name"):
            tool_calls.append({
                "id": block.get("id") or "call_%d" % len(tool_calls),
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block.get("input") or {}),
                },
            })
    usage = payload.get("usage") or {}
    out_usage = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
    }
    # keep prompt-cache reads: without this every anthropic reply looks
    # like a full-price call in usage stats
    cached_raw = usage.get("cache_read_input_tokens")
    if cached_raw is not None:
        try:
            cached = int(cached_raw)
            if cached >= 0:
                out_usage["prompt_tokens_details"] = {"cached_tokens": cached}
        except (TypeError, ValueError):
            pass
    if tool_calls:
        finish = "tool_calls"
    else:
        finish = _STOP_MAP.get(payload.get("stop_reason"), "stop")
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": payload.get("id") or "chatcmpl-%s" % uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model") or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": out_usage,
    }


class StreamTranslator:
    """Feed Anthropic SSE lines in, get OpenAI SSE chunks out."""

    def __init__(self, model):
        self.model = model
        self.id = "chatcmpl-%s" % uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.sent_role = False
        self.done = False
        # native anthropic usage seen mid-stream (message_start carries
        # input tokens, message_delta carries output tokens)
        self._seen_usage = {}
        # streamed tool calls by block index: {id, name, args}
        self._tools = {}

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
        if etype == "message_start":
            for key, val in ((ev.get("message") or {}).get("usage") or {}).items():
                if isinstance(val, int):
                    self._seen_usage[key] = val
            if not self.sent_role:
                self.sent_role = True
                out += self._emit(self._chunk({"role": "assistant", "content": ""}))
        elif etype == "content_block_start":
            block = ev.get("content_block") or {}
            if block.get("type") == "tool_use" and block.get("name"):
                idx = ev.get("index", 0)
                self._tools[idx] = {"id": block.get("id") or "call_%d" % idx,
                                    "name": block["name"], "args": ""}
                if not self.sent_role:
                    self.sent_role = True
                    out += self._emit(self._chunk({"role": "assistant", "content": ""}))
                out += self._emit(self._chunk({"tool_calls": [{
                    "index": idx, "id": self._tools[idx]["id"],
                    "type": "function",
                    "function": {"name": block["name"], "arguments": ""}}]}))
        elif etype == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "input_json_delta":
                idx = ev.get("index", 0)
                frag = delta.get("partial_json") or ""
                if idx in self._tools and frag:
                    self._tools[idx]["args"] += frag
                    out += self._emit(self._chunk({"tool_calls": [{
                        "index": idx,
                        "function": {"arguments": frag}}]}))
                return out
            text = delta.get("text") or delta.get("partial_json") or ""
            if text:
                if not self.sent_role:
                    self.sent_role = True
                    out += self._emit(self._chunk({"role": "assistant", "content": ""}))
                out += self._emit(self._chunk({"content": text}))
        elif etype == "message_delta":
            for key, val in (ev.get("usage") or {}).items():
                if isinstance(val, int):
                    self._seen_usage[key] = val
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

    def usage(self):
        """Tokens seen in this stream, as pin/pout/cached ({} if none)."""
        from . import usage as U
        return U.extract_anthropic_usage(self._seen_usage)


def chat_to_responses(body):
    """Build an OpenAI Responses-API request from a chat request.

    For models served ONLY on /responses (opencode Zen's Muse Spark):
    text flattens to input text, images ride structured (input_image),
    token/temperature/tool knobs mapped.
    """
    lines = []
    items = []
    for msg in body.get("messages") or []:
        text = _text_of(msg.get("content"))
        role = msg.get("role")
        # keep prior tool traffic in the transcript so multi-turn agent
        # loops don't lose what tools ran and what they returned
        calls = []
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                if fn.get("name"):
                    calls.append("%s(%s)" % (fn["name"], fn.get("arguments") or ""))
        # images ride structured; the Responses API takes input_image parts
        # (data: or https: urls). Dropping them made vision requests silently
        # text-only, which reads as the model ignoring the picture.
        images = []
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                url = None
                if part.get("type") == "image_url":
                    iu = part.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                elif part.get("type") == "input_image":
                    url = part.get("image_url")
                if isinstance(url, str) and url:
                    images.append(url)
        if images:
            parts = []
            if text:
                parts.append({"type": "input_text", "text": text})
            parts.extend({"type": "input_image", "image_url": u} for u in images)
            items.append({"role": role if role in ("user", "assistant", "system") else "user",
                          "content": parts})
            continue
        if role == "user":
            lines.append(text)
        elif role == "system":
            lines.append("System: " + text)
        elif role == "assistant":
            lines.append("Assistant: " + text)
            lines.extend("Assistant called: " + c for c in calls)
        elif role == "tool":
            lines.append("Tool result: " + text)
        else:
            lines.append(text)
        if not text and not calls:
            lines.pop()
    if items:
        # mixed text+vision: text history stays readable, vision structured
        out = {"model": body.get("model"), "input": (
            [{"role": "user", "content": [{"type": "input_text", "text": "\n\n".join(lines)}]}]
            + items if lines else items) or "hi"}
    else:
        out = {"model": body.get("model"), "input": "\n\n".join(lines) or "hi"}
    want = body.get("max_tokens")
    if want in (None, 0, ""):
        want = body.get("max_completion_tokens")
    if want not in (None, 0, ""):
        out["max_output_tokens"] = want
    for key in ("temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]
    # agentic clients send tools; without them the model can only talk
    # about calling tools, never actually call them. Shapes are nearly
    # identical on both sides (function wrapper differs).
    rtools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        f = t.get("function") if t.get("type") == "function" else t
        if not isinstance(f, dict) or not f.get("name"):
            continue
        r = {"type": "function", "name": f["name"]}
        for k in ("description", "parameters", "strict"):
            if f.get(k) is not None:
                r[k] = f[k]
        rtools.append(r)
    if rtools:
        out["tools"] = rtools
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            name = (choice.get("function") or {}).get("name")
            out["tool_choice"] = {"type": "function", "name": name} if name else "auto"
        else:
            out["tool_choice"] = "auto"
    elif choice in ("auto", "none", "required"):
        out["tool_choice"] = choice
    if body.get("parallel_tool_calls") is not None:
        out["parallel_tool_calls"] = bool(body.get("parallel_tool_calls"))
    return out


def responses_to_chat(payload, model):
    """Fold a Responses-API object into an OpenAI chat.completion."""
    text = ""
    tool_calls = []
    if isinstance(payload, dict):
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                        text += part.get("text") or ""
            elif item.get("type") == "function_call" and item.get("name"):
                args = item.get("arguments")
                tool_calls.append({
                    "id": item.get("call_id") or item.get("id") or "call_%d" % len(tool_calls),
                    "type": "function",
                    "function": {
                        "name": item.get("name"),
                        "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                    },
                })
    usage = (payload or {}).get("usage") or {}
    prompt = usage.get("input_tokens", 0)
    completion = usage.get("output_tokens", 0)
    # upstream status 'incomplete' means the output budget ran out mid-write
    # (Spark-class models burn max_tokens on reasoning); surface as 'length'
    # so clients don't read an empty/cut reply as a clean 'stop'.
    if tool_calls:
        finish = "tool_calls"
    else:
        finish = "stop" if (payload or {}).get("status", "completed") == "completed" else "length"
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": (payload or {}).get("id") or "chatcmpl-%s" % uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": (payload or {}).get("model") or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }
