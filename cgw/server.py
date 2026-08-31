"""The gateway HTTP server. OpenAI-compatible surface on 127.0.0.1.

No inbound auth: binds loopback only. Anything on this box can use it.
"""

import json
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config as C
from . import http as H
from . import keyring as K
from . import translate as T

RETRY_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504, 529)

# refuse absurd inbound bodies rather than reading them into memory
MAX_BODY_BYTES = 32 * 1024 * 1024

# don't stat the config file more than this often (seconds)
MTIME_CHECK_INTERVAL = 1.0


class State:
    """Config + keyrings, reloaded from disk when the file changes."""

    def __init__(self, path=None):
        self.path = path or C.CONFIG_PATH
        self.registry = K.Registry()
        self.lock = threading.Lock()
        self.cfg = C.load(self.path)
        self.mtime = C.mtime(self.path)
        self.checked = time.time()
        self.recent = []
        self.reload_error = None

    def refresh(self):
        """Return the current config, reloading if the file changed.

        Readers must never see a half-swapped config, so the swap and every
        read of self.cfg happen under the lock. A corrupt file keeps the
        in-memory config and records the error instead of blanking providers.
        """
        now = time.time()
        with self.lock:
            if now - self.checked < MTIME_CHECK_INTERVAL:
                return self.cfg
            self.checked = now
            m = C.mtime(self.path)
            if m == self.mtime:
                return self.cfg
            try:
                cfg = C.load(self.path)
            except C.ConfigError as e:
                self.mtime = m  # don't retry the same broken file every second
                self.reload_error = str(e)
                return self.cfg
            self.cfg = cfg
            self.mtime = m
            self.reload_error = None
            return self.cfg

    def providers(self):
        return [p for p in self.refresh().get("providers", []) if p.get("enabled", True)]

    def route(self, model_id):
        """'name/model' -> (provider, upstream_model). Falls back to a unique
        bare model name if exactly one provider offers it. Disabled models
        are never routable, in either form."""
        provs = self.providers()
        if "/" in model_id:
            head, rest = model_id.split("/", 1)
            for p in provs:
                if p["name"] == head:
                    meta = (p.get("models") or {}).get(rest)
                    if meta is not None and not meta.get("enabled", True):
                        return None, None
                    return p, rest
        hits = [p for p in provs
                if (p.get("models") or {}).get(model_id, {}).get("enabled", True)
                and model_id in p["models"]]
        if len(hits) == 1:
            return hits[0], model_id
        return None, None

    def log(self, entry):
        with self.lock:
            self.recent.append(entry)
            del self.recent[:-100]

    def tail(self, n=20):
        with self.lock:
            return list(self.recent[-n:])


def _upstream_url(provider, kind):
    base = provider["base_url"].rstrip("/")
    if provider.get("flavor") == "anthropic" and kind == "chat":
        return base + "/messages"
    return base + {"chat": "/chat/completions", "completions": "/completions",
                   "embeddings": "/embeddings"}[kind]


def _auth_for(provider, key):
    if provider.get("flavor") == "anthropic":
        return H.anthropic_auth(key)
    return H.openai_auth(key)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CG/1.0"
    state = None  # injected

    def log_message(self, fmt, *args):
        pass  # no request log spam; TUI shows the ring buffer

    # ---------- plumbing ----------

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        """Parse the JSON body. Returns (obj, error_message).

        A malformed body used to become {}, which then failed as
        404 "no provider serves model ''" — a misleading error for what is
        really a bad request.
        """
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            return None, "invalid Content-Length header"
        if n < 0:
            return None, "invalid Content-Length header"
        if n > MAX_BODY_BYTES:
            return None, "request body too large (%d bytes, limit %d)" % (n, MAX_BODY_BYTES)
        if not n:
            # no Content-Length: chunked or empty. Read what's there, bounded.
            if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
                raw = self._read_chunked()
                if raw is None:
                    return None, "malformed chunked body"
            else:
                return None, "empty request body"
        else:
            raw = self.rfile.read(n)
            if len(raw) < n:
                return None, "truncated request body"
        if not raw.strip():
            return None, "empty request body"
        try:
            obj = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as e:
            return None, "invalid JSON body: %s" % e
        if not isinstance(obj, dict):
            return None, "request body must be a JSON object"
        return obj, None

    def _read_chunked(self):
        """Minimal chunked-body reader, bounded by MAX_BODY_BYTES."""
        out = bytearray()
        while True:
            line = self.rfile.readline(64)
            if not line:
                return None
            try:
                size = int(line.split(b";", 1)[0].strip() or b"0", 16)
            except ValueError:
                return None
            if size == 0:
                self.rfile.readline(8)  # trailing CRLF
                return bytes(out)
            if len(out) + size > MAX_BODY_BYTES:
                return None
            chunk = self.rfile.read(size)
            if len(chunk) < size:
                return None
            out += chunk
            self.rfile.readline(8)  # CRLF after each chunk

    def _error(self, status, msg, extra=None):
        obj = {"error": {"message": msg, "type": "cg_error"}}
        if extra:
            obj["error"].update(extra)
        self._send_json(status, obj)

    # ---------- routes ----------

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/v1/models", "/models"):
            return self._models()
        if path == "/healthz":
            return self._healthz()
        if path in ("/v1/logs", "/logs"):
            return self._logs()
        if path == "/":
            return self._send_json(200, {"service": "cg", "endpoints": ["/v1/models", "/v1/chat/completions", "/v1/logs", "/healthz"]})
        return self._error(404, "not found: %s" % path)

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/v1/revive", "/revive"):
            return self._revive()
        kinds = {
            "/v1/chat/completions": "chat",
            "/chat/completions": "chat",
            "/v1/completions": "completions",
            "/v1/embeddings": "embeddings",
        }
        if path not in kinds:
            return self._error(404, "not found: %s" % path)
        return self._proxy(kinds[path])

    def _revive(self):
        """Clear dead/cooldown key state. Body: {"provider": "name"} or {}.

        Keyring health lives in the server's memory, so the TUI (a separate
        process) needs a way in; without this a 401/403 blip killed a key
        until the next restart.
        """
        body, err = self._read_body()
        if err and err != "empty request body":
            return self._error(400, err)
        name = (body or {}).get("provider") or None
        n = self.state.registry.revive(name)
        self._send_json(200, {"ok": True, "revived": n,
                              "provider": name or "all",
                              "keys": self.state.registry.snapshot()})

    def _models(self):
        out = []
        for p in self.state.providers():
            for mid, meta in (p.get("models") or {}).items():
                if not meta.get("enabled", True):
                    continue  # toggled off: don't advertise what won't route
                entry = {
                    "id": "%s/%s" % (p["name"], mid),
                    "object": "model",
                    "created": 0,
                    "owned_by": p["name"],
                }
                if meta.get("context") is not None:
                    try:
                        entry["context"] = int(meta["context"])
                    except (TypeError, ValueError):
                        pass
                out.append(entry)
        out.sort(key=lambda m: m["id"])
        self._send_json(200, {"object": "list", "data": out})

    def _logs(self):
        """Recent per-request ring buffer entries (incl. the key label used)."""
        try:
            n = int(parse_qs(urlparse(self.path).query).get("n", ["20"])[0])
        except (ValueError, TypeError):
            n = 20
        n = max(1, min(n, 100))
        self._send_json(200, {"entries": self.state.tail(n)})

    def _healthz(self):
        provs = []
        for p in self.state.refresh().get("providers", []):
            ring = self.state.registry.get(p)
            provs.append(
                {
                    "name": p["name"],
                    "enabled": p.get("enabled", True),
                    "flavor": p.get("flavor"),
                    "base_url": p.get("base_url"),
                    "rotation": p.get("rotation"),
                    "models": len(p.get("models") or {}),
                    "keys": ring.summary(),
                }
            )
        out = {"ok": True, "providers": provs, "recent": self.state.tail()}
        if self.state.reload_error:
            # a corrupt config on disk is not fatal, but it must be visible
            out["ok"] = False
            out["config_error"] = self.state.reload_error
        self._send_json(200, out)

    # ---------- proxying ----------

    def _proxy(self, kind):
        body, err = self._read_body()
        if err or body is None:
            return self._error(400, err or "invalid request body")
        model_id = body.get("model") or ""
        if not model_id:
            return self._error(400, "request is missing the 'model' field")
        provider, upstream_model = self.state.route(model_id)
        if provider is None:
            return self._error(404, "no provider serves model %r" % model_id)

        ring = self.state.registry.get(provider)
        order = ring.try_order()
        if not order:
            return self._error(503, "provider %s has no usable keys" % provider["name"])

        body = dict(body)
        body["model"] = upstream_model
        streaming = bool(body.get("stream"))
        anthropic = provider.get("flavor") == "anthropic"
        payload = T.openai_to_anthropic(body) if (anthropic and kind == "chat") else body

        url = _upstream_url(provider, kind)
        last = None
        for attempt, idx in enumerate(order):
            key = ring.key_at(idx)
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=H.base_headers(dict(_auth_for(provider, key), **{"Content-Type": "application/json"})),
                method="POST",
            )
            started = time.time()
            try:
                resp = urllib.request.urlopen(req, timeout=600)
            except urllib.error.HTTPError as e:
                status = e.code
                try:
                    text = e.read()
                finally:
                    e.close()
                ring.report_failure(idx, status)
                last = (status, text)
                self.state.log(
                    {"t": started, "model": model_id, "key": ring.state[idx].label,
                     "status": status, "ms": int((time.time() - started) * 1000)}
                )
                if status in RETRY_STATUSES or status in K.DEAD_STATUSES:
                    continue
                break
            except Exception as e:
                ring.report_failure(idx, 0)
                last = (502, json.dumps({"error": {"message": repr(e)}}).encode())
                self.state.log({"t": started, "model": model_id, "key": ring.state[idx].label,
                                "status": 0, "ms": int((time.time() - started) * 1000)})
                continue

            ring.report_success(idx)
            self.state.log({"t": started, "model": model_id, "key": ring.state[idx].label,
                            "status": 200, "ms": int((time.time() - started) * 1000)})
            with resp:
                if streaming:
                    return self._relay_stream(resp, anthropic and kind == "chat", model_id, attempt)
                return self._relay_body(resp, anthropic and kind == "chat", model_id)

        status, text = last if last else (502, b'{"error":{"message":"no attempt made"}}')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(text)))
        self.send_header("X-CG-Keys-Tried", str(len(order)))
        self.end_headers()
        self.wfile.write(text)

    def _relay_body(self, resp, translate, model_id):
        raw = resp.read()
        if translate:
            try:
                obj = T.anthropic_to_openai(json.loads(raw.decode("utf-8", "replace")), model_id)
                raw = json.dumps(obj).encode()
            except ValueError:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _relay_stream(self, resp, translate, model_id, attempt):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        if translate:
            tr = T.StreamTranslator(model_id)
            buf = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    out = tr.feed(line.decode("utf-8", "replace"))
                    if out:
                        self.wfile.write(out)
                        self.wfile.flush()
            tail = tr.finish()
            if tail:
                self.wfile.write(tail)
        else:
            while True:
                chunk = resp.read(2048)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()


def serve(path=None, host=None, port=None):
    state = State(path)
    listen = state.refresh().get("listen") or {}
    host = host if host is not None else listen.get("host", "127.0.0.1")
    # port 0 is a valid request for "pick a free port", so test for None
    port = int(port if port is not None else listen.get("port", C.DEFAULT_PORT))
    handler = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd, state
