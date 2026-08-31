"""Minimal HTTP helper over urllib.

Sends a browser-like User-Agent: many providers sit behind Cloudflare, which
403s both Python's default agent AND `curl/*` with an HTML challenge page.
"""

import json
import urllib.error
import urllib.request

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")
DEFAULT_TIMEOUT = 30


class Resp:
    __slots__ = ("status", "body", "headers", "error")

    def __init__(self, status, body=b"", headers=None, error=None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        self.error = error

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8", "replace"))
        except ValueError:
            return None

    def text(self, limit=2000):
        return self.body.decode("utf-8", "replace")[:limit]

    @property
    def ok(self):
        return 200 <= self.status < 300


def base_headers(extra=None):
    h = {"User-Agent": UA, "Accept": "application/json"}
    if extra:
        h.update(extra)
    return h


def request(method, url, headers=None, body=None, timeout=DEFAULT_TIMEOUT):
    data = None
    hdrs = base_headers(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Resp(r.status, r.read(), dict(r.headers))
    except urllib.error.HTTPError as e:
        # HTTPError is a file object; close it or Python warns about the socket
        try:
            err_body = e.read()
            err_headers = dict(e.headers or {})
        finally:
            e.close()
        return Resp(e.code, err_body, err_headers)
    except Exception as e:  # timeouts, DNS, TLS, connection refused
        return Resp(0, b"", {}, error=repr(e))


def get(url, headers=None, timeout=DEFAULT_TIMEOUT):
    return request("GET", url, headers=headers, timeout=timeout)


def post(url, body, headers=None, timeout=DEFAULT_TIMEOUT):
    return request("POST", url, headers=headers, body=body, timeout=timeout)


def openai_auth(key):
    return {"Authorization": "Bearer %s" % key}


def anthropic_auth(key):
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}
