"""Per-request token accounting for CG.

Upstream replies already carry token counts; the gateway used to throw them
away and log only timing. This module extracts, persists, and summarizes
them: tokens in / out, cache hits, per model and per provider.

Storage is one JSON line per successful request, next to the config file
(usage.jsonl) — plain text, no database, in keeping with CG's no-deps rule.
Entries without usage info (providers that report none) are kept and counted
as unknown, never as zero.
"""

import json
import os
import time
from typing import Union

MAX_BYTES = 12 * 1024 * 1024
TRIM_BYTES = 8 * 1024 * 1024


def usage_path_for(config_path):
    """usage.jsonl lives next to the config file, so --config keeps working."""
    base = os.path.dirname(os.path.abspath(config_path or ""))
    return os.path.join(base, "usage.jsonl")


def _num(val):
    if isinstance(val, bool):
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def _cached_from(details):
    """Pull a cached-tokens figure out of a details object (dict or list)."""
    if isinstance(details, dict):
        details = [details]
    if not isinstance(details, list):
        return None
    for d in details:
        if isinstance(d, dict):
            n = _num(d.get("cached_tokens"))
            if n is not None:
                return n
    return None


def _parts(pin, pout, cached):
    out = {}
    if pin is not None:
        out["pin"] = pin
    if pout is not None:
        out["pout"] = pout
    if cached is not None:
        out["cached"] = cached
    return out


def extract_openai_usage(usage):
    """chat.completion / chat chunk usage: prompt/completion_tokens + details."""
    if not isinstance(usage, dict):
        return {}
    det = usage.get("prompt_tokens_details")
    cached = _cached_from(det)
    if cached is None:
        # some proxies report the hit count as a top-level field instead
        cached = _num(usage.get("prompt_cache_hit_tokens"))
    return _parts(_num(usage.get("prompt_tokens")),
                   _num(usage.get("completion_tokens")), cached)


def extract_responses_usage(usage):
    """/responses object usage: input/output_tokens + input details."""
    if not isinstance(usage, dict):
        return {}
    return _parts(_num(usage.get("input_tokens")),
                   _num(usage.get("output_tokens")),
                   _cached_from(usage.get("input_tokens_details")))


def extract_anthropic_usage(usage):
    """Native Anthropic message usage (used for streamed SSE accounting)."""
    if not isinstance(usage, dict):
        return {}
    return _parts(_num(usage.get("input_tokens")),
                   _num(usage.get("output_tokens")),
                   _num(usage.get("cache_read_input_tokens")))


def extract_chunk_usage(obj):
    """A decoded streaming data chunk that may carry a trailing usage object."""
    if not isinstance(obj, dict):
        return {}
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return {}
    out = extract_openai_usage(usage)
    if not out:
        out = extract_responses_usage(usage)
    if not out:
        out = extract_anthropic_usage(usage)
    return out


def append(path, entry):
    """Best-effort JSONL append. Accounting must never break a request."""
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if os.path.getsize(path) > MAX_BYTES:
            trim(path)
    except Exception:
        pass  # OSError and serialization alike: never fail the request


def trim(path, keep_bytes=TRIM_BYTES):
    """Rewrite the file keeping roughly the last keep_bytes of lines."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) <= keep_bytes:
            return
        with open(path, "rb") as fh:
            fh.seek(-keep_bytes, os.SEEK_END)
            tail = fh.read().split(b"\n", 1)[-1]
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(tail if tail.endswith(b"\n") else tail + b"\n")
        os.replace(tmp, path)
    except OSError:
        pass


def load(path, since=None):
    """Read entries back; corrupt lines are skipped, never fatal."""
    entries = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and (since is None or (obj.get("t") or 0) >= since):
                    entries.append(obj)
    except OSError:
        pass
    return entries


def _blank():
    return {"reqs": 0, "errs": 0, "pin": 0, "pout": 0, "cached": 0, "unknown": 0}


def _known(pin, pout):
    """An entry is token-known when either half is present; the missing half
    counts as 0 (render and summary must agree — they diverged here once)."""
    return isinstance(pin, int) or isinstance(pout, int)


# picker windows, shortest first: (label, seconds)
WINDOWS = (
    ("15min", 900),
    ("30min", 1800),
    ("1h", 3600),
    ("3h", 10800),
    ("6h", 21600),
    ("12h", 43200),
    ("24h", 86400),
    ("3d", 3 * 86400),
    ("7d", 7 * 86400),
    ("30d", 30 * 86400),
)
WINDOW_SECS = dict(WINDOWS)


def hermes_sessions(db_path=None, limit=100):
    """Hermes chat sessions, newest first: id, name, start/end, msg count.

    Read-only open of the Hermes state.db; any failure (missing file,
    locked, Hermes absent) yields [] instead of breaking the caller.
    """
    import sqlite3

    if db_path is None:
        db_path = os.path.join(os.path.expanduser("~"), ".hermes", "state.db")
    out = []
    try:
        con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True, timeout=2)
        try:
            cur = con.execute(
                "SELECT id, display_name, started_at, ended_at,"
                " message_count, source FROM sessions"
                " ORDER BY started_at DESC LIMIT ?", (limit,))
            rows = cur.fetchall()
            first_words = {}
            if rows:
                ids = [r[0] for r in rows]
                q = ("SELECT m.session_id, m.content FROM messages m JOIN"
                     " (SELECT session_id, MIN(id) AS first FROM messages"
                     " WHERE role='user' GROUP BY session_id) f"
                     " ON f.session_id=m.session_id AND f.first=m.id"
                     " WHERE m.session_id IN (%s)" % ",".join("?" * len(ids)))
                try:
                    for sid, content in con.execute(q, ids).fetchall():
                        text = " ".join(str(content or "").split())
                        if text:
                            first_words[sid] = text
                except Exception:
                    pass
            for sid, name, started, ended, count, source in rows:
                out.append({"id": sid or "?", "name": name or sid or "?",
                            "title": first_words.get(sid or "", ""),
                            "started": started or 0, "ended": ended,
                            "count": count or 0, "source": source or ""})
        finally:
            con.close()
    except Exception:
        pass
    return out


def session_cutoff(host="127.0.0.1", port=20185, timeout=2):
    """Gateway boot time, or None when it is not running.

    Defines the 'session' window: everything since the current server
    process started. A restart starts a new session.
    """
    import urllib.request

    try:
        with urllib.request.urlopen("http://%s:%s/healthz" % (host, port), timeout=timeout) as r:
            obj = json.loads(r.read().decode("utf-8", "replace"))
        ts = obj.get("started")
        return float(ts) if ts else None
    except Exception:
        return None


def aggregate(entries):
    """Totals grouped by model and by provider ('name/model' -> 'name')."""
    by_model, by_provider = {}, {}
    for e in entries:
        mid = e.get("model") or "?"
        prov = mid.split("/", 1)[0] if "/" in mid else "?"
        ok = e.get("status") == 200
        for table, key in ((by_model, mid), (by_provider, prov)):
            st = table.setdefault(key, _blank())
            st["reqs"] += 1
            if not ok:
                st["errs"] += 1
                continue
            if _known(e.get("pin"), e.get("pout")):
                st["pin"] += e.get("pin") or 0
                st["pout"] += e.get("pout") or 0
                if isinstance(e.get("cached"), int):
                    st["cached"] += e["cached"]
            else:
                st["unknown"] += 1
    return {"by_model": by_model, "by_provider": by_provider}


def _human(n):
    """Compact counts: 1500 -> '1.5k', 2000000 -> '2.0M', None -> '-'."""
    if n is None:
        return "-"
    if n >= 1_000_000:
        return "%.1fM" % (n / 1_000_000)
    if n >= 1000:
        return "%.1fk" % (n / 1000)
    return str(n)


def hit_rate(stats):
    """Share of input tokens served from cache, or None when unknown."""
    pin, cached = stats.get("pin"), stats.get("cached")
    if not pin:
        return None
    return 100.0 * cached / pin


def summary_for(entries, key, windows=("7d", "30d")):
    """Per-window stats for one scope: a 'prov/model' id or a bare provider.

    windows items may be a WINDOWS label ('15min'...'30d'), a day count
    (int, legacy), a (label, seconds) pair, or a (label, seconds, since,
    until) quad — the last pins an explicit span (Hermes sessions that
    ended long ago). Returns [(label, stats)].
    A provider scope sums every model id starting with 'provider/'.
    """
    now = time.time()
    norm = []
    for w in windows:
        if isinstance(w, str):
            norm.append((w, WINDOW_SECS[w], None, None))
        elif isinstance(w, int):
            norm.append(("%dd" % w, w * 86400, None, None))
        elif len(w) > 3:
            norm.append((w[0], w[1], w[2], w[3]))
        elif len(w) > 2:
            norm.append((w[0], w[1], w[2], None))
        else:
            norm.append((w[0], w[1], None, None))
    out = []
    for label, secs, since, until in norm:
        cutoff = since if since is not None else now - secs
        total = _blank()
        for e in entries:
            t = e.get("t") or 0
            if t < cutoff or (until is not None and t > until):
                continue
            mid = e.get("model") or "?"
            if mid != key and not mid.startswith(key + "/"):
                continue
            ok = e.get("status") == 200
            total["reqs"] += 1
            if not ok:
                total["errs"] += 1
                continue
            if _known(e.get("pin"), e.get("pout")):
                total["pin"] += e.get("pin") or 0
                total["pout"] += e.get("pout") or 0
                if isinstance(e.get("cached"), int):
                    total["cached"] += e["cached"]
            else:
                total["unknown"] += 1
        out.append((label, total))
    return out


def _row_cells(st):
    """Render cells; a row with no counts at all reads n/a, never 0."""
    if not st["pin"] and not st["pout"] and not st["cached"]:
        return ("n/a", "n/a", "n/a", "n/a")
    rate = hit_rate(st)
    return (_human(st["pin"]), _human(st["pout"]), _human(st["cached"]),
            ("%.0f%%" % rate) if rate else "-")


def render(entries, days: Union[int, str] = 7, by="model", window=None):
    """Plain-text usage table for `cg stats`."""
    until = None
    if window is not None:
        label = window[0]
        if len(window) > 3:
            _secs, since, until = window[1], window[2], window[3]
            cutoff = since
        elif len(window) > 2:
            _secs, since = window[1], window[2]
            cutoff = since
        else:
            cutoff = time.time() - window[1]
    elif isinstance(days, str):
        label, secs = days, WINDOW_SECS[days]
        cutoff = time.time() - secs
    else:
        label, secs = "%dd" % days, days * 86400
        cutoff = time.time() - secs
    window = [e for e in entries
              if (e.get("t") or 0) >= cutoff
              and (until is None or (e.get("t") or 0) <= until)]
    if not window:
        if label.startswith("session"):
            return "no usage recorded in %s." % label
        return "no usage recorded in the last %s." % label
    table = aggregate(window)["by_provider" if by == "provider" else "by_model"]
    rows = sorted(table.items(), key=lambda kv: -(kv[1]["pin"] + kv[1]["pout"]))
    head = "%-42s %5s %4s %8s %8s %8s %6s" % (
        by, "reqs", "err", "in", "out", "cached", "hit%")
    lines = ["usage — last %s (by %s)" % (label, by), head, "-" * len(head)]
    total = _blank()
    for name, st in rows:
        for k in total:
            total[k] += st[k]
        pin, pout, cached, hit = _row_cells(st)
        lines.append("%-42s %5d %4d %8s %8s %8s %6s" % (
            name[:42], st["reqs"], st["errs"], pin, pout, cached, hit))
    lines.append("-" * len(head))
    pin, pout, cached, hit = _row_cells(total)
    lines.append("%-42s %5d %4d %8s %8s %8s %6s" % (
        "TOTAL", total["reqs"], total["errs"], pin, pout, cached, hit))
    if total["unknown"]:
        lines.append("(%d successful requests reported no token counts)" % total["unknown"])
    return "\n".join(lines)
