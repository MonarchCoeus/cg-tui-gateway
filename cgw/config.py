"""Config storage for CG (Coeus Gateway).

Plain JSON at ~/.config/cg/config.json, mode 0600, atomic writes.
No env-var indirection: keys live in the file.
"""

import json
import os
import tempfile

CONFIG_DIR = os.path.expanduser("~/.config/cg")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")

DEFAULT_PORT = 20185
ROTATIONS = ("fill_first", "round_robin")
FLAVORS = ("openai", "anthropic", "unknown")


class ConfigError(Exception):
    """An existing config file could not be read or parsed.

    Raised instead of quietly returning an empty config: a caller that saves
    after a silent fallback would overwrite the real providers and keys.
    """


def as_int(val, default):
    """Coerce a hand-edited value to int, falling back instead of raising.

    A single typo in the JSON ("128k", "", null) used to abort the whole
    load, which surfaced as "providers: 0" and got written back on the next
    save. Bad fields degrade to the default; nothing else is lost. The
    default may be None (callers that just want to know "is this a number"),
    in which case None is returned on failure instead of raising.
    """
    if isinstance(val, bool) or val is None:
        return int(default) if default is not None else None
    try:
        return int(val)
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(val).strip()))
    except (TypeError, ValueError):
        return int(default) if default is not None else None


def default_config():
    return {
        "version": 1,
        "listen": {"host": "127.0.0.1", "port": DEFAULT_PORT},
        "providers": [],
    }


def new_provider(name, base_url, keys, rotation="fill_first", flavor="unknown"):
    """keys: list of raw key strings, or list of dicts already shaped.

    Keys and the URL are normalised here, so every entry point (CLI, TUI,
    hand-edited file) gets the same paste-accident handling.
    """
    shaped = []
    for i, k in enumerate(keys):
        if isinstance(k, dict):
            cleaned = clean_key(k.get("key", ""))
            if not cleaned:
                continue
            shaped.append(
                {
                    "key": cleaned,
                    "label": k.get("label") or "k%d" % (i + 1),
                    "enabled": bool(k.get("enabled", True)),
                }
            )
        else:
            cleaned = clean_key(k)
            if not cleaned:
                continue
            shaped.append({"key": cleaned, "label": "k%d" % (i + 1), "enabled": True})
    return {
        "name": name,
        "base_url": clean_url(base_url),
        "flavor": flavor if flavor in FLAVORS else "unknown",
        "keys": shaped,
        "rotation": rotation if rotation in ROTATIONS else "fill_first",
        "enabled": True,
        "models": {},
    }


def clean_key(raw):
    """Normalise a pasted API key.

    Paste accidents are the single most common cause of a "valid key rejected"
    report, and every one of them is silent. Handled here rather than at each
    call site: surrounding whitespace and newlines, wrapping quotes, a copied
    `Authorization: Bearer ` prefix, a trailing comma from a JSON snippet, and
    zero-width/BOM characters that survive a copy from a web page.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    # zero-width space/joiner, BOM, non-breaking space
    for junk in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0"):
        s = s.replace(junk, "")
    s = s.strip()
    low = s.lower()
    for prefix in ("authorization:", "x-api-key:", "api-key:"):
        if low.startswith(prefix):
            s = s[len(prefix):].strip()
            low = s.lower()
    if low.startswith("bearer "):
        s = s[7:].strip()
    while s and s[-1] in ",;":
        s = s[:-1].strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def clean_url(raw):
    """Normalise a pasted base URL.

    Accepts what people actually paste: a full chat-completions endpoint, a
    trailing slash, a bare host with no scheme, or a copied URL with
    surrounding quotes. Returns a base URL suitable for appending /models.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    for junk in ("\u200b", "\ufeff", "\u00a0"):
        s = s.replace(junk, "")
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    s = s.rstrip("/")
    # a pasted endpoint is a base URL with a known suffix glued on
    for suffix in ("/chat/completions", "/completions", "/messages",
                   "/models", "/responses"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = s.rstrip("/")
    if s and "://" not in s:
        s = "https://" + s
    return s


def normalize(cfg):
    """Fill in anything missing so older/hand-edited files still load."""
    base = default_config()
    if not isinstance(cfg, dict):
        return base
    out = dict(base)
    out["version"] = cfg.get("version", 1)
    listen = cfg.get("listen") or {}
    out["listen"] = {
        "host": listen.get("host", "127.0.0.1"),
        "port": as_int(listen.get("port", DEFAULT_PORT), DEFAULT_PORT),
    }
    provs = []
    for p in cfg.get("providers") or []:
        if not isinstance(p, dict) or not p.get("name"):
            continue
        q = new_provider(
            p["name"],
            p.get("base_url", ""),
            p.get("keys") or [],
            p.get("rotation", "fill_first"),
            p.get("flavor", "unknown"),
        )
        q["enabled"] = bool(p.get("enabled", True))
        models = {}
        for mid, meta in (p.get("models") or {}).items():
            if not isinstance(meta, dict):
                meta = {}
            entry = {}
            for field in ("reasoning", "vision"):
                if meta.get(field) is not None:
                    entry[field] = bool(meta[field])
            for field in ("reasoning_note", "vision_note", "checked",
                          "available", "available_note", "endpoint"):
                if meta.get(field) is not None:
                    entry[field] = meta[field]
            # a manually-set context window (tokens); None means unset
            if meta.get("context") is not None:
                entry["context"] = as_int(meta.get("context"), None)
            if meta.get("context_source") is not None:
                entry["context_source"] = meta["context_source"]
            # per-model enable flag survives load/save round-trips
            if meta.get("enabled") is not None:
                entry["enabled"] = bool(meta["enabled"])
            models[mid] = entry
        q["models"] = models
        provs.append(q)
    out["providers"] = provs
    return out


def load(path=None, strict=True):
    """Read the config.

    Missing file -> fresh default. Unreadable/corrupt existing file -> raises
    ConfigError when strict (the default), because the alternative is handing
    back an empty config that a later save would write over the real keys.
    Pass strict=False only where losing the file's contents is acceptable.
    """
    path = path or CONFIG_PATH
    if not os.path.exists(path):
        return default_config()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return normalize(json.load(fh))
    except (ValueError, OSError) as e:
        if strict:
            raise ConfigError("%s: %s" % (path, e)) from e
        return default_config()


def save(cfg, path=None):
    path = path or CONFIG_PATH
    # a bare relative filename has no dirname; makedirs("") would raise
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def mtime(path=None):
    path = path or CONFIG_PATH
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0


def find(cfg, name):
    for p in cfg.get("providers", []):
        if p["name"] == name:
            return p
    return None


# Metadata that cost a real request (or a human) to establish. A re-listing
# must not throw it away — the CLI and the TUI used to disagree about which
# of these survived a refresh.
EARNED_SOURCES = ("manual",)


def is_earned(meta):
    if not isinstance(meta, dict):
        return False
    if meta.get("source") in EARNED_SOURCES:
        return True
    return meta.get("reasoning") is not None or meta.get("vision") is not None \
        or meta.get("context") is not None


def merge_models(old, new):
    """Fresh listing + previously earned facts, earned facts winning.

    Models that vanished from the listing are dropped unless they were added
    by hand ('manual'), since those exist precisely because the provider does
    not list them.
    """
    old = old or {}
    new = new or {}
    out = dict(new)
    for mid, meta in old.items():
        if not is_earned(meta) and "enabled" not in meta:
            continue
        if mid in out:
            merged = dict(out[mid])
            merged.update(meta)
            out[mid] = merged
        elif meta.get("source") == "manual":
            out[mid] = dict(meta)
        elif "enabled" in meta and mid not in out:
            # toggled-off model no longer listed upstream: keep it disabled
            # (it was visible to the user and they turned it off; a re-listing
            # must not silently resurrect it as enabled again)
            out[mid] = dict(meta)
    return out
