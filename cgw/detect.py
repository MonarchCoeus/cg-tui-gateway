"""Auto-detection: API flavor, model list, and per-model capabilities.

Two separate passes, on purpose:

  discover()      cheap, listing-only. One HTTP call per provider. Context
                  is read from the listing when volunteered, otherwise left
                  as a flagged default. Never probes.

  inspect_model() on demand, one model at a time. Measures context,
                  reasoning, and vision against the live endpoint.

Nothing here trusts a bare HTTP 200. Providers routinely accept a field,
ignore it, and answer anyway, so every capability verdict needs positive
evidence (a non-empty reasoning trace, extra input tokens for an image).
Anything inconclusive is reported as unknown rather than guessed.
"""

import base64
import re
import struct
import threading
import zlib

from . import http as H

# The vision differential uses a 512x512 image. A REAL vision model costs
# hundreds of input tokens for it (the fake seer adds 512; b.ai/Qwen charge
# ~250-370). Routers that silently drop the image but still count the
# base64 payload move the counter by single digits. Anything under this
# floor is not vision — it's overhead. (256x256 was too small: cheap
# providers billed it at ~60 tokens, just under the floor, so real vision
# read as a silent drop.)
VISION_TOKEN_FLOOR = 64

# Error wording that means "this model is text-only", as opposed to a billing
# or auth failure (which tells us nothing about the model).
VISION_REFUSAL = re.compile(
    r"image|vision|multi-?modal|unsupported content|only supports? text"
    r"|invalid content type|content\[\d+\]",
    re.I,
)

# Statuses that are about the account, not the model: never a capability answer.
ACCOUNT_STATUSES = (401, 402, 403, 429)

# Some models have no separate reasoning field and instead ship the trace
# inline at the head of `content`. vLLM serves these verbatim unless the
# deployment enables a reasoning parser.
INLINE_THINK = re.compile(
    r"<(think|thinking|reasoning|thought)>(.*?)(?:</\1>|$)",
    re.I | re.S,
)

# Some gateways answer HTTP 200 with a canned "top up your account" message
# instead of running the model (AIHubMix does this on exhausted free tiers).
# Treating that as success would report every capability as present, so the
# response is checked for the tell-tale shape before it is believed.
QUOTA_SENTINEL = re.compile(
    r"prevent abuse of free resources|can only try \d+ times|please recharge"
    r"|topup|top ?up your|insufficient balance|free quota after recharging",
    re.I,
)


def _first_text(payload, flavor):
    """Assistant text out of either response shape, or '' if absent."""
    if not isinstance(payload, dict):
        return ""
    if flavor == "anthropic":
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text") or ""
        return ""
    for choice in payload.get("choices") or []:
        msg = choice.get("message") or {}
        if isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


def looks_faked(payload, flavor="openai"):
    """True when a 200 response is a billing notice wearing a completion's hat.

    Two independent signals, either is enough: the text matches a known
    quota/top-up notice, or the response id is marked fake while usage is
    all zeros (no tokens were actually spent).
    """
    if not isinstance(payload, dict):
        return False
    if QUOTA_SENTINEL.search(_first_text(payload, flavor) or ""):
        return True
    rid = str(payload.get("id") or "")
    usage = payload.get("usage") or {}
    spent = sum(int(usage.get(k) or 0) for k in
                ("total_tokens", "completion_tokens", "prompt_tokens",
                 "input_tokens", "output_tokens"))
    return "fake" in rid.lower() and spent == 0


def _dig(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _model_items(payload):
    """Pull a list of model dicts out of whatever shape the provider returned."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("data") or payload.get("models") or payload.get("result") or []
        if isinstance(items, dict):  # {id: {...}} mapping style
            items = [dict(v, id=v.get("id", k)) for k, v in items.items() if isinstance(v, dict)]
    else:
        return []
    out = []
    for it in items:
        if isinstance(it, str):
            out.append({"id": it})
        elif isinstance(it, dict) and (it.get("id") or it.get("name") or it.get("model")):
            d = dict(it)
            d["id"] = d.get("id") or d.get("name") or d.get("model")
            out.append(d)
    return out


def _candidate_bases(base_url):
    """base as given, plus /v1-adjusted variants, deduped in order."""
    base = base_url.rstrip("/")
    cands = [base]
    if base.endswith("/v1"):
        cands.append(base[: -len("/v1")])
    else:
        cands.append(base + "/v1")
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# Listing fields that some providers use to advertise capabilities directly.
# Only trusted when the value is a real bool: a missing key means "unstated",
# not "false".
REASONING_FIELDS = ("reasoning", "supports_reasoning", "is_reasoning",
                    "thinking", "supports_thinking")
VISION_FIELDS = ("vision", "supports_vision", "is_vision", "multimodal",
                 "supports_images", "image_input")

# Context-size fields a provider may publish in its /models listing. Read
# ONLY from the listing — never probed, never defaulted. A provider that
# doesn't advertise a size simply has no context until one is set manually.
CONTEXT_FIELDS = (
    "context_window",
    "context_length",
    "max_context_length",
    "max_context_tokens",
    "max_input_tokens",
    "max_model_len",
    "context_size",
    "limit.context",
    "top_provider.context_length",
    "model_info.max_input_tokens",
    "capabilities.context_length",
)


def context_from_fields(obj):
    """Context window advertised by the provider's listing, or None.

    Only sane positive sizes count (>= 1000 tokens); booleans and tiny
    values are junk, not context.
    """
    if not isinstance(obj, dict):
        return None
    for field in CONTEXT_FIELDS:
        val = _dig(obj, field)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and val >= 1000:
            return int(val)
        if isinstance(val, str) and val.isdigit() and int(val) >= 1000:
            return int(val)
    return None


def stated_facts(meta):
    """Rebuild a listing-like dict from a saved model entry.

    Lets `inspect_model` reuse what the provider already told us about
    capability flags without re-fetching the listing. Only values that were
    actually stated are included — a defaulted verdict is a guess, not a fact.
    """
    meta = meta or {}
    item = {}
    for which in ("reasoning", "vision"):
        if meta.get(which) is not None and "listing" in str(meta.get("%s_note" % which, "")):
            item[which] = bool(meta[which])
    return item


def capability_from_fields(obj, which):
    """True/False from a listing entry's capability flag, or None if unstated."""
    if not isinstance(obj, dict):
        return None
    fields = REASONING_FIELDS if which == "reasoning" else VISION_FIELDS
    for name in fields:
        if name in obj and isinstance(obj[name], bool):
            return obj[name]
    modalities = obj.get("input_modalities") or obj.get("modalities")
    if which == "vision" and isinstance(modalities, list):
        return any(str(m).lower() in ("image", "vision") for m in modalities)
    return None


def _diagnose(status, body_snippet):
    """Turn a failed listing attempt into something the user can act on."""
    text = (body_snippet or "").lower()
    if status == 401:
        return "key rejected (401 unauthorized) — wrong or expired key"
    if status == 403:
        if "cloudflare" in text or "<!doctype html" in text or "attention required" in text:
            return "403 Cloudflare challenge (HTML, not JSON) — blocked before the API"
        return "403 forbidden — key lacks access, or the account is unfunded"
    if status == 404:
        return "404 — no model listing at this path"
    if status == 429:
        return "429 rate limited — try again shortly"
    if status == 0:
        return "could not connect (DNS/TLS/timeout)"
    if 200 <= status < 300:
        if "<!doctype html" in text or "<html" in text:
            return "%d but returned a web page, not JSON — wrong base url?" % status
        return "%d but no model list in the response" % status
    return "HTTP %s" % status


def detect_flavor(base_url, key, timeout=20):
    """Poke the URL. Returns (flavor, resolved_base, items, note).

    On failure the note names the most informative attempt rather than
    concatenating every raw status, so the UI can show a real reason.
    """
    attempts = []
    for base in _candidate_bases(base_url):
        url = base + "/models"

        r = H.get(url, H.openai_auth(key), timeout=timeout)
        items = _model_items(r.json()) if r.ok else []
        if r.ok and items:
            return "openai", base, items, "listed %d models" % len(items)
        attempts.append((base, "openai", r.status, r.text(300)))

        r2 = H.get(url, H.anthropic_auth(key), timeout=timeout)
        items2 = _model_items(r2.json()) if r2.ok else []
        if r2.ok and items2:
            return "anthropic", base, items2, "listed %d models (anthropic)" % len(items2)
        attempts.append((base, "anthropic", r2.status, r2.text(300)))

    # Report the attempt that carries the most signal: a real API error beats
    # a Cloudflare page, and anything beats a bare connection failure.
    def rank(a):
        status = a[2]
        if status in (401, 403, 429):
            return 0
        if status == 404:
            return 1
        if 200 <= status < 300:
            return 1
        return 2

    best = sorted(attempts, key=rank)[0] if attempts else None
    if best:
        base, shape, status, body = best
        note = "%s: %s" % (base + "/models", _diagnose(status, body))
    else:
        note = "no endpoints tried"
    return "unknown", base_url.rstrip("/"), [], note


# ---------------------------------------------------------------- inspection


def _chat_url(base, flavor):
    return base + ("/messages" if flavor == "anthropic" else "/chat/completions")


def _auth(key, flavor):
    return H.anthropic_auth(key) if flavor == "anthropic" else H.openai_auth(key)


def _error_text(r):
    """Best-effort human error string out of a failed response."""
    payload = r.json()
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if isinstance(err, str):
            return err
        if payload.get("message"):
            return str(payload["message"])
    return r.text(600)


def _error_in_200(payload):
    """Error message when an HTTP 200 response hides a failure in its body.

    Some providers (and misconfigured routers) answer 200 to any well-formed
    request but put an `error` object inside the payload instead of using a
    4xx status. Treating that as success was the availability false positive:
    a model that errors on every request looked "available" because the HTTP
    status was 2xx. Returns the message string, or None when the body is a
    real completion.
    """
    if not isinstance(payload, dict):
        return None
    err = payload.get("error")
    if isinstance(err, dict) and err.get("message"):
        return str(err["message"])[:120]
    if isinstance(err, str) and err.strip():
        return err.strip()[:120]
    return None


def _account_problem(r):
    """True when the failure is about billing/auth/limits, not the model."""
    if r.status in ACCOUNT_STATUSES:
        return True
    text = _error_text(r).lower()
    return any(w in text for w in ("balance", "quota", "insufficient", "recharge",
                                   "credit", "billing", "invalid api key", "unauthorized"))


def _prompt_tokens(payload, flavor):
    """Input-token count from either response shape, or None if absent."""
    usage = (payload or {}).get("usage") or {}
    for field in ("prompt_tokens", "input_tokens"):
        val = usage.get(field)
        if isinstance(val, int) and val > 0:
            return val
    return None


def _reasoning_evidence(payload, flavor):
    """Non-empty reasoning trace, or '' when the model produced none.

    Deliberately strict: the mere presence of a `reasoning` key proves
    nothing, so only non-blank text or a positive reasoning-token count
    counts as evidence. Three shapes are recognised — a dedicated field,
    a reasoning-token count, and an inline <think> block at the head of
    `content` (common on raw vLLM deployments with no reasoning parser).
    """
    if not isinstance(payload, dict):
        return ""
    if flavor == "anthropic":
        for block in payload.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "thinking":
                text = (block.get("thinking") or "").strip()
                if text:
                    return text
        return ""

    usage = payload.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    if isinstance(details, dict):
        try:
            if int(details.get("reasoning_tokens") or 0) > 0:
                return "usage:%s tokens" % details["reasoning_tokens"]
        except (TypeError, ValueError):
            pass
    for choice in payload.get("choices") or []:
        msg = choice.get("message") or {}
        for field in ("reasoning", "reasoning_content"):
            val = msg.get(field) or choice.get(field)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list):  # some gateways ship blocks
                joined = " ".join(str(b.get("text", "")) for b in val
                                  if isinstance(b, dict)).strip()
                if joined:
                    return joined
        content = msg.get("content")
        if isinstance(content, str):
            m = INLINE_THINK.match(content.lstrip())
            if m and (m.group(2) or "").strip():
                return m.group(2).strip()
    return ""


def _png(w, h, rgb=(200, 30, 30)):
    """Smallest valid RGB PNG of the requested size, stdlib only."""
    raw = b""
    row = b"\x00" + bytes(rgb) * w
    for _ in range(h):
        raw += row

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def hf_config_facts(model_id, timeout=20):
    """Vision support from the model's HuggingFace config.

    Only meaningful for `org/model` style ids (vLLM-backed providers serve
    them verbatim). Costs no provider tokens and needs no key. Returns
    {} when the id isn't on the Hub or the config says nothing useful.
    """
    if "/" not in model_id or model_id.count("/") != 1:
        return {}
    url = "https://huggingface.co/%s/resolve/main/config.json" % model_id
    r = H.get(url, timeout=timeout)
    if not r.ok:
        return {}
    cfg = r.json()
    if not isinstance(cfg, dict):
        return {}

    out = {}
    arch = " ".join(cfg.get("architectures") or [])
    has_vision = bool(cfg.get("vision_config") or cfg.get("image_token_index")
                      or cfg.get("vision_tower") or "VL" in arch
                      or "Vision" in arch or "ForConditionalGeneration" in arch)
    out["vision"] = has_vision
    out["arch"] = arch or None
    return out


def probe_availability(base, model_id, key, flavor, timeout=30, retries=1):
    """One minimal chat call: does this model answer at all?

    Cheap enough to always run first (~1 token): a dead or misrouted model
    fails here in one request instead of burning a reasoning probe and a
    vision differential. Returns (True/False/None, note).

    Only a definitive rejection counts as False: a 404/"no such model", or
    the provider being flat-out unreachable (connection refused, DNS). A
    timeout or 5xx is flakiness, not death — free tiers wobble constantly —
    so those are retried once and otherwise reported as None
    (inconclusive), which lets the capability probes run anyway.
    """
    url, auth = _chat_url(base, flavor), _auth(key, flavor)
    body = {"model": model_id, "max_tokens": 4, "temperature": 0,
            "messages": [{"role": "user", "content": "ping"}]}
    last = None
    for attempt in range(retries + 1):
        r = H.post(url, body, auth, timeout=timeout)
        if r.ok:
            payload = r.json() or {}
            if looks_faked(payload, flavor):
                return None, "provider returned a quota notice, not the model"
            msg = _error_in_200(payload)
            if msg:
                return False, "provider errored inside an HTTP 200: %s" % msg
            return True, "answered a minimal request"
        if _account_problem(r):
            return None, "account error (%s) — cannot tell" % (r.status or r.error)
        text = _error_text(r)
        if r.status == 404 or re.search(r"not found|no such model|does not exist", text, re.I):
            return False, "%s: %s" % (r.status, text[:70])
        if not r.status and "timed out" not in (r.error or "").lower():
            # refused / no route: the provider itself is down, and every
            # capability probe would fail the same way — skip them
            return False, "provider unreachable (%s)" % (r.error or "connection")[:70]
        last = "%s: %s" % (r.status or "timeout", text[:70])
    # slow answers after retries: flaky, not proven dead
    return None, "no answer after %d tries (%s) — inconclusive" % (retries + 1, last)


def probe_reasoning(base, model_id, key, flavor, timeout=90, fast=False):
    """Does this model emit a reasoning trace? Evidence-based.

    Asks with reasoning enabled and requires a non-empty trace in the reply.
    When one shows up, a second call without the flag separates "reasons
    only when asked" from "always reasons, can't be turned off" — the
    latter breaks JSON-mode consumers, so it's worth naming. `fast=True`
    skips that confirm call (one request instead of two).
    """
    url, auth = _chat_url(base, flavor), _auth(key, flavor)
    # A trivial greeting lets a thinking model skip its trace entirely, so the
    # prompt asks for something small that still needs a step or two.
    ask = "What is 17*23?"
    if flavor == "anthropic":
        on = {"model": model_id, "max_tokens": 1024,
              "thinking": {"type": "enabled", "budget_tokens": 1024},
              "messages": [{"role": "user", "content": ask}]}
    else:
        on = {"model": model_id, "max_tokens": 160, "reasoning_effort": "low",
              "messages": [{"role": "user", "content": ask}]}

    r = H.post(url, on, auth, timeout=timeout)
    if not r.ok:
        if _account_problem(r):
            return None, "account error (%s)" % (r.status or r.error)
        text = _error_text(r)
        if re.search(r"reasoning|thinking|budget_tokens|effort", text, re.I):
            # provider rejected the field: retry plain to see if it reasons anyway
            plain = {"model": model_id, "max_tokens": 160,
                     "messages": [{"role": "user", "content": ask}]}
            r2 = H.post(url, plain, auth, timeout=timeout)
            if r2.ok and not looks_faked(r2.json() or {}, flavor):
                if _reasoning_evidence(r2.json() or {}, flavor):
                    return True, "reasons always (rejects the effort flag)"
                return False, "rejected the reasoning flag, no trace without it"
            return None, "rejected the flag; plain call failed (%s)" % (r2.status or r2.error)
        return None, "error %s: %s" % (r.status or r.error, text[:80])

    payload = r.json() or {}
    if looks_faked(payload, flavor):
        return None, "provider returned a quota notice, not the model"
    msg = _error_in_200(payload)
    if msg:
        return None, "provider errored inside an HTTP 200: %s" % msg

    trace = _reasoning_evidence(payload, flavor)
    if not trace:
        return False, "no reasoning trace even with the flag set"
    if fast:
        return True, "reasons (fast: %d chars of trace)" % len(trace)

    # confirm whether it can be switched off
    plain = {"model": model_id, "max_tokens": 160,
             "messages": [{"role": "user", "content": ask}]}
    r2 = H.post(url, plain, auth, timeout=timeout)
    if r2.ok and not looks_faked(r2.json() or {}, flavor):
        if _reasoning_evidence(r2.json() or {}, flavor):
            return True, "always reasons (trace present without the flag too)"
        return True, "reasons on request (%d chars of trace)" % len(trace)
    return True, "returned a %d-char reasoning trace" % len(trace)


def probe_vision(base, model_id, key, flavor, timeout=120, fast=False):
    """Does the model actually ingest images? Token-differential test.

    A no-error response proves nothing: several vLLM gateways accept an
    image part, silently drop it, and answer the text. So the same prompt
    is sent with and without a 512x512 image and the input-token counts
    are compared — real image ingestion cannot cost zero tokens.

    The image must be big enough that real ingestion clears the token
    floor: some providers (b.ai, Qwen hosted) charge only ~60 tokens for
    a 256x256 image, which fell under the old floor and read as a silent
    drop. 512x512 costs ~250 tokens there — unambiguous.

    The two calls run concurrently, so the wall clock is roughly one
    request even though two are made. `fast` is accepted for call-site
    compatibility but does NOT shortcut the differential: there is no
    single-call way to distinguish real vision from a silent drop, and
    claiming "yes" from a bare 200 was a false positive on real providers.
    """
    url, auth = _chat_url(base, flavor), _auth(key, flavor)
    prompt = "Describe what you see."
    img = _png(512, 512)

    def send(parts):
        body = {"model": model_id, "max_tokens": 16,
                "messages": [{"role": "user", "content": parts}]}
        return H.post(url, body, auth, timeout=timeout)

    if flavor == "anthropic":
        text_part = {"type": "text", "text": prompt}
        img_part = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                               "data": base64.b64encode(img).decode()}}
    else:
        text_part = {"type": "text", "text": prompt}
        img_part = {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(img).decode()}}

    # The image and text-baseline calls are independent and both slow; run
    # them concurrently so a slow free tier isn't charged twice sequentially.
    # (This is also the ONLY honest vision test — see the docstring.)
    results = {}
    def call(which, parts):
        results[which] = send(parts)
    t_img = threading.Thread(target=call, args=("img", [text_part, img_part]))
    t_txt = threading.Thread(target=call, args=("txt", [text_part]))
    t_img.start()
    t_txt.start()
    t_img.join()
    t_txt.join()
    r_img = results["img"]
    r_txt = results["txt"]

    if not r_img.ok:
        if _account_problem(r_img):
            return None, "account error (%s)" % (r_img.status or r_img.error)
        text = _error_text(r_img)
        if VISION_REFUSAL.search(text):
            return False, "refused image input: %s" % text[:70]
        return None, "error %s: %s" % (r_img.status or r_img.error, text[:70])

    img_payload = r_img.json() or {}
    if looks_faked(img_payload, flavor):
        return None, "provider returned a quota notice, not the model"
    msg = _error_in_200(img_payload)
    if msg:
        return None, "provider errored inside an HTTP 200: %s" % msg

    if not r_txt.ok:
        return None, "image call succeeded but the text baseline failed (%s)" % (
            r_txt.status or r_txt.error)

    img_pt = _prompt_tokens(img_payload, flavor)
    txt_pt = _prompt_tokens(r_txt.json() or {}, flavor)
    if img_pt is None or txt_pt is None:
        return None, "no usage counts returned — cannot prove the image was read"
    delta = img_pt - txt_pt
    if delta > VISION_TOKEN_FLOOR:
        return True, "image cost %d extra input tokens" % delta
    if delta > 0:
        # some routers count the base64 payload without the model seeing it
        return False, ("image silently dropped (%d tokens either way, "
                       "%d base64 overhead)" % (txt_pt, delta))
    return False, "image silently dropped (%d tokens either way)" % txt_pt


def merge_inspection(meta, res):
    """Fold an inspection result into a saved model entry.

    Only real verdicts are applied. A probe the user declined (or that came
    back inconclusive) carries `None` — overwriting with it would wipe a
    previously measured yes/no, which is exactly the blank-spot bug: probing
    vision, saying no to the reasoning prompt, and losing the vision result.
    """
    entry = dict(meta or {})
    for which in ("reasoning", "vision"):
        if res.get(which) is not None:
            entry[which] = res[which]
            entry["%s_note" % which] = res["%s_note" % which]
    if res.get("available") is not None:
        entry["available"] = res["available"]
        entry["available_note"] = res["available_note"]
    return entry


def inspect_model(base, model_id, key, flavor, listing_item=None,
                  progress=None, ask=None):
    """Probe ONE model for availability, reasoning, and vision.

    On-demand only, never part of a bulk pass. Flow:

      1. availability: ONE minimal live call, always. If the model does not
         answer, capability probing is skipped entirely — no point burning a
         reasoning probe and a vision differential on a dead route.
      2. reasoning/vision: stated listing flags are trusted without asking;
         unstated ones are offered to `ask` (callable question -> bool).
         With no `ask` callback, both are probed (non-interactive default).

    `listing_item` should be the provider's own /models entry when available:
    some providers publish capability flags there, which saves both requests
    and guesses.

    Context size is deliberately NOT measured here: providers leak it
    inconsistently (or not at all), and the client that actually consumes a
    model is in a better position to probe it at runtime. CG reports only
    what it can prove — availability, reasoning, vision.
    """
    def step(label):
        if progress:
            progress(label)

    def want(question):
        return True if ask is None else bool(ask(question))

    out = {"model": model_id, "flavor": flavor}

    step("availability")
    out["available"], out["available_note"] = probe_availability(base, model_id, key, flavor)

    stated = {}
    for which in ("reasoning", "vision"):
        val = capability_from_fields(listing_item or {}, which)
        if val is not None:
            stated[which] = (bool(val), "stated by the provider listing")

    if out["available"] is False:
        # dead route: keep any free facts, skip every paid probe
        for which in ("reasoning", "vision"):
            if which in stated:
                out[which], out["%s_note" % which] = stated[which]
            else:
                out[which], out["%s_note" % which] = None, "skipped: model unavailable"
        return out

    for which in ("reasoning", "vision"):
        if which in stated:
            out[which], out["%s_note" % which] = stated[which]
            continue
        if not want("probe %s for %s?" % (which, model_id)):
            out[which], out["%s_note" % which] = None, "skipped: declined"
            continue
        step(which)
        try:
            if which == "reasoning":
                out[which], out["%s_note" % which] = probe_reasoning(
                    base, model_id, key, flavor, fast=True)
            else:
                # vision has NO fast path: a bare 200 can't prove ingestion
                out[which], out["%s_note" % which] = probe_vision(
                    base, model_id, key, flavor)
        except Exception as e:  # noqa: BLE001 — a probe must never kill inspect
            out[which], out["%s_note" % which] = None, "probe crashed: %s" % e

    # Fall back to stated facts only where measurement was inconclusive.
    for which in ("reasoning", "vision"):
        if out[which] is not None:
            continue
        stated = capability_from_fields(listing_item or {}, which)
        if stated is not None:
            out[which] = stated
            out["%s_note" % which] = "%s (falling back to the provider listing)" % \
                out["%s_note" % which]
    if out["vision"] is None:
        hf = hf_config_facts(model_id)
        if "vision" in hf:
            out["vision"] = hf["vision"]
            out["vision_note"] = "%s (from HF config: %s)" % (
                out["vision_note"], hf.get("arch") or "no vision_config")
    return out


def discover(base_url, key, flavor=None, progress=None):
    """Cheap listing pass: flavor + model ids, no per-model requests.

    Context is NOT resolved here — the client (Hermes) probes it at runtime
    when the model is first used. Per-model facts (reasoning, vision) come
    from `inspect_model`, run on demand for the one model you care about.
    """
    if flavor in ("openai", "anthropic"):
        base, items, note = None, [], ""
        for cand in _candidate_bases(base_url):
            auth = H.anthropic_auth(key) if flavor == "anthropic" else H.openai_auth(key)
            r = H.get(cand + "/models", auth)
            got = _model_items(r.json()) if r.ok else []
            if got:
                base, items, note = cand, got, "listed %d models" % len(got)
                break
        if base is None:
            base, note = base_url.rstrip("/"), "manual flavor, model list unavailable"
    else:
        flavor, base, items, note = detect_flavor(base_url, key)

    models = {}
    for i, item in enumerate(items):
        if progress:
            progress(i + 1, len(items), item["id"])
        entry = {}
        # A few providers publish capability flags in the listing. Record them
        # as stated (not measured) so the UI can show something useful before
        # anyone spends a token, and mark where they came from.
        for which in ("reasoning", "vision"):
            val = capability_from_fields(item, which)
            if val is not None:
                entry[which] = val
                entry["%s_note" % which] = "stated by the provider listing"
        # Context is read from the listing when the provider volunteers it.
        # No probing, no default: if it's absent the user sets it manually
        # or the client discovers it at runtime.
        ctx = context_from_fields(item)
        if ctx is not None:
            entry["context"] = ctx
            entry["context_source"] = "listing"
        models[item["id"]] = entry

    return {"flavor": flavor, "base_url": base, "models": models, "note": note}
