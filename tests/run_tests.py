#!/usr/bin/env python3
"""CG test suite. Runs against a local fake upstream — no real API calls,
no credits burned, no network access.

    python3 tests/run_tests.py
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from cgw import config as C  # noqa: E402
from cgw import detect as D  # noqa: E402
from cgw import http as H  # noqa: E402
from cgw import keyring as K  # noqa: E402
from cgw import server as SV  # noqa: E402
from cgw import translate as T  # noqa: E402
from cgw import usage as U  # noqa: E402
from cgw.server import serve  # noqa: E402

import fake_upstream  # noqa: E402
from fake_upstream import Fake  # noqa: E402

UP = None
UPBASE = ""


def setUpModule():
    global UP, UPBASE
    UP = fake_upstream.start()
    threading.Thread(target=UP.serve_forever, daemon=True).start()
    UPBASE = "http://127.0.0.1:%d" % UP.server_address[1]


def tearDownModule():
    UP.shutdown()
    UP.server_close()


class TestConfig(unittest.TestCase):
    @unittest.skipUnless(os.name != "nt", "POSIX permissions: no-op on Windows")
    def test_roundtrip_and_permissions(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg = C.default_config()
            cfg["providers"].append(C.new_provider("p1", "http://x/v1/", ["k1", "k2"]))
            C.save(cfg, path)
            self.assertEqual(oct(os.stat(path).st_mode)[-3:], "600")
            back = C.load(path)
            self.assertEqual(back["providers"][0]["name"], "p1")
            self.assertEqual(back["providers"][0]["base_url"], "http://x/v1")
            self.assertEqual([k["label"] for k in back["providers"][0]["keys"]], ["k1", "k2"])

    def test_model_enabled_flag_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cfg = C.default_config()
            p = C.new_provider("p1", "http://x/v1/", ["k"])
            p["models"] = {"on-model": {"reasoning": True},
                           "off-model": {"enabled": False}}
            cfg["providers"].append(p)
            C.save(cfg, path)
            back = C.load(path)
            m = back["providers"][0]["models"]
            self.assertTrue(m["on-model"].get("enabled", True))
            self.assertFalse(m["off-model"]["enabled"])

    def test_disabled_model_survives_relisting(self):
        """A model toggled off stays off through a listing merge; an on one
        stays on; a toggled-off model dropped from the listing is kept."""
        old = {"m1": {"enabled": False}, "m2": {"enabled": True}}
        # listing no longer includes m1 (dropped upstream) and defaults m2
        merged = C.merge_models(old, {"m2": {"source": "listing"}})
        self.assertIn("m1", merged, "toggled-off model must not vanish")
        self.assertFalse(merged["m1"]["enabled"])
        self.assertTrue(merged["m2"]["enabled"], "enabled stays enabled")
        self.assertFalse(merged["m2"]["source"] == "manual")

    def test_corrupt_file_raises_instead_of_blanking(self):
        """A corrupt existing file must not read back as an empty config.

        Callers save the whole config after loading it, so a silent empty
        fallback overwrote every real provider and key.
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            with open(path, "w") as fh:
                fh.write("{not json")
            with self.assertRaises(C.ConfigError):
                C.load(path)
            # explicit opt-in still allows the lossy fallback
            self.assertEqual(C.load(path, strict=False)["providers"], [])

    def test_missing_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = C.load(os.path.join(d, "nope.json"))
            self.assertEqual(cfg["providers"], [])

    def test_bad_int_field_keeps_providers(self):
        """One hand-edited typo used to abort the load and blank everything."""
        raw = {
            "version": 1,
            "listen": {"host": "127.0.0.1", "port": "not-a-port"},
            "providers": [{
                "name": "R", "base_url": "https://x.com/v1", "keys": ["sk-real"],
                "models": {"m": {}},
            }],
        }
        cfg = C.normalize(raw)
        self.assertEqual(len(cfg["providers"]), 1)
        p = cfg["providers"][0]
        self.assertEqual(p["keys"][0]["key"], "sk-real")
        self.assertEqual(p["models"]["m"], {})
        self.assertEqual(cfg["listen"]["port"], C.DEFAULT_PORT)

    def test_save_accepts_relative_path(self):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                C.save(C.default_config(), "config.json")
                self.assertTrue(os.path.exists(os.path.join(d, "config.json")))
            finally:
                os.chdir(cwd)

    def test_merge_models_keeps_earned_facts(self):
        old = {
            "a": {"reasoning": True, "reasoning_note": "r"},
            "b": {},
            "c": {"vision": False, "vision_note": "v", "source": "manual"},
            "d": {"vision": True, "vision_note": "stated by the provider listing"},
        }
        new = {
            "a": {},
            "b": {},
            "d": {},
        }
        out = C.merge_models(old, new)
        self.assertIs(out["a"]["reasoning"], True)      # measured fact kept
        self.assertIsNone(out["b"].get("reasoning"))    # guess not preserved
        self.assertIs(out["c"]["vision"], False)        # manual model kept
        self.assertIs(out["d"]["vision"], True)         # capability facts kept

    def test_normalize_fills_missing(self):
        cfg = C.normalize({"providers": [{"name": "x", "base_url": "http://y", "keys": ["a"]}]})
        p = cfg["providers"][0]
        self.assertEqual(p["rotation"], "fill_first")
        self.assertEqual(p["flavor"], "unknown")

    def test_normalize_keeps_manual_source(self):
        """'source: manual' must survive a load/save round-trip: merge_models
        uses it to keep hand-added models when a re-listing omits them, so
        dropping it on normalize silently deleted those models."""
        cfg = C.normalize({"providers": [{
            "name": "x", "base_url": "http://y", "keys": ["a"],
            "models": {"hand": {"source": "manual"},
                       "hand-off": {"source": "manual", "enabled": False}},
        }]})
        m = cfg["providers"][0]["models"]
        self.assertEqual(m["hand"].get("source"), "manual")
        self.assertEqual(m["hand-off"].get("source"), "manual")
        # and a full save/load round-trip keeps it too
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            C.save(cfg, path)
            self.assertEqual(
                C.load(path)["providers"][0]["models"]["hand"].get("source"),
                "manual")

    def test_as_int_none_default_returns_none_not_raises(self):
        """The TUI context prompt feeds '1M' -> as_int('1m', None); it must
        return None (so the k/m suffix parser takes over), not raise. This
        was the bug that made typing 1M in the TUI drop out of the prompt."""
        self.assertIsNone(C.as_int("1m", None))
        self.assertIsNone(C.as_int("128k", None))
        self.assertIsNone(C.as_int("1M", None))
        self.assertIsNone(C.as_int("", None))
        self.assertIsNone(C.as_int(None, None))
        self.assertEqual(C.as_int("128000", None), 128000)
        self.assertEqual(C.as_int("128000", 0), 128000)
        self.assertEqual(C.as_int("bogus", 0), 0)
        self.assertEqual(C.as_int(None, 0), 0)

    def test_context_suffixes_parse_both_cases(self):
        """'1M'/'1m'/'128K'/'128k' all resolve to the same token counts,
        through the exact parse the TUI uses for the manual context prompt."""
        def tui_parse(val):
            val = val.strip().lower().replace("_", "")
            if not val:
                return None
            n = C.as_int(val, None)
            if n is None and val.endswith("k"):
                try:
                    n = int(float(val[:-1]) * 1000)
                except ValueError:
                    n = None
            if n is None and val.endswith("m"):
                try:
                    n = int(float(val[:-1]) * 1_000_000)
                except ValueError:
                    n = None
            return n if n and n > 0 else None
        self.assertEqual(tui_parse("1M"), 1_000_000)
        self.assertEqual(tui_parse("1m"), 1_000_000)
        self.assertEqual(tui_parse("128K"), 128_000)
        self.assertEqual(tui_parse("128k"), 128_000)
        self.assertEqual(tui_parse("1.5M"), 1_500_000)
        self.assertIsNone(tui_parse("bogus"))
        self.assertIsNone(tui_parse(""))


class TestKeyRing(unittest.TestCase):
    def ring(self, n=3, rotation="fill_first"):
        return K.KeyRing([{"key": "k%d" % i, "label": "k%d" % i} for i in range(n)], rotation)

    def test_fill_first_prefers_first_key(self):
        r = self.ring()
        self.assertEqual(r.try_order()[0], 0)
        self.assertEqual(r.try_order()[0], 0)  # stable, not rotating

    def test_round_robin_advances(self):
        r = self.ring(rotation="round_robin")
        firsts = [r.try_order()[0] for _ in range(4)]
        self.assertGreater(len(set(firsts)), 1)

    def test_429_benches_key_then_recovers(self):
        r = self.ring()
        r.report_failure(0, 429)
        self.assertEqual(r.try_order()[0], 1)
        self.assertEqual(r.state[0].status_text()[:4], "cool")
        r.state[0].cooldown_until = time.time() - 1  # simulate elapsed cooldown
        self.assertEqual(r.try_order()[0], 0)

    def test_401_kills_key_permanently(self):
        r = self.ring()
        r.report_failure(0, 401)
        self.assertTrue(r.state[0].dead)
        self.assertEqual(r.state[0].status_text(), "dead")
        for _ in range(5):
            self.assertNotIn(0, r.try_order())
        r.revive_all()
        self.assertIn(0, r.try_order())

    def test_backoff_grows(self):
        r = self.ring()
        r.report_failure(0, 429)
        first = r.state[0].cooldown_until - time.time()
        r.report_failure(0, 429)
        second = r.state[0].cooldown_until - time.time()
        self.assertGreater(second, first)

    def test_success_resets_failures(self):
        r = self.ring()
        r.report_failure(0, 429)
        r.report_success(0)
        self.assertEqual(r.state[0].failures, 0)
        self.assertEqual(r.try_order()[0], 0)

    def test_all_cooling_still_returns_one(self):
        r = self.ring(2)
        r.report_failure(0, 429)
        r.report_failure(1, 429)
        self.assertEqual(len(r.try_order()), 1)

    def test_disabled_key_skipped(self):
        r = K.KeyRing([{"key": "a", "label": "k1", "enabled": False}, {"key": "b", "label": "k2"}])
        self.assertEqual(r.try_order(), [1])

    def test_registry_rebuilds_on_key_change(self):
        reg = K.Registry()
        p = C.new_provider("p", "http://x", ["a"])
        r1 = reg.get(p)
        r2 = reg.get(p)
        self.assertIs(r1, r2)
        p["keys"].append({"key": "b", "label": "k2", "enabled": True})
        self.assertIsNot(reg.get(p), r1)


    def test_transport_failure_gets_short_cooldown(self):
        """A timeout or client hangup says nothing about the key."""
        r = self.ring()
        r.report_failure(0, 0, now=1000.0)
        self.assertLessEqual(r.state[0].cooldown_until - 1000.0, K.TRANSPORT_COOLDOWN)
        self.assertFalse(r.state[0].dead)

    def test_http_failure_uses_exponential_cooldown(self):
        r = self.ring()
        r.report_failure(0, 429, now=1000.0)
        self.assertEqual(r.state[0].cooldown_until - 1000.0, K.COOLDOWN_BASE)

    def test_single_key_revive(self):
        r = self.ring()
        r.report_failure(0, 403)
        self.assertTrue(r.state[0].dead)
        r.revive(0)
        self.assertFalse(r.state[0].dead)
        self.assertEqual(r.state[0].cooldown_until, 0.0)

    def test_round_robin_cursor_stable_while_keys_cool(self):
        """The cursor counts dispatches; a shrinking healthy list skewed it."""
        r = self.ring(3, rotation="round_robin")
        first = [r.try_order(now=1000.0)[0] for _ in range(3)]
        self.assertEqual(sorted(first), [0, 1, 2])
        r.report_failure(1, 429, now=1000.0)
        seen = [r.try_order(now=1000.0)[0] for _ in range(4)]
        self.assertNotIn(1, seen)
        self.assertEqual(set(seen), {0, 2})

    def test_registry_revive_by_name(self):
        reg = K.Registry()
        p = C.new_provider("x", "https://a.com/v1", ["k1"])
        ring = reg.get(p)
        ring.report_failure(0, 401)
        self.assertTrue(ring.state[0].dead)
        self.assertEqual(reg.revive("nope"), 0)
        self.assertTrue(ring.state[0].dead)
        self.assertEqual(reg.revive("x"), 1)
        self.assertFalse(ring.state[0].dead)


class TestDetect(unittest.TestCase):
    def test_rich_provider_reads_capabilities_from_listing(self):
        res = D.discover(UPBASE + "/rich/v1", "anykey")
        self.assertEqual(res["flavor"], "openai")
        # rich provider advertises context_window / context_length
        self.assertEqual(res["models"]["rich-a"]["context"], 512000)
        self.assertEqual(res["models"]["rich-a"]["context_source"], "listing")
        self.assertEqual(res["models"]["rich-b"]["context"], 262144)

    def test_bare_provider_listing(self):
        res = D.discover(UPBASE + "/bare/v1", "anykey")
        self.assertEqual(res["models"]["bare-a"], {})

    def test_anthropic_flavor_detected(self):
        res = D.discover(UPBASE + "/anthropic/v1", "anykey")
        self.assertEqual(res["flavor"], "anthropic")
        self.assertIn("claude-fake", res["models"])
        # anthropic provider lists max_context_length
        self.assertEqual(res["models"]["claude-fake"]["context"], 200000)
        self.assertEqual(res["models"]["claude-fake"]["context_source"], "listing")

    def test_unknown_provider_reports_unknown(self):
        res = D.discover(UPBASE + "/nope/v1", "anykey")
        self.assertEqual(res["flavor"], "unknown")
        self.assertEqual(res["models"], {})

    def test_base_url_without_v1_is_recovered(self):
        # user pastes the host root; detection should find /rich/v1 style paths
        res = D.discover(UPBASE + "/rich", "anykey")
        self.assertEqual(res["flavor"], "openai")
        self.assertEqual(res["base_url"], UPBASE + "/rich/v1")

    def test_model_items_handles_shapes(self):
        self.assertEqual([m["id"] for m in D._model_items(["a", "b"])], ["a", "b"])
        self.assertEqual([m["id"] for m in D._model_items({"data": [{"id": "x"}]})], ["x"])
        self.assertEqual([m["id"] for m in D._model_items({"models": {"y": {"context": 1}}})], ["y"])


class TestReasoningEvidence(unittest.TestCase):
    """A 'yes' must rest on an actual trace, never on a key being present."""

    def _msg(self, **kw):
        return {"choices": [{"message": dict({"role": "assistant"}, **kw)}]}

    def test_empty_reasoning_field_is_not_evidence(self):
        self.assertEqual(D._reasoning_evidence(self._msg(reasoning="", content="hi"), "openai"), "")
        self.assertEqual(D._reasoning_evidence(self._msg(reasoning="   ", content="hi"), "openai"), "")
        self.assertEqual(D._reasoning_evidence(self._msg(reasoning=None, content="hi"), "openai"), "")

    def test_populated_reasoning_field_is_evidence(self):
        got = D._reasoning_evidence(self._msg(reasoning="We need to add.", content="4"), "openai")
        self.assertEqual(got, "We need to add.")

    def test_inline_think_block_is_evidence(self):
        payload = self._msg(content="<think>17*23 = 391</think>The answer is 391.")
        self.assertIn("391", D._reasoning_evidence(payload, "openai"))

    def test_plain_worked_answer_is_not_evidence(self):
        # Kimi-style: shows its work in normal prose, no trace channel
        payload = self._msg(content="17 x 20 = 340, plus 51 = 391.")
        self.assertEqual(D._reasoning_evidence(payload, "openai"), "")

    def test_reasoning_token_count_is_evidence(self):
        payload = {"choices": [{"message": {"content": "4"}}],
                   "usage": {"completion_tokens_details": {"reasoning_tokens": 12}}}
        self.assertIn("12", D._reasoning_evidence(payload, "openai"))

    def test_zero_reasoning_tokens_is_not_evidence(self):
        payload = {"choices": [{"message": {"content": "4"}}],
                   "usage": {"completion_tokens_details": {"reasoning_tokens": 0}}}
        self.assertEqual(D._reasoning_evidence(payload, "openai"), "")

    def test_anthropic_thinking_block(self):
        payload = {"content": [{"type": "thinking", "thinking": "counting"},
                               {"type": "text", "text": "391"}]}
        self.assertEqual(D._reasoning_evidence(payload, "anthropic"), "counting")

    def test_anthropic_empty_thinking_block(self):
        payload = {"content": [{"type": "thinking", "thinking": ""}]}
        self.assertEqual(D._reasoning_evidence(payload, "anthropic"), "")


class TestVisionHelpers(unittest.TestCase):
    def test_png_is_valid_and_scales(self):
        small, big = D._png(1, 1), D._png(256, 256)
        self.assertTrue(small.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(big.endswith(b"IEND\xae\x42\x60\x82"))
        self.assertGreater(len(big), len(small))

    def test_prompt_tokens_reads_both_shapes(self):
        self.assertEqual(D._prompt_tokens({"usage": {"prompt_tokens": 88}}, "openai"), 88)
        self.assertEqual(D._prompt_tokens({"usage": {"input_tokens": 40}}, "anthropic"), 40)
        self.assertIsNone(D._prompt_tokens({"usage": {}}, "openai"))
        self.assertIsNone(D._prompt_tokens({}, "openai"))


class TestPasteHygiene(unittest.TestCase):
    """Silent paste accidents are the top cause of 'valid key rejected'."""

    KEY = "sk-nry-gv7Qfs13o4PVDyCra8SRo7l_JcQtlH9W6Ucl0Od9xmk"

    def test_plain_key_untouched(self):
        self.assertEqual(C.clean_key(self.KEY), self.KEY)

    def test_whitespace_and_newlines(self):
        self.assertEqual(C.clean_key("  %s\n" % self.KEY), self.KEY)
        self.assertEqual(C.clean_key("\t%s \r\n" % self.KEY), self.KEY)

    def test_quotes_stripped(self):
        self.assertEqual(C.clean_key('"%s"' % self.KEY), self.KEY)
        self.assertEqual(C.clean_key("'%s'" % self.KEY), self.KEY)

    def test_bearer_prefix_stripped(self):
        self.assertEqual(C.clean_key("Bearer %s" % self.KEY), self.KEY)
        self.assertEqual(C.clean_key("Authorization: Bearer %s" % self.KEY), self.KEY)
        self.assertEqual(C.clean_key("x-api-key: %s" % self.KEY), self.KEY)

    def test_json_trailing_comma(self):
        self.assertEqual(C.clean_key('"%s",' % self.KEY), self.KEY)

    def test_zero_width_characters_removed(self):
        self.assertEqual(C.clean_key("\ufeff%s\u200b" % self.KEY), self.KEY)

    def test_non_string_safe(self):
        self.assertEqual(C.clean_key(None), "")

    def test_url_endpoint_reduced_to_base(self):
        self.assertEqual(C.clean_url("https://api.x.com/v1/chat/completions"),
                         "https://api.x.com/v1")
        self.assertEqual(C.clean_url("https://api.x.com/v1/messages"),
                         "https://api.x.com/v1")
        self.assertEqual(C.clean_url("https://api.x.com/v1/models"),
                         "https://api.x.com/v1")

    def test_url_trailing_slash_and_quotes(self):
        self.assertEqual(C.clean_url(' "https://api.x.com/v1/" '), "https://api.x.com/v1")

    def test_url_scheme_added(self):
        self.assertEqual(C.clean_url("api.x.com/v1"), "https://api.x.com/v1")

    def test_new_provider_applies_hygiene(self):
        p = C.new_provider("p", "api.x.com/v1/chat/completions/",
                           ["Bearer %s\n" % self.KEY, "  ", '"k2",'])
        self.assertEqual(p["base_url"], "https://api.x.com/v1")
        self.assertEqual([k["key"] for k in p["keys"]], [self.KEY, "k2"])


class TestMergeInspection(unittest.TestCase):
    """A measurement is always applied; nothing about context is merged."""

    def _res(self, reasoning=True, vision=False):
        return {"reasoning": reasoning, "vision": vision,
                "reasoning_note": "r", "vision_note": "v"}

    def test_capabilities_always_updated(self):
        meta = {"vision": True, "vision_note": "stated by the provider listing"}
        out = D.merge_inspection(meta, self._res())
        self.assertIs(out["vision"], False)
        self.assertEqual(out["vision_note"], "v")

    def test_empty_meta(self):
        out = D.merge_inspection(None, self._res())
        self.assertIs(out["reasoning"], True)
        self.assertIs(out["vision"], False)

    def test_meta_survives(self):
        meta = {"reasoning_note": "old", "available": True}
        out = D.merge_inspection(meta, self._res())
        self.assertEqual(out["available"], True)
        self.assertEqual(out["reasoning_note"], "r")

    def test_declined_probe_keeps_previous_result(self):
        # the reported bug: probing vision (yes), then declining the
        # reasoning prompt wiped the vision verdict to a blank spot
        meta = {"reasoning": True, "reasoning_note": "measured earlier",
                "vision": True, "vision_note": "measured earlier"}
        res = {"reasoning": None, "reasoning_note": "skipped: declined",
               "vision": True, "vision_note": "image cost 245 extra input tokens"}
        out = D.merge_inspection(meta, res)
        self.assertIs(out["reasoning"], True)
        self.assertEqual(out["reasoning_note"], "measured earlier")
        self.assertIs(out["vision"], True)
        self.assertEqual(out["vision_note"], "image cost 245 extra input tokens")

    def test_declined_vision_keeps_previous_reasoning(self):
        meta = {"reasoning": False, "reasoning_note": "no trace",
                "vision": True, "vision_note": "yes earlier"}
        res = {"reasoning": True, "reasoning_note": "trace found",
               "vision": None, "vision_note": "skipped: declined"}
        out = D.merge_inspection(meta, res)
        # reasoning updated to the fresh real verdict...
        self.assertIs(out["reasoning"], True)
        self.assertEqual(out["reasoning_note"], "trace found")
        # ...but the declined vision keeps its previous value
        self.assertIs(out["vision"], True)
        self.assertEqual(out["vision_note"], "yes earlier")

    def test_inconclusive_keeps_previous_availability(self):
        meta = {"available": True, "available_note": "answered earlier"}
        res = {"reasoning": None, "reasoning_note": "skipped: model unavailable",
               "vision": None, "vision_note": "skipped: model unavailable",
               "available": None, "available_note": "account error (429) — cannot tell"}
        out = D.merge_inspection(meta, res)
        self.assertIs(out["available"], True)
        self.assertEqual(out["available_note"], "answered earlier")

    def test_new_availability_overrides_old(self):
        meta = {"available": True, "available_note": "answered earlier"}
        res = {"reasoning": None, "reasoning_note": "skipped: model unavailable",
               "vision": None, "vision_note": "skipped: model unavailable",
               "available": False, "available_note": "404: no such model"}
        out = D.merge_inspection(meta, res)
        self.assertIs(out["available"], False)
        self.assertEqual(out["available_note"], "404: no such model")


class TestStatedFacts(unittest.TestCase):
    """Saved listing facts are reusable; guesses are not."""

    def test_defaulted_stated_facts_not_reused(self):
        meta = {}
        self.assertEqual(D.stated_facts(meta), {})

    def test_listing_capability_flags_reused(self):
        meta = {"context": 1, "source": "default", "vision": True,
                "vision_note": "stated by the provider listing"}
        self.assertEqual(D.stated_facts(meta), {"vision": True})

    def test_measured_capability_not_fed_back(self):
        # a measured verdict must not be recycled as an input to the next probe
        meta = {"vision": False, "vision_note": "image silently dropped (88 tokens either way)"}
        self.assertEqual(D.stated_facts(meta), {})

    def test_none_safe(self):
        self.assertEqual(D.stated_facts(None), {})


class TestCapabilityFields(unittest.TestCase):
    """Listing-advertised capability flags (byNara publishes these)."""

    def test_bool_flags_read(self):
        item = {"id": "m", "vision": True, "reasoning": True}
        self.assertIs(D.capability_from_fields(item, "vision"), True)
        self.assertIs(D.capability_from_fields(item, "reasoning"), True)

    def test_empty_modalities_is_unknown_not_text_only(self):
        self.assertIsNone(D.capability_from_fields({"modalities": []}, "vision"))
        self.assertIs(D.capability_from_fields({"modalities": ["text"]}, "vision"), False)
        self.assertIs(D.capability_from_fields({"modalities": ["text", "image"]}, "vision"), True)

    def test_false_is_preserved_not_treated_as_missing(self):
        item = {"id": "m", "vision": False, "reasoning": False}
        self.assertIs(D.capability_from_fields(item, "vision"), False)
        self.assertIs(D.capability_from_fields(item, "reasoning"), False)

    def test_absent_flag_is_none_not_false(self):
        self.assertIsNone(D.capability_from_fields({"id": "m"}, "vision"))
        self.assertIsNone(D.capability_from_fields({"id": "m"}, "reasoning"))

    def test_non_bool_ignored(self):
        # "true"/1 are not trustworthy booleans in a JSON listing
        self.assertIsNone(D.capability_from_fields({"vision": "yes"}, "vision"))
        self.assertIsNone(D.capability_from_fields({"vision": 1}, "vision"))

    def test_modalities_list(self):
        item = {"input_modalities": ["text", "image"]}
        self.assertIs(D.capability_from_fields(item, "vision"), True)
        self.assertIs(D.capability_from_fields({"input_modalities": ["text"]}, "vision"), False)


class TestDiagnose(unittest.TestCase):
    """Failure notes must name a cause, not echo a status code."""

    def test_401_names_the_key(self):
        self.assertIn("key rejected", D._diagnose(401, "{}"))

    def test_cloudflare_403_distinguished_from_plain_403(self):
        cf = D._diagnose(403, "<!DOCTYPE html> cloudflare attention required")
        plain = D._diagnose(403, '{"error":"no access"}')
        self.assertIn("Cloudflare", cf)
        self.assertNotIn("Cloudflare", plain)

    def test_html_200_flagged_as_wrong_url(self):
        self.assertIn("web page", D._diagnose(200, "<!DOCTYPE html><html>"))

    def test_connection_failure(self):
        self.assertIn("could not connect", D._diagnose(0, ""))


class TestCapabilityProbes(unittest.TestCase):
    """End-to-end capability probes against upstreams built to mislead."""

    def test_inline_think_model_reads_as_reasoning(self):
        state, note = D.probe_reasoning(UPBASE + "/thinker/v1", "bare-a", "k", "openai")
        self.assertIs(state, True)
        self.assertIn("always reasons", note)

    def test_blank_reasoning_key_is_not_a_yes(self):
        # the exact false positive that shipped: key present, value empty
        state, note = D.probe_reasoning(UPBASE + "/emptyreason/v1", "bare-a", "k", "openai")
        self.assertIs(state, False)
        self.assertIn("no reasoning trace", note)

    def test_silently_dropped_image_reads_as_no_vision(self):
        state, note = D.probe_vision(UPBASE + "/thinker/v1", "bare-a", "k", "openai")
        self.assertIs(state, False)
        self.assertIn("silently dropped", note)

    def test_real_vision_detected_by_token_delta(self):
        state, note = D.probe_vision(UPBASE + "/seer/v1", "bare-a", "k", "openai")
        self.assertIs(state, True)
        self.assertIn("512 extra input tokens", note)

    def test_cheap_vision_detected(self):
        # b.ai/Qwen charge ~60 tokens for a 256px image — cheap enough that
        # the old 64-token floor read them as silent drops. A 512x512 probe
        # image must clear the floor with room to spare.
        state, note = D.probe_vision(UPBASE + "/cheapseer/v1", "bare-a", "k", "openai")
        self.assertIs(state, True)
        self.assertIn("extra input tokens", note)

    def test_vision_probe_sends_512px_image(self):
        # regression guard: shrinking the probe image back to 256px would
        # reintroduce the cheap-vision false negative on b.ai/Qwen
        Fake.hits.clear()
        state, note = D.probe_vision(UPBASE + "/seer/v1", "bare-a", "k", "openai")
        self.assertIs(state, True)
        # the image call is the one carrying an image_url; decode its size
        for hit in Fake.hits:
            msgs = (hit.get("body") or {}).get("messages") or []
            for m in msgs:
                parts = m.get("content")
                if not isinstance(parts, list):
                    continue
                for p in parts:
                    if not isinstance(p, dict) or p.get("type") != "image_url":
                        continue
                    url = (p.get("image_url") or {}).get("url", "")
                    b64 = url.split("base64,", 1)[-1]
                    import base64 as b64mod
                    data = b64mod.b64decode(b64)
                    # PNG IHDR: width at offset 16, height at 20 (big-endian)
                    w = int.from_bytes(data[16:20], "big")
                    h = int.from_bytes(data[20:24], "big")
                    self.assertEqual((w, h), (512, 512))
                    return
        self.fail("probe never sent an image_url")

    def test_quota_notice_yields_unknown_not_yes(self):
        r_state, r_note = D.probe_reasoning(UPBASE + "/faker/v1", "bare-a", "k", "openai")
        v_state, v_note = D.probe_vision(UPBASE + "/faker/v1", "bare-a", "k", "openai")
        self.assertIsNone(r_state)
        self.assertIsNone(v_state)
        self.assertIn("quota notice", r_note)
        self.assertIn("quota notice", v_note)

    def test_availability_error_inside_200_is_false(self):
        # the reported false positive: provider answers 200 but the body is
        # an error — a dead model looked available because the status was 2xx
        state, note = D.probe_availability(UPBASE + "/err200/v1", "bare-a", "k", "openai")
        self.assertIs(state, False)
        self.assertIn("inside an HTTP 200", note)
        self.assertIn("not available", note)

    def test_availability_responses_only_model_is_true(self):
        # chat 500s but /responses answers (opencode Zen Spark shape):
        # the model is up, on the other endpoint
        state, note = D.probe_availability(UPBASE + "/respondent/v1", "resp-a", "k", "openai")
        self.assertIs(state, True)
        self.assertIn("responses-only", note)

    def test_availability_full_reports_endpoint(self):
        state, note, endpoint = D.probe_availability_full(UPBASE + "/respondent/v1", "resp-a",
                                                           "k", "openai")
        self.assertIs(state, True)
        self.assertEqual(endpoint, "responses")
        state, note, endpoint = D.probe_availability_full(UPBASE + "/bare/v1", "bare-a",
                                                           "k", "openai")
        self.assertIs(state, True)
        self.assertIsNone(endpoint)

    def test_availability_responses_account_error_stays_inconclusive(self):
        # a billing failure on chat must NOT flip to True via /responses;
        # deadkey 401s every path, so the verdict stays inconclusive
        state, note = D.probe_availability(UPBASE + "/deadkey/v1", "bare-a", "bad", "openai")
        self.assertIsNone(state)
        self.assertIn("account error", note)

    def test_reasoning_error_inside_200_is_inconclusive(self):
        state, note = D.probe_reasoning(UPBASE + "/err200/v1", "bare-a", "k", "openai")
        self.assertIsNone(state)
        self.assertIn("inside an HTTP 200", note)

    def test_vision_error_inside_200_is_inconclusive(self):
        state, note = D.probe_vision(UPBASE + "/err200/v1", "bare-a", "k", "openai")
        self.assertIsNone(state)
        self.assertIn("inside an HTTP 200", note)

    def test_error_in_200_helper(self):
        self.assertEqual(D._error_in_200({"error": {"message": "boom"}}), "boom")
        self.assertEqual(D._error_in_200({"error": "plain"}), "plain")
        self.assertIsNone(D._error_in_200({"choices": []}))
        self.assertIsNone(D._error_in_200(None))
        self.assertIsNone(D._error_in_200({"error": {"message": ""}}))

    def test_inspect_model_combines_everything(self):
        # /thinker/v1 listing is bare (no capability flags), so inspect falls
        # to the probes. reasoning fast=True, vision full differential.
        # thinker silently drops images → differential catches it → no vision.
        res = D.inspect_model(UPBASE + "/thinker/v1", "bare-a", "k", "openai")
        self.assertIs(res["reasoning"], True)
        self.assertIn("fast", res["reasoning_note"])
        self.assertIs(res["vision"], False)
        self.assertIn("silently dropped", res["vision_note"])
        self.assertNotIn("context", res)

    def test_inspect_trusts_stated_capabilities_without_probing(self):
        # A listing that states the flags must be taken at face value: no
        # capability probes run. The only live call left is the cheap
        # availability ping.
        listing = {"id": "m1", "reasoning": True, "vision": False}
        Fake.hits.clear()
        res = D.inspect_model(UPBASE + "/thinker/v1", "m1", "k", "openai",
                              listing_item=listing)
        self.assertIs(res["reasoning"], True)
        self.assertIs(res["vision"], False)
        self.assertIn("stated by the provider listing", res["reasoning_note"])
        self.assertIn("stated by the provider listing", res["vision_note"])
        self.assertIs(res["available"], True)
        self.assertEqual(len(Fake.hits), 1)  # availability ping only

    def test_inspect_responses_only_model_probes_via_responses(self):
        res = D.inspect_model(UPBASE + "/respondent/v1", "resp-a", "k", "openai")
        self.assertIs(res["available"], True)
        self.assertIn("responses-only", res["available_note"])
        self.assertEqual(res["endpoint"], "responses")
        self.assertIs(res["reasoning"], True)
        self.assertIs(res["vision"], True)
        self.assertIn("/responses", res["vision_note"])

    def test_reasoning_via_responses(self):
        state, note = D.probe_reasoning(UPBASE + "/respondent/v1", "resp-a", "k",
                                        "openai", fast=True, endpoint="responses")
        self.assertIs(state, True)

    def test_vision_via_responses(self):
        state, note = D.probe_vision(UPBASE + "/respondent/v1", "resp-a", "k",
                                     "openai", endpoint="responses")
        self.assertIs(state, True)
        self.assertIn("300 extra input tokens", note)

    def test_inspect_ask_declines_skip_probes(self):
        # ask=False: unstated capabilities are skipped, not probed
        asked = []

        def ask(q):
            asked.append(q)
            return False
        res = D.inspect_model(UPBASE + "/thinker/v1", "bare-a", "k", "openai",
                              ask=ask)
        self.assertIsNone(res["reasoning"])
        self.assertIn("declined", res["reasoning_note"])
        self.assertIsNone(res["vision"])
        self.assertIn("declined", res["vision_note"])
        self.assertEqual(len(asked), 2)

    def test_inspect_ask_yes_probes(self):
        res = D.inspect_model(UPBASE + "/thinker/v1", "bare-a", "k", "openai",
                              ask=lambda q: True)
        self.assertIs(res["reasoning"], True)
        self.assertIs(res["vision"], False)

    def test_probe_availability_dead_endpoint_is_false(self):
        # nothing listens on port 1: a connection failure is a dead route
        state, note = D.probe_availability("http://127.0.0.1:1/v1", "m", "k", "openai",
                                           timeout=3)
        self.assertIs(state, False)

    def test_probe_availability_live_endpoint_is_true(self):
        state, note = D.probe_availability(UPBASE + "/bare/v1", "bare-a", "k", "openai")
        self.assertIs(state, True)

    def test_probe_availability_bad_key_is_inconclusive(self):
        # 401 is an account problem, not a dead model: must not read as False
        state, note = D.probe_availability(UPBASE + "/deadkey/v1", "bare-a", "wrong", "openai")
        self.assertIsNone(state)

    def test_inspect_mixes_stated_and_probed(self):
        # reasoning stated, vision not: only vision gets a probe, and it must
        # use the full differential (a 200 with a silently dropped image is
        # NOT evidence of vision).
        listing = {"id": "m1", "reasoning": True}
        res = D.inspect_model(UPBASE + "/thinker/v1", "bare-a", "k", "openai",
                              listing_item=listing)
        self.assertIs(res["reasoning"], True)
        self.assertIn("stated by the provider listing", res["reasoning_note"])
        self.assertIs(res["vision"], False)
        self.assertIn("silently dropped", res["vision_note"])


class TestHfConfigFacts(unittest.TestCase):
    def test_non_hub_id_skipped_without_network(self):
        # plain ids (gpt-4o-mini) and multi-slash ids are not Hub repos
        self.assertEqual(D.hf_config_facts("gpt-4o-mini"), {})
        self.assertEqual(D.hf_config_facts("a/b/c"), {})

    def test_arch_heuristic_rejects_text_gen_names(self):
        # T5/BART-style conditional generation is text-only, not vision
        self.assertIs(D._arch_has_vision("T5ForConditionalGeneration", {}), False)
        self.assertIs(D._arch_has_vision("BartForConditionalGeneration", {}), False)
        self.assertIs(D._arch_has_vision("LlamaForCausalLM", {}), False)
        self.assertIs(D._arch_has_vision("Qwen2VLForConditionalGeneration", {}), True)
        self.assertIs(D._arch_has_vision("LlavaLlamaForCausalLM", {"vision_tower": {}}), True)
        self.assertIs(D._arch_has_vision("", {}), False)


class TestQuotaSentinel(unittest.TestCase):
    """A 200 that is really a billing notice must not count as a capability."""

    def test_topup_text_detected(self):
        payload = {"id": "chatcmpl-fake-1", "usage": {"total_tokens": 0},
                   "choices": [{"message": {"role": "assistant", "content":
                       "Sorry, to prevent abuse of free resources, accounts that have "
                       "not been recharged can only try 10 times."}}]}
        self.assertTrue(D.looks_faked(payload))

    def test_fake_id_with_zero_usage_detected(self):
        payload = {"id": "chatcmpl-fake-999", "usage": {"total_tokens": 0},
                   "choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        self.assertTrue(D.looks_faked(payload))

    def test_real_completion_not_flagged(self):
        payload = {"id": "chatcmpl-abc", "usage": {"total_tokens": 7},
                   "choices": [{"message": {"role": "assistant", "content": "hello"}}]}
        self.assertFalse(D.looks_faked(payload))

    def test_anthropic_shape(self):
        payload = {"id": "msg_1", "content": [{"type": "text", "text": "please recharge"}]}
        self.assertTrue(D.looks_faked(payload, "anthropic"))


class TestTranslate(unittest.TestCase):
    def test_system_message_hoisted(self):
        out = T.openai_to_anthropic({"model": "m", "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"}]})
        self.assertEqual(out["system"], "be nice")
        self.assertEqual(out["messages"], [{"role": "user", "content": "hi"}])
        self.assertIn("max_tokens", out)

    def test_thinking_budget_never_meets_max_tokens(self):
        """Anthropic 400s when budget_tokens >= max_tokens; the translator
        must lift the cap so reasoning requests on small budgets still go."""
        out = T.openai_to_anthropic({"model": "m", "reasoning_effort": "high",
                                     "max_tokens": 50, "messages": [
                                         {"role": "user", "content": "hi"}]})
        self.assertEqual(out["thinking"]["budget_tokens"], 4096)
        self.assertGreater(out["max_tokens"], out["thinking"]["budget_tokens"])
        # the 4096 default cap hits the same wall without an explicit max
        out2 = T.openai_to_anthropic({"model": "m", "reasoning_effort": "max",
                                      "messages": [{"role": "user", "content": "hi"}]})
        self.assertGreater(out2["max_tokens"], out2["thinking"]["budget_tokens"])
        # no thinking requested: cap untouched
        out3 = T.openai_to_anthropic({"model": "m", "max_tokens": 50, "messages": [
            {"role": "user", "content": "hi"}]})
        self.assertEqual(out3["max_tokens"], 50)

    def test_consecutive_roles_merged(self):
        out = T.openai_to_anthropic({"model": "m", "messages": [
            {"role": "user", "content": "a"}, {"role": "user", "content": "b"}]})
        self.assertEqual(len(out["messages"]), 1)
        self.assertIn("a", out["messages"][0]["content"])
        self.assertIn("b", out["messages"][0]["content"])

    def test_content_parts_become_blocks(self):
        """Text parts stay text; block form is required to carry images."""
        out = T.openai_to_anthropic({"model": "m", "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}]})
        blocks = out["messages"][0]["content"]
        self.assertEqual([b["text"] for b in blocks], ["x", "y"])

    def test_image_part_survives_translation(self):
        """A vision request used to reach the model silently text-only."""
        png = "iVBORw0KGgo="
        out = T.openai_to_anthropic({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}},
        ]}]})
        blocks = out["messages"][0]["content"]
        kinds = [b["type"] for b in blocks]
        self.assertIn("image", kinds)
        img = blocks[kinds.index("image")]
        self.assertEqual(img["source"]["media_type"], "image/png")
        self.assertEqual(img["source"]["data"], png)

    def test_undecodable_image_is_reported_not_dropped(self):
        out = T.openai_to_anthropic({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}},
        ]}]})
        blocks = out["messages"][0]["content"]
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("image omitted", blocks[0]["text"])

    def test_merge_keeps_images_across_same_role_turns(self):
        out = T.openai_to_anthropic({"model": "m", "messages": [
            {"role": "user", "content": "first"},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]},
        ]})
        self.assertEqual(len(out["messages"]), 1)
        kinds = [b["type"] for b in out["messages"][0]["content"]]
        self.assertEqual(kinds, ["text", "image"])

    def test_zero_max_tokens_is_treated_as_unset(self):
        out = T.openai_to_anthropic({"model": "m", "max_tokens": 0,
                                     "messages": [{"role": "user", "content": "h"}]})
        self.assertEqual(out["max_tokens"], 4096)

    def test_reasoning_effort_maps_to_thinking(self):
        out = T.openai_to_anthropic({"model": "m",
                                     "messages": [{"role": "user", "content": "h"}],
                                     "reasoning_effort": "max"})
        self.assertIn("thinking", out)
        self.assertEqual(out["thinking"]["type"], "enabled")
        self.assertGreater(out["thinking"]["budget_tokens"], 10000)

    def test_reasoning_effort_low_small_budget(self):
        out = T.openai_to_anthropic({"model": "m",
                                     "messages": [{"role": "user", "content": "h"}],
                                     "reasoning_effort": "low"})
        self.assertEqual(out["thinking"]["budget_tokens"], 1024)

    def test_preexisting_thinking_passed_through(self):
        out = T.openai_to_anthropic({"model": "m",
                                     "messages": [{"role": "user", "content": "h"}],
                                     "thinking": {"type": "enabled", "budget_tokens": 5000}})
        self.assertEqual(out["thinking"]["budget_tokens"], 5000)

    def test_no_thinking_when_unset(self):
        out = T.openai_to_anthropic({"model": "m",
                                     "messages": [{"role": "user", "content": "h"}]})
        self.assertNotIn("thinking", out)

    def test_stop_string_becomes_list(self):
        out = T.openai_to_anthropic({"model": "m", "messages": [{"role": "user", "content": "h"}], "stop": "END"})
        self.assertEqual(out["stop_sequences"], ["END"])

    def test_anthropic_to_openai_extracts_tool_use(self):
        got = T.anthropic_to_openai({
            "id": "msg_2", "model": "claude", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "toolu_7",
                         "name": "get_time", "input": {"x": 1}}],
            "usage": {"input_tokens": 8, "output_tokens": 3}}, "claude")
        msg = got["choices"][0]["message"]
        self.assertEqual(msg["content"], "checking")
        self.assertEqual(msg["tool_calls"], [{"id": "toolu_7", "type": "function",
                                              "function": {"name": "get_time",
                                                           "arguments": '{"x": 1}'}}])
        self.assertEqual(got["choices"][0]["finish_reason"], "tool_calls")

    def test_openai_to_anthropic_forwards_tools(self):
        out = T.openai_to_anthropic({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "get_time", "description": "clock",
                "parameters": {"type": "object", "properties": {}}}}],
            "tool_choice": "none"})
        self.assertEqual(out["tools"], [{"name": "get_time", "description": "clock",
                                         "input_schema": {"type": "object", "properties": {}}}])
        self.assertEqual(out["tool_choice"], {"type": "none"})

    def test_openai_to_anthropic_keeps_tool_history(self):
        out = T.openai_to_anthropic({
            "model": "m", "messages": [
                {"role": "user", "content": "time?"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_time", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "noon"}]})
        kinds = []
        for m in out["messages"]:
            c = m["content"]
            kinds.extend(b.get("type") for b in (c if isinstance(c, list) else []))
        self.assertIn("tool_use", kinds)
        self.assertIn("tool_result", kinds)
        texts = []
        for m in out["messages"]:
            c = m["content"]
            texts.extend(b.get("text") for b in (c if isinstance(c, list) else [])
                         if b.get("type") == "text")
        self.assertNotIn("", texts)

    def test_chat_to_responses_keeps_images(self):
        out = T.chat_to_responses({
            "model": "m", "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "see?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]}]})
        self.assertIsInstance(out["input"], list)
        blob = out["input"][0]["content"]
        self.assertIn({"type": "input_text", "text": "see?"}, blob)
        self.assertIn({"type": "input_image", "image_url": "data:image/png;base64,AAA"}, blob)

    def test_response_converted(self):
        got = T.anthropic_to_openai({
            "id": "msg_1", "model": "claude", "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 3, "output_tokens": 2}}, "claude")
        self.assertEqual(got["choices"][0]["message"]["content"], "hello")
        self.assertEqual(got["choices"][0]["finish_reason"], "length")
        self.assertEqual(got["usage"]["total_tokens"], 5)

    def test_chat_to_responses(self):
        out = T.chat_to_responses({"model": "m", "max_tokens": 44, "temperature": 0.5,
                                   "messages": [{"role": "system", "content": "be brief"},
                                                {"role": "user", "content": "hi"}]})
        self.assertEqual(out["model"], "m")
        self.assertEqual(out["input"], "System: be brief\n\nhi")
        self.assertEqual(out["max_output_tokens"], 44)
        self.assertEqual(out["temperature"], 0.5)
        self.assertNotIn("stream", out)

    def test_chat_to_responses_forwards_tools(self):
        out = T.chat_to_responses({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "get_time", "description": "clock",
                "parameters": {"type": "object", "properties": {}}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_time"}}})
        self.assertEqual(out["tools"], [{"type": "function", "name": "get_time",
                                         "description": "clock",
                                         "parameters": {"type": "object", "properties": {}}}])
        self.assertEqual(out["tool_choice"], {"type": "function", "name": "get_time"})

    def test_chat_to_responses_keeps_tool_history(self):
        out = T.chat_to_responses({
            "model": "m", "messages": [
                {"role": "user", "content": "what time is it"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "get_time", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "noon"}]})
        self.assertIn("Assistant called: get_time({})", out["input"])
        self.assertIn("Tool result: noon", out["input"])

    def test_responses_to_chat_extracts_function_calls(self):
        obj = {"id": "resp_2", "model": "spark", "output": [
            {"type": "function_call", "call_id": "call_9",
             "name": "get_time", "arguments": "{}"}],
            "usage": {"input_tokens": 4, "output_tokens": 2}}
        got = T.responses_to_chat(obj, "spark")
        msg = got["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"], [{"id": "call_9", "type": "function",
                                              "function": {"name": "get_time",
                                                           "arguments": "{}"}}])
        self.assertEqual(got["choices"][0]["finish_reason"], "tool_calls")

    def test_responses_to_chat(self):
        obj = {"id": "resp_1", "model": "spark", "output": [
            {"type": "reasoning", "status": "completed"},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "hello"}]}],
            "usage": {"input_tokens": 6, "output_tokens": 3}}
        got = T.responses_to_chat(obj, "spark")
        self.assertEqual(got["object"], "chat.completion")
        self.assertEqual(got["choices"][0]["message"]["content"], "hello")
        self.assertEqual(got["usage"], {"prompt_tokens": 6, "completion_tokens": 3,
                                        "total_tokens": 9})

    def test_stream_translation(self):
        tr = T.StreamTranslator("m")
        out = b""
        for line in ['data: {"type":"message_start","message":{}}',
                     'data: {"type":"content_block_delta","delta":{"text":"hi"}}',
                     'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
                     'data: {"type":"message_stop"}']:
            out += tr.feed(line)
        text = out.decode()
        self.assertIn('"role": "assistant"', text)
        self.assertIn('"content": "hi"', text)
        self.assertIn("[DONE]", text)

    def test_stream_ignores_event_lines(self):
        tr = T.StreamTranslator("m")
        self.assertEqual(tr.feed("event: message_start"), b"")
        self.assertEqual(tr.feed(""), b"")

    def test_stream_translator_rebuilds_tool_calls(self):
        tr = T.StreamTranslator("m")
        out = b""
        for line in ['data: {"type":"content_block_start","index":0,'
                     '"content_block":{"type":"tool_use","id":"toolu_1","name":"get_time"}}',
                     'data: {"type":"content_block_delta","index":0,'
                     '"delta":{"type":"input_json_delta","partial_json":"{\\"x\\""}}',
                     'data: {"type":"content_block_delta","index":0,'
                     '"delta":{"type":"input_json_delta","partial_json":": 1}"}}',
                     'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
                     'data: {"type":"message_stop"}']:
            out += tr.feed(line)
        text = out.decode()
        self.assertIn('"name": "get_time"', text)
        self.assertIn('toolu_1', text)
        self.assertIn('{\\"x\\"', text)
        self.assertIn(': 1}', text)
        self.assertIn('"finish_reason": "tool_calls"', text)


class ServerCase(unittest.TestCase):
    """Boots a real CG instance against the fake upstream."""

    def boot(self, providers):
        self.dir = tempfile.mkdtemp()
        path = os.path.join(self.dir, "config.json")
        cfg = C.default_config()
        cfg["providers"] = providers
        C.save(cfg, path)
        self.httpd, self.state = serve(path, "127.0.0.1", 0)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        return self.base

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()


class TestServer(ServerCase):
    def test_models_endpoint_namespaces(self):
        p = C.new_provider("rich", UPBASE + "/rich/v1", ["k"], flavor="openai")
        p["models"] = {"rich-a": {"reasoning": True, "vision": False}}
        base = self.boot([p])
        r = H.get(base + "/v1/models")
        self.assertTrue(r.ok)
        m = r.json()["data"][0]
        self.assertEqual(m["id"], "rich/rich-a")
        self.assertNotIn("context_length", m)

    def test_chat_proxied(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "bare/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "bare ok")

    def test_responses_only_model_served_as_chat(self):
        p = C.new_provider("resp", UPBASE + "/respondent/v1", ["k"], flavor="openai")
        p["models"] = {"resp-a": {"endpoint": "responses"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "resp/resp-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        body = r.json()
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "resp ok")

    def test_responses_only_model_streamed_as_chat(self):
        p = C.new_provider("resp", UPBASE + "/respondent/v1", ["k"], flavor="openai")
        p["models"] = {"resp-a": {"endpoint": "responses"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "resp/resp-a", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        self.assertIn("resp ok", r.text())
        self.assertIn('"finish_reason": "stop"', r.text())
        self.assertIn("[DONE]", r.text())

    def test_logs_endpoint_records_key(self):
        p = C.new_provider("logs", UPBASE + "/logs/v1", ["k1", "k2"],
                           rotation="round_robin", flavor="openai")
        p["models"] = {"m": {}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "logs/m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        r = H.get(base + "/v1/logs")
        self.assertTrue(r.ok)
        entries = (r.json() or {}).get("entries", [])
        self.assertTrue(entries, "log must contain the proxied request")
        last = entries[-1]
        self.assertEqual(last["key"], "k1")
        self.assertEqual(last["status"], 200)
        self.assertEqual(last["model"], "logs/m")

    def test_disabled_model_not_routable_nor_listed(self):
        """A model toggled off disappears from /v1/models and 404s in both
        'provider/model' and bare-name forms."""
        p = C.new_provider("off", UPBASE + "/off/v1", ["k"], flavor="openai")
        p["models"] = {"alive": {}, "dead": {"enabled": False}}
        base = self.boot([p])
        r = H.get(base + "/v1/models")
        self.assertTrue(r.ok)
        ids = [m["id"] for m in (r.json() or {}).get("data", [])]
        self.assertIn("off/alive", ids)
        self.assertNotIn("off/dead", ids)
        r = H.post(base + "/v1/chat/completions",
                   {"model": "off/dead", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 404)
        r = H.post(base + "/v1/chat/completions",
                   {"model": "dead", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 404)
        # the alive twin still routes fine
        r = H.post(base + "/v1/chat/completions",
                   {"model": "off/alive", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())

    def test_unknown_model_404(self):
        base = self.boot([])
        r = H.post(base + "/v1/chat/completions", {"model": "nope/x", "messages": []})
        self.assertEqual(r.status, 404)

    def test_bare_model_name_routes_when_unique(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())

    def test_fill_first_failover_on_429(self):
        p = C.new_provider("flaky", UPBASE + "/flaky/v1", ["bad1", "bad2", "good"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "flaky/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, "client should never see the 429: %s" % r.text())
        self.assertEqual(r.json()["choices"][0]["message"]["content"], "flaky ok")
        ring = self.state.registry.get(self.state.cfg["providers"][0])
        self.assertEqual(ring.state[0].status_text()[:4], "cool")
        self.assertEqual(ring.state[2].status_text(), "ok")

    def test_dead_key_marked_and_skipped(self):
        p = C.new_provider("dead", UPBASE + "/deadkey/v1", ["bad", "good"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "dead/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        ring = self.state.registry.get(self.state.cfg["providers"][0])
        self.assertTrue(ring.state[0].dead)
        self.assertEqual(ring.try_order(), [1])

    def test_all_keys_bad_returns_upstream_error(self):
        p = C.new_provider("dead", UPBASE + "/deadkey/v1", ["bad1", "bad2"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "dead/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 401)

    def test_streaming_passthrough(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "bare/bare-a", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
        text = r.text(4000)
        self.assertIn("chat.completion.chunk", text)
        self.assertIn("[DONE]", text)

    def test_anthropic_upstream_translated_nonstreaming(self):
        p = C.new_provider("anth", UPBASE + "/anthropic/v1", ["k"], flavor="anthropic")
        p["models"] = {"claude-fake": {"context": 200000, "source": "listing"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "anth/claude-fake",
                    "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        obj = r.json()
        self.assertEqual(obj["object"], "chat.completion")
        self.assertEqual(obj["choices"][0]["message"]["content"], "hello from anthropic")

    def test_anthropic_upstream_translated_streaming(self):
        p = C.new_provider("anth", UPBASE + "/anthropic/v1", ["k"], flavor="anthropic")
        p["models"] = {"claude-fake": {"context": 200000, "source": "listing"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "anth/claude-fake", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}]})
        text = r.text(4000)
        self.assertIn("chat.completion.chunk", text)
        self.assertIn("streamed", text)
        self.assertIn("[DONE]", text)

    def test_healthz_reports_keys(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["a", "b"], flavor="openai")
        base = self.boot([p])
        r = H.get(base + "/healthz")
        self.assertTrue(r.ok)
        data = r.json()
        self.assertEqual(data["providers"][0]["name"], "bare")
        self.assertEqual(len(data["providers"][0]["keys"]), 2)

    def test_disabled_provider_hidden(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["a"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        p["enabled"] = False
        base = self.boot([p])
        self.assertEqual(H.get(base + "/v1/models").json()["data"], [])

    def test_manual_context_exposed_in_models(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {"context": 128000, "context_source": "manual"}}
        base = self.boot([p])
        r = H.get(base + "/v1/models")
        models = r.json()["data"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["context"], 128000)

    def test_reasoning_effort_passed_through_for_openai(self):
        Fake.hits.clear()
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "bare/bare-a", "reasoning_effort": "max",
                    "messages": [{"role": "user", "content": "think"}]})
        self.assertTrue(r.ok, r.text())
        self.assertEqual(Fake.hits[-1]["model"], "bare-a")
        self.assertEqual(Fake.hits[-1]["body"].get("reasoning_effort"), "max")

    def test_reasoning_effort_translated_to_thinking_for_anthropic(self):
        Fake.hits.clear()
        p = C.new_provider("anth", UPBASE + "/anthropic/v1", ["k"], flavor="anthropic")
        p["models"] = {"claude-fake": {}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "anth/claude-fake", "reasoning_effort": "high",
                    "messages": [{"role": "user", "content": "think"}]})
        self.assertTrue(r.ok, r.text())
        thinking = (Fake.hits[-1]["body"] or {}).get("thinking")
        self.assertIsNotNone(thinking)
        self.assertEqual(thinking["type"], "enabled")
        self.assertEqual(thinking["budget_tokens"], 4096)

    def test_config_hot_reload(self):
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["a"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        self.assertEqual(len(H.get(base + "/v1/models").json()["data"]), 1)
        cfg = C.load(self.state.path)
        cfg["providers"][0]["models"]["bare-z"] = {"context": 2000, "source": "manual"}
        time.sleep(0.01)
        C.save(cfg, self.state.path)
        # the mtime check is throttled so every request doesn't stat the file
        time.sleep(SV.MTIME_CHECK_INTERVAL + 0.1)
        ids = [m["id"] for m in H.get(base + "/v1/models").json()["data"]]
        self.assertIn("bare/bare-z", ids)

    def test_corrupt_config_does_not_blank_a_running_server(self):
        """A bad hand-edit must not silently empty a live gateway."""
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["a"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        self.assertEqual(len(H.get(base + "/v1/models").json()["data"]), 1)
        time.sleep(0.01)
        with open(self.state.path, "w") as fh:
            fh.write("{ truncated")
        time.sleep(SV.MTIME_CHECK_INTERVAL + 0.1)
        self.assertEqual(len(H.get(base + "/v1/models").json()["data"]), 1)
        health = H.get(base + "/healthz").json()
        self.assertFalse(health["ok"])
        self.assertIn("config_error", health)

    def test_malformed_json_body_is_400(self):
        base = self.boot([])
        import urllib.request
        req = urllib.request.Request(
            base + "/v1/chat/completions", data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            self.fail("expected an HTTP error")
        except urllib.error.HTTPError as e:
            body = e.read()
            e.close()
            self.assertEqual(e.code, 400)
            self.assertIn("invalid JSON", body.decode())

    def test_missing_model_field_is_400(self):
        base = self.boot([])
        r = H.post(base + "/v1/chat/completions", {"messages": []})
        self.assertEqual(r.status, 400)
        self.assertIn("model", r.text())

    def test_revive_endpoint_clears_dead_keys(self):
        p = C.new_provider("dead", UPBASE + "/deadkey/v1", ["bad", "good"], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        H.post(base + "/v1/chat/completions",
               {"model": "dead/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        ring = self.state.registry.get(self.state.cfg["providers"][0])
        self.assertTrue(ring.state[0].dead)
        r = H.post(base + "/v1/revive", {"provider": "dead"})
        self.assertTrue(r.ok, r.text())
        self.assertEqual(r.json()["revived"], 1)
        self.assertFalse(ring.state[0].dead)
        self.assertEqual(ring.try_order(), [0, 1])

    def test_concurrent_requests_during_reload(self):
        """Readers must never see a half-swapped config."""
        p = C.new_provider("bare", UPBASE + "/bare/v1", ["a"], flavor="openai")
        p["models"] = {"bare-%d" % i: {"context": 1000, "source": "default"} for i in range(20)}
        base = self.boot([p])
        errors = []

        def hammer():
            for _ in range(15):
                try:
                    data = H.get(base + "/v1/models", timeout=10).json()
                    if len(data.get("data") or []) != 20:
                        errors.append(len(data.get("data") or []))
                except Exception as e:  # noqa: BLE001
                    errors.append(repr(e))

        cfg = C.load(self.state.path)
        threads = [threading.Thread(target=hammer) for _ in range(6)]
        for t in threads:
            t.start()
        for _ in range(5):
            time.sleep(0.05)
            C.save(cfg, self.state.path)
        for t in threads:
            t.join(30)
        self.assertEqual(errors, [])

    def test_no_keys_returns_503(self):
        p = C.new_provider("nokeys", UPBASE + "/bare/v1", [], flavor="openai")
        p["models"] = {"bare-a": {"context": 1000, "source": "default"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "nokeys/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 503)


class TestCli(unittest.TestCase):
    def test_add_list_roundtrip(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            r = subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                                UPBASE + "/rich/v1", "k1", "k2"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("openai", r.stdout)
            self.assertIn("2 models", r.stdout)
            r2 = subprocess.run([sys.executable, cg, "--config", path, "list"],
                                capture_output=True, text=True, timeout=60)
            self.assertIn("rich-a", r2.stdout)
            self.assertIn("rsn  vis  avail", r2.stdout)
            cfg = C.load(path)
            self.assertEqual(len(cfg["providers"][0]["keys"]), 2)

    def test_inspect_writes_capabilities(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "ec",
                            UPBASE + "/errctx/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            r = subprocess.run([sys.executable, cg, "--config", path, "inspect",
                                "ec", "bare-a"],
                               capture_output=True, text=True, timeout=90)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("reasoning", r.stdout)
            meta = C.load(path)["providers"][0]["models"]["bare-a"]
            self.assertIn("reasoning", meta)
            self.assertIn("vision", meta)
            self.assertNotIn("context", meta)

    def test_toggle_disables_and_reenables(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            r = subprocess.run([sys.executable, cg, "--config", path, "toggle", "rich", "off"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("disabled", r.stdout)
            self.assertFalse(C.load(path)["providers"][0]["enabled"])
            # flip back without an explicit state
            subprocess.run([sys.executable, cg, "--config", path, "toggle", "rich"],
                           capture_output=True, text=True, timeout=60)
            self.assertTrue(C.load(path)["providers"][0]["enabled"])
            # unknown provider is an error
            r = subprocess.run([sys.executable, cg, "--config", path, "toggle", "nope", "on"],
                               capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)

    def test_context_set_clear_query(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            # query before any override — rich provider lists context
            r = subprocess.run([sys.executable, cg, "--config", path, "context",
                                "rich", "rich-a"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("512k", r.stdout)
            self.assertIn("listing", r.stdout)
            # set with a k-suffix (manual override)
            r = subprocess.run([sys.executable, cg, "--config", path, "context",
                                "rich", "rich-a", "128k"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("128k", r.stdout)
            meta = C.load(path)["providers"][0]["models"]["rich-a"]
            self.assertEqual(meta.get("context"), 128000)
            self.assertEqual(meta.get("context_source"), "manual")
            # query shows the override
            r = subprocess.run([sys.executable, cg, "--config", path, "context",
                                "rich", "rich-a"],
                               capture_output=True, text=True, timeout=60)
            self.assertIn("128k", r.stdout)
            self.assertIn("manual", r.stdout)
            # clear
            r = subprocess.run([sys.executable, cg, "--config", path, "context",
                                "rich", "rich-a", "clear"],
                               capture_output=True, text=True, timeout=60)
            self.assertIn("cleared", r.stdout)
            meta = C.load(path)["providers"][0]["models"]["rich-a"]
            self.assertNotIn("context", meta)
            # invalid size rejected
            r = subprocess.run([sys.executable, cg, "--config", path, "context",
                                "rich", "rich-a", "bogus"],
                               capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)

    def test_manual_context_appears_in_listing(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            subprocess.run([sys.executable, cg, "--config", path, "context",
                            "rich", "rich-a", "1m"],
                           capture_output=True, text=True, timeout=60)
            r = subprocess.run([sys.executable, cg, "--config", path, "list"],
                               capture_output=True, text=True, timeout=60)
            self.assertIn("1M", r.stdout)

    def test_context_from_rich_listing(self):
        """A provider that advertises `context_window` has it stored as listing."""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            cfg = C.load(path)
            meta = cfg["providers"][0]["models"]["rich-a"]
            self.assertEqual(meta.get("context"), 512000)
            self.assertEqual(meta.get("context_source"), "listing")

    def test_no_context_from_bare_listing(self):
        """A bare listing (no context fields) leaves context unset."""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "bare",
                            UPBASE + "/bare/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            cfg = C.load(path)
            meta = cfg["providers"][0]["models"]["bare-a"]
            self.assertNotIn("context", meta)
            self.assertNotIn("context_source", meta)

    def test_reset_clears_probed_values_and_manual_context(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            # seed probed values + a manual context override
            cfg = C.load(path)
            prov = cfg["providers"][0]
            prov["models"]["rich-a"] = {
                "reasoning": True, "reasoning_note": "n",
                "vision": False, "vision_note": "n2",
                "available": True, "available_note": "n3",
                "checked": "2026-08-31 00:00",
                "context": 128000, "context_source": "manual",
            }
            C.save(cfg, path)
            r = subprocess.run([sys.executable, cg, "--config", path, "reset",
                                "rich", "rich-a"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("reset", r.stdout)
            meta = C.load(path)["providers"][0]["models"]["rich-a"]
            for f in ("reasoning", "reasoning_note", "vision", "vision_note",
                      "available", "available_note", "checked",
                      "context", "context_source"):
                self.assertNotIn(f, meta)

    def test_reset_keeps_listing_context(self):
        """A context that came from the provider listing survives a reset."""
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            cfg = C.load(path)
            prov = cfg["providers"][0]
            prov["models"]["rich-a"] = {
                "reasoning": True, "reasoning_note": "n",
                "context": 512000, "context_source": "listing",
            }
            C.save(cfg, path)
            subprocess.run([sys.executable, cg, "--config", path, "reset",
                            "rich", "rich-a"],
                           capture_output=True, text=True, timeout=60)
            meta = C.load(path)["providers"][0]["models"]["rich-a"]
            self.assertNotIn("reasoning", meta)
            self.assertEqual(meta.get("context"), 512000)
            self.assertEqual(meta.get("context_source"), "listing")

    def test_reset_unknown_model_is_not_an_error(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "config.json")
            cg = os.path.join(ROOT, "cg")
            subprocess.run([sys.executable, cg, "--config", path, "add", "rich",
                            UPBASE + "/rich/v1", "k1"],
                           capture_output=True, text=True, timeout=60)
            r = subprocess.run([sys.executable, cg, "--config", path, "reset",
                                "rich", "nope"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("no saved values", r.stdout)


class TestUsageExtract(unittest.TestCase):
    def test_openai_shape(self):
        got = U.extract_openai_usage({"prompt_tokens": 10, "completion_tokens": 4,
                                      "total_tokens": 14})
        self.assertEqual(got, {"pin": 10, "pout": 4})

    def test_openai_cached_details(self):
        got = U.extract_openai_usage({"prompt_tokens": 10, "completion_tokens": 4,
                                      "prompt_tokens_details": {"cached_tokens": 7}})
        self.assertEqual(got, {"pin": 10, "pout": 4, "cached": 7})

    def test_openai_cache_hit_variant(self):
        got = U.extract_openai_usage({"prompt_tokens": 10, "completion_tokens": 4,
                                      "prompt_cache_hit_tokens": 3})
        self.assertEqual(got.get("cached"), 3)

    def test_responses_shape(self):
        got = U.extract_responses_usage({"input_tokens": 306, "output_tokens": 3,
                                        "input_tokens_details": {"cached_tokens": 2}})
        self.assertEqual(got, {"pin": 306, "pout": 3, "cached": 2})

    def test_responses_list_details(self):
        got = U.extract_responses_usage({"input_tokens": 5, "output_tokens": 1,
                                        "input_tokens_details": [{"cached_tokens": 4}]})
        self.assertEqual(got.get("cached"), 4)

    def test_anthropic_shape(self):
        got = U.extract_anthropic_usage({"input_tokens": 11, "output_tokens": 4,
                                        "cache_read_input_tokens": 9})
        self.assertEqual(got, {"pin": 11, "pout": 4, "cached": 9})

    def test_chunk_shape(self):
        got = U.extract_chunk_usage({"choices": [], "usage": {"prompt_tokens": 5,
                                     "completion_tokens": 2}})
        self.assertEqual(got, {"pin": 5, "pout": 2})

    def test_garbage_is_empty(self):
        self.assertEqual(U.extract_openai_usage(None), {})
        self.assertEqual(U.extract_openai_usage({"prompt_tokens": "x"}), {})
        self.assertEqual(U.extract_chunk_usage({"choices": []}), {})

    def test_translator_collects_anthropic_usage(self):
        tr = T.StreamTranslator("m")
        tr.feed('data: {"type":"message_start","message":{"usage":{"input_tokens":11,'
                '"cache_read_input_tokens":9}}}')
        tr.feed('data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
                '"usage":{"output_tokens":4}}')
        self.assertEqual(tr.usage(), {"pin": 11, "pout": 4, "cached": 9})

    def test_anthropic_fold_keeps_cache(self):
        obj = T.anthropic_to_openai({"id": "m", "model": "mm", "stop_reason": "end_turn",
                                    "content": [{"type": "text", "text": "hi"}],
                                    "usage": {"input_tokens": 11, "output_tokens": 4,
                                              "cache_read_input_tokens": 9}}, "mm")
        self.assertEqual(obj["usage"]["prompt_tokens_details"], {"cached_tokens": 9})


class TestUsageAggregate(unittest.TestCase):
    def test_totals_and_hit_rate(self):
        now = time.time()
        entries = [
            {"t": now, "model": "a/m1", "status": 200, "pin": 100, "pout": 20, "cached": 40},
            {"t": now, "model": "a/m1", "status": 200, "pin": 100, "pout": 10},
            {"t": now, "model": "b/m2", "status": 500},
            {"t": now, "model": "b/m2", "status": 200},
        ]
        agg = U.aggregate(entries)
        m1 = agg["by_model"]["a/m1"]
        self.assertEqual((m1["reqs"], m1["errs"]), (2, 0))
        self.assertEqual((m1["pin"], m1["pout"], m1["cached"]), (200, 30, 40))
        self.assertEqual(m1["unknown"], 0)
        self.assertAlmostEqual(U.hit_rate(m1), 20.0)
        m2 = agg["by_model"]["b/m2"]
        self.assertEqual((m2["reqs"], m2["errs"], m2["unknown"]), (2, 1, 1))
        self.assertEqual(sorted(agg["by_provider"]), ["a", "b"])

    def test_render_marks_unknown(self):
        now = time.time()
        out = U.render([{"t": now, "model": "a/m1", "status": 200}], days=7, by="model")
        self.assertIn("a/m1", out)
        self.assertIn("n/a", out)

    def test_summary_for_scopes(self):
        now = time.time()
        entries = [
            {"t": now - 100, "model": "a/m1", "status": 200, "pin": 100, "pout": 10, "cached": 25},
            {"t": now - 100, "model": "a/m2", "status": 200, "pin": 300, "pout": 30},
            {"t": now - 20 * 86400, "model": "a/m1", "status": 200, "pin": 50, "pout": 5},
        ]
        week = dict(U.summary_for(entries, "a/m1"))
        self.assertEqual(week["7d"]["reqs"], 1)
        self.assertEqual(week["7d"]["pin"], 100)
        self.assertEqual(week["30d"]["reqs"], 2)
        prov = dict(U.summary_for(entries, "a"))
        self.assertEqual(prov["7d"]["reqs"], 2)
        self.assertEqual(prov["7d"]["pin"], 400)
        self.assertEqual(prov["7d"]["cached"], 25)
        # a bare name never matches 'other/m1' by prefix
        other = dict(U.summary_for(entries, "m1"))
        self.assertEqual(other["7d"]["reqs"], 0)
        # legacy int windows and explicit (label, seconds) pairs
        legacy = dict(U.summary_for(entries, "a/m1", windows=(7,)))
        self.assertEqual(legacy["7d"]["reqs"], 1)
        sess = dict(U.summary_for(entries, "a/m1", windows=[("session", 3600)]))
        self.assertEqual(sess["session"]["reqs"], 1)

    def test_render_window_label(self):
        now = time.time()
        entries = [{"t": now - 100, "model": "a/m1", "status": 200,
                    "pin": 100, "pout": 10, "cached": 25}]
        out = U.render(entries, days="24h", by="model")
        self.assertIn("last 24h", out)
        self.assertIn("a/m1", out)
        out = U.render(entries, by="model", window=("session", 3600))
        self.assertIn("last session", out)
        # explicit-span quad: only entries inside [since, until] count
        out = U.render(
            [{"t": now - 5000, "model": "a/m1", "status": 200, "pin": 10, "pout": 1},
             {"t": now - 100, "model": "a/m1", "status": 200, "pin": 100, "pout": 10}],
            by="model", window=("session abc", 0, now - 1000, now - 50))
        self.assertIn("a/m1", out)
        self.assertIn("100", out)
        self.assertNotIn("110", out)

    def test_hermes_sessions_reads_db(self):
        import sqlite3
        import tempfile
        import os
        d = tempfile.mkdtemp()
        db = os.path.join(d, "state.db")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT,"
                    " display_name TEXT, started_at REAL, ended_at REAL,"
                    " message_count INTEGER)")
        con.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
                    " role TEXT, content TEXT)")
        con.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                    ("20260904_x", "cli", "work", 1000.0, 2000.0, 5))
        con.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?)",
                    ("20260905_y", "cli", None, 3000.0, None, 2))
        con.execute("INSERT INTO messages VALUES (?,?,?,?)",
                    (1, "20260904_x", "user", "  hello\nworld  "))
        con.commit()
        con.close()
        got = U.hermes_sessions(db_path=db)
        self.assertEqual([s["id"] for s in got], ["20260905_y", "20260904_x"])
        self.assertEqual(got[0]["name"], "20260905_y")
        self.assertIsNone(got[0]["ended"])
        self.assertEqual(got[1]["title"], "hello world")
        self.assertEqual(U.hermes_sessions(db_path=os.path.join(d, "nope.db")), [])

    def test_half_present_counts_agree(self):
        # pin-only entries are known (missing half = 0) in BOTH views
        now = time.time()
        entries = [{"t": now - 50, "model": "a/m1", "key": "k1", "status": 200,
                    "pin": 100}]
        agg = U.aggregate(entries)["by_model"]["a/m1"]
        summ = dict(U.summary_for(entries, "a/m1"))["7d"]
        self.assertEqual((agg["unknown"], agg["pin"]), (0, 100))
        self.assertEqual((summ["unknown"], summ["pin"]), (0, 100))


class TestUsageFile(unittest.TestCase):
    def test_roundtrip_and_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "usage.jsonl")
            U.append(path, {"t": 1.0, "model": "a/m", "status": 200, "pin": 3})
            with open(path, "a") as fh:
                fh.write("not json\n")
            back = U.load(path)
            self.assertEqual(len(back), 1)
            self.assertEqual(back[0]["pin"], 3)
            self.assertEqual(U.load(path, since=2.0), [])

    def test_trim_caps_size(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "usage.jsonl")
            for i in range(200):
                U.append(path, {"t": float(i), "model": "a/m", "status": 200,
                                "pin": i, "note": "x" * 200})
            U.trim(path, keep_bytes=4096)
            self.assertLess(os.path.getsize(path), 8192)
            back = U.load(path)
            self.assertTrue(back)
            self.assertEqual(back[-1]["pin"], 199)


class FakeScr:
    """Minimal curses window double for prompt(): canned keys in, text out."""

    def __init__(self, keys, width=80):
        self.keys = list(keys)
        self.width = width

    def getmaxyx(self):
        return (24, self.width)

    def move(self, y, x):
        pass

    def clrtoeol(self):
        pass

    def addnstr(self, y, x, s, n, *a):
        pass

    def refresh(self):
        pass

    def get_wch(self):
        if not self.keys:
            return "\n"
        return self.keys.pop(0)


class TestPromptLive(unittest.TestCase):
    @unittest.skipUnless(os.name != "nt", "curses unavailable on Windows")
    def test_on_change_fires_per_keystroke(self):
        from cgw.tui import Tui

        seen = []
        scr = FakeScr(["a", "b", "\x7f", "c", "\n"])
        got = Tui.prompt(Tui.__new__(Tui), scr, "filter: ", on_change=seen.append)
        self.assertEqual(got, "ac")
        self.assertEqual(seen, ["a", "ab", "a", "ac"])

    @unittest.skipUnless(os.name != "nt", "curses unavailable on Windows")
    def test_no_callback_behaves_as_before(self):
        from cgw.tui import Tui

        scr = FakeScr(["x", "y", "\n"])
        self.assertEqual(Tui.prompt(Tui.__new__(Tui), scr, "name: "), "xy")


class TestUsageServer(ServerCase):
    def _last_log(self, base, model):
        r = H.get(base + "/v1/logs?n=20")
        self.assertTrue(r.ok)
        hits = [e for e in r.json()["entries"] if e.get("model") == model]
        self.assertTrue(hits, "no log entry for %s" % model)
        return hits[-1]

    def test_nonstream_usage_logged_and_persisted(self):
        p = C.new_provider("chat", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "chat/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        entry = self._last_log(base, "chat/bare-a")
        self.assertEqual((entry.get("pin"), entry.get("pout"), entry.get("cached")), (5, 2, 1))
        rows = U.load(os.path.join(self.dir, "usage.jsonl"))
        self.assertTrue(any(e.get("model") == "chat/bare-a" and e.get("pin") == 5
                            for e in rows))

    def test_stream_usage_logged(self):
        p = C.new_provider("chat", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "chat/bare-a", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        entry = self._last_log(base, "chat/bare-a")
        self.assertEqual((entry.get("pin"), entry.get("pout"), entry.get("cached")), (5, 2, 1))

    def test_responses_usage_logged(self):
        p = C.new_provider("resp", UPBASE + "/respondent/v1", ["k"], flavor="openai")
        p["models"] = {"resp-a": {"endpoint": "responses"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "resp/resp-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertTrue(r.ok, r.text())
        entry = self._last_log(base, "resp/resp-a")
        self.assertEqual((entry.get("pin"), entry.get("pout"), entry.get("cached")), (6, 3, 2))

    def test_healthz_reports_boot_time(self):
        base = self.boot([])
        r = H.get(base + "/healthz")
        self.assertTrue(r.ok)
        started = (r.json() or {}).get("started")
        self.assertTrue(isinstance(started, int) and started <= time.time())

    def test_responses_tool_calls_survive_roundtrip(self):
        p = C.new_provider("resp", UPBASE + "/respondent/v1", ["k"], flavor="openai")
        p["models"] = {"resp-a": {"endpoint": "responses"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "resp/resp-a", "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "get_time"}}]})
        self.assertTrue(r.ok, r.text())
        msg = r.json()["choices"][0]["message"]
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "get_time")
        self.assertEqual(r.json()["choices"][0]["finish_reason"], "tool_calls")
        r = H.post(base + "/v1/chat/completions",
                   {"model": "resp/resp-a", "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": [{"type": "function", "function": {"name": "get_time"}}]})
        self.assertTrue(r.ok, r.text())
        self.assertIn("get_time", r.text())
        self.assertIn('"finish_reason": "tool_calls"', r.text())

    def test_garbled_responses_payload_logs_502(self):
        p = C.new_provider("gar", UPBASE + "/garbled/v1", ["k"], flavor="openai")
        p["models"] = {"gar-a": {"endpoint": "responses"}}
        base = self.boot([p])
        r = H.post(base + "/v1/chat/completions",
                   {"model": "gar/gar-a", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status, 502)
        r = H.get(base + "/v1/logs?n=5")
        entries = [e for e in r.json()["entries"] if e["model"] == "gar/gar-a"]
        self.assertTrue(entries)
        self.assertEqual(entries[-1]["status"], 502)

    def test_stats_command_reads_log(self):
        import subprocess
        p = C.new_provider("chat", UPBASE + "/bare/v1", ["k"], flavor="openai")
        p["models"] = {"bare-a": {}}
        base = self.boot([p])
        H.post(base + "/v1/chat/completions",
               {"model": "chat/bare-a", "messages": [{"role": "user", "content": "hi"}]})
        cg = os.path.join(ROOT, "cg")
        cfg = os.path.join(self.dir, "config.json")
        r = subprocess.run([sys.executable, cg, "--config", cfg, "stats", "--days", "1"],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("chat/bare-a", r.stdout)
        self.assertIn("5", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
