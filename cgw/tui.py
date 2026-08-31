"""Curses TUI for CG. Two panes: providers on the left, detail on the right."""

import curses
import os
import re
import select
import shutil
import subprocess
import sys
import textwrap
import threading
import time

from . import config as C
from . import detect as D
from . import http as H

def _ctx(val):
    """Human-readable context size: '200k' / '1.5M' / '-' when unset."""
    if val is None:
        return "-"
    try:
        n = int(val)
    except (TypeError, ValueError):
        return "-"
    if n >= 1_000_000:
        return "%gM" % (n / 1_000_000)
    if n >= 1000:
        return "%gk" % round(n / 1000)
    return str(n)


SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _glyph(val, cl):
    """Colored capability glyph: ✓ yes / ✗ no / · not checked."""
    if val is True:
        return "✓", cl["ok"]
    if val is False:
        return "✗", cl["bad"]
    return "·", cl["dim"]


def round_box(win, title=""):
    """Rounded-corner modal frame (╭─╮ / ╰─╯) with an inset title."""
    h, w = win.getmaxyx()
    try:
        win.addch(0, 0, "╭")
        win.addch(0, w - 1, "╮")
        win.addch(h - 1, 0, "╰")
        win.addch(h - 1, w - 1, "╯")
        for i in range(1, w - 1):
            win.addch(0, i, "─")
            win.addch(h - 1, i, "─")
        for i in range(1, h - 1):
            win.addch(i, 0, "│")
            win.addch(i, w - 1, "│")
    except curses.error:
        win.box()
    if title:
        win.addnstr(0, 2, " %s " % title[:w - 6], w - 4, curses.A_BOLD)


def _cap(val):
    """Tri-state capability marker: yes / no / not yet checked."""
    if val is True:
        return "yes"
    if val is False:
        return "no"
    return "-"


def _now():
    return time.strftime("%Y-%m-%d %H:%M")


class Tui:
    def __init__(self, path=None):
        self.path = path or C.CONFIG_PATH
        self.load_error = None
        self.cfg = self._load_or_keep(C.default_config())
        self.sel = 0
        self.msel = 0
        self.mtop = 0
        self.filter = ""
        self.focus = "left"
        self.msg = ""
        self.busy = None
        self._health = (None, 0.0)
        self.cl = {"ok": 0, "bad": 0, "warn": 0, "dim": curses.A_DIM, "hi": curses.A_BOLD}

    # ---------- helpers ----------

    def _load_or_keep(self, fallback):
        """Load the config, or keep what we have if the file is unreadable.

        Every edit here writes the whole config back, so accepting an empty
        fallback for a corrupt file would erase the real providers and keys on
        the next keypress. A broken file freezes editing instead.
        """
        try:
            cfg = C.load(self.path)
        except C.ConfigError as e:
            self.load_error = str(e)
            return fallback
        self.load_error = None
        return cfg

    def provs(self):
        return self.cfg.get("providers", [])

    def cur(self):
        p = self.provs()
        return p[self.sel] if p and 0 <= self.sel < len(p) else None

    def models(self, p=None):
        """Sorted (id, meta) pairs for the current provider, filter applied."""
        p = p or self.cur()
        if not p:
            return []
        items = sorted((p.get("models") or {}).items())
        if self.filter:
            f = self.filter.lower()
            items = [it for it in items if f in it[0].lower()]
        return items

    def cur_model(self):
        items = self.models()
        if items and 0 <= self.msel < len(items):
            return items[self.msel]
        return None, None

    def save(self):
        if self.load_error:
            # refuse to write a config we couldn't read: that's the data-loss path
            self.msg = "config unreadable — not saving (%s)" % self.load_error
            return False
        C.save(self.cfg, self.path)
        self.cfg = self._load_or_keep(self.cfg)
        return True

    # ---------- theme ----------

    def _init_colors(self):
        """Foreground-only muted accents on the terminal's default background.

        Monochrome pairs (2: default fg) so everything still reads on a
        8-color terminal; the ✓/✗ glyphs get the color when there is one.
        The selection bar is a tint of the accent blue (#0087d7, cube 32)
        blended over the terminal's real background: a solid #0087d7 slab
        reads brighter than the same-color glyphs, so it never *looks* like
        the bindings' blue. With OSC 11 the tint is computed for the actual
        background (pale blue on light themes, dim deep blue on dark);
        without an answer it falls back to the solid accent.
        """
        ACCENT_RGB = (0, 135, 215)  # #0087d7
        SEL_ALPHA = 0.30
        self.cl = {"ok": curses.A_BOLD, "bad": curses.A_DIM,
                   "warn": curses.A_BOLD, "dim": curses.A_DIM, "hi": curses.A_BOLD,
                   "sel": curses.A_REVERSE, "accent": curses.A_BOLD,
                   "right": curses.A_BOLD}
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            if curses.COLORS >= 256:
                ACCENT, RIGHT = 32, 172  # #0087d7 blue, #d78700 orange
                bg = self._query_terminal_bg()
                fill = 75  # free cube slot: rewritten to the tint when possible
                if bg is not None:
                    try:
                        r = int(SEL_ALPHA * ACCENT_RGB[0] + (1 - SEL_ALPHA) * bg[0])
                        g = int(SEL_ALPHA * ACCENT_RGB[1] + (1 - SEL_ALPHA) * bg[1])
                        b = int(SEL_ALPHA * ACCENT_RGB[2] + (1 - SEL_ALPHA) * bg[2])
                        curses.init_color(fill, r * 1000 // 255, g * 1000 // 255,
                                          b * 1000 // 255)
                    except curses.error:
                        fill = ACCENT  # solid accent if the terminal refuses OSC 4
                else:
                    fill = ACCENT
            else:
                ACCENT, fill, RIGHT = curses.COLOR_BLUE, curses.COLOR_BLUE, curses.COLOR_YELLOW
            for i, fg, bgc in ((1, curses.COLOR_GREEN, -1), (2, curses.COLOR_RED, -1),
                               (3, curses.COLOR_YELLOW, -1),
                               (4, -1, fill),     # selection bar: default ink on blue tint
                               (5, ACCENT, -1),   # blue text (headings, bindings, top bar)
                               (6, RIGHT, -1)):   # muted orange text (right pane labels)
                curses.init_pair(i, fg, bgc)
            self.cl["ok"] = curses.color_pair(1)
            self.cl["bad"] = curses.color_pair(2)
            self.cl["warn"] = curses.color_pair(3)
            self.cl["sel"] = curses.color_pair(4)
            self.cl["accent"] = curses.color_pair(5)
            self.cl["right"] = curses.color_pair(6)
        except curses.error:
            pass  # keep the bold/dim fallback

    @staticmethod
    def _query_terminal_bg():
        """Best-effort OSC 11 query: terminal default background, 0-255 RGB.

        Done outside curses: we raw-mode stdin, write the query, read the
        reply with a short timeout, then restore the terminal so curses can
        take over. Returns None when there is no tty or no reply.
        """
        try:
            import termios
            import tty
        except ImportError:
            return None
        if not (os.isatty(sys.stdin.fileno()) and os.isatty(sys.stdout.fileno())):
            return None
        old = termios.tcgetattr(sys.stdin.fileno())
        tty.setraw(sys.stdin.fileno())
        try:
            os.write(sys.stdout.fileno(), b"\x1b]11;?\x07")
            buf = b""
            deadline = time.monotonic() + 0.5
            while (time.monotonic() < deadline and b"\x07" not in buf
                   and b"\x1b\\" not in buf):
                r, _, _ = select.select([sys.stdin.fileno()], [], [], 0.1)
                if not r:
                    continue
                try:
                    buf += os.read(sys.stdin.fileno(), 256)
                except OSError:
                    break
        finally:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
        m = re.search(rb"11;rgb:([0-9a-f]{1,4})(?:/([0-9a-f]{1,4})/([0-9a-f]{1,4}))?(?:\x07|\x1b\\)", buf)
        if not m:
            return None

        def conv(h):
            x = int(h, 16)
            return x >> 8 if x > 255 else x

        a = conv(m.group(1))
        return (a, conv(m.group(2) or m.group(1)), conv(m.group(3) or m.group(1)))

    def _healthz(self):
        """Cached gateway liveness ('up'/'down'), re-probed at most every 30s."""
        val, ts = self._health
        if val is not None and time.time() - ts < 30:
            return val
        listen = self.cfg.get("listen") or {}
        url = "http://%s:%s/healthz" % (listen.get("host", "127.0.0.1"),
                                        listen.get("port", C.DEFAULT_PORT))
        try:
            r = H.get(url, timeout=1)
            val = "up" if r.ok else "down"
        except Exception:
            val = "down"
        self._health = (val, time.time())
        return val

    # Keep at least this many columns for the value, shortening the label if
    # the terminal is too narrow to hold both.
    MIN_INPUT_COLS = 10

    def prompt(self, scr, label, secret=False):
        """Read a line at the bottom of the screen. Never truncates the value.

        Hand-rolled rather than curses.getstr(), which echoes into the window
        and stops at the right margin: a 50-char API key typed after a 29-char
        label on an 80-column terminal came back one character short, and a
        silently shortened key is indistinguishable from a rejected one.

        Invariants, verified by tests/prompt_widths.py across many widths:
          * what the caller gets is exactly what was typed, at any width and
            any value length — the buffer is independent of the display;
          * drawing can never raise, so a narrow terminal cannot lose input;
          * the label is shortened, not the value, when space runs out;
          * the geometry is re-read every keystroke, so a mid-entry resize is
            handled.
        """
        buf = []
        while True:
            h, w = scr.getmaxyx()
            # Reserve room for the value first; the label yields when cramped.
            lab = label
            if w - len(lab) - 1 < self.MIN_INPUT_COLS:
                lab = label[: max(0, w - self.MIN_INPUT_COLS - 1)]
            room = max(1, w - len(lab) - 1)

            shown = ("*" * len(buf)) if secret else "".join(buf)
            tail = shown[-room:]
            try:
                scr.move(h - 1, 0)
                scr.clrtoeol()
                if lab:
                    scr.addnstr(h - 1, 0, lab, min(len(lab), w - 1), curses.A_BOLD)
                if tail:
                    scr.addnstr(h - 1, len(lab), tail, min(len(tail), w - len(lab) - 1))
                scr.move(h - 1, min(len(lab) + len(tail), w - 1))
                scr.refresh()
            except curses.error:
                # A cramped or resizing terminal must not cost us the buffer.
                pass

            try:
                ch = scr.get_wch()
            except KeyboardInterrupt:
                return ""
            except curses.error:
                continue  # interrupted read (e.g. SIGWINCH): just redraw
            except Exception:
                return "".join(buf).strip()

            if ch in ("\n", "\r", curses.KEY_ENTER, 10, 13):
                break
            if ch == "\x1b":  # ESC cancels
                return ""
            if ch in ("\x7f", "\b", curses.KEY_BACKSPACE, 263, 127, 8):
                if buf:
                    buf.pop()
                continue
            if ch == "\x15":  # ctrl-u clears the line
                buf = []
                continue
            if ch == "\x17":  # ctrl-w deletes the last word
                while buf and buf[-1].isspace():
                    buf.pop()
                while buf and not buf[-1].isspace():
                    buf.pop()
                continue
            if isinstance(ch, str) and ch.isprintable():
                buf.append(ch)
            # arrows, resize, function keys and other non-text are ignored

        return "".join(buf).strip()

    def confirm(self, scr, label):
        """Ask a yes/no question; bare Enter means yes (defaults to Y/n)."""
        val = self.prompt(scr, label + " [Y/n] ").strip().lower()
        return val == "" or val.startswith("y")

    # ---------- drawing ----------

    def draw(self, scr):
        scr.erase()
        h, w = scr.getmaxyx()
        cl = self.cl

        listen = self.cfg.get("listen") or {}
        host = listen.get("host", "127.0.0.1")
        port = listen.get("port", C.DEFAULT_PORT)
        health = None if self.busy else self._healthz()
        if health == "up":
            right, r_attr = "● running", cl["ok"]
        elif health == "down":
            right, r_attr = "● stopped", cl["bad"]
        else:
            right, r_attr = "http://%s:%s/v1" % (host, port), cl["dim"]
        if health is not None:
            right += "  http://%s:%s/v1" % (host, port)

        # title bar: name left, live gateway state right
        scr.addnstr(0, 0, " CG — Coeus Gateway", min(w - 1, 20), curses.A_BOLD)
        if right:
            scr.addnstr(0, max(0, w - len(right) - 1), right, len(right), r_attr)
        try:
            for i in range(w - 1):
                scr.addch(1, i, "─", curses.A_DIM)
        except curses.error:
            pass

        # pane geometry: providers | models | inspection
        left_w = max(14, min(22, w // 6))
        right_w = 0
        if w - left_w - 40 - 2 >= 22:
            right_w = min(34, w - left_w - 40 - 2)
        mid_x = left_w
        mid_w = (w - left_w - right_w - 2) if right_w else (w - left_w - 1)
        insp_x = mid_x + mid_w + 1
        insp_w = w - insp_x - 1 if right_w else 0

        # left pane: providers
        scr.addnstr(2, 0, " providers", left_w - 1,
                    curses.A_BOLD if self.focus == "left" else cl["dim"])
        for i, p in enumerate(self.provs()):
            if 3 + i >= h - 2:
                break
            if p.get("flavor") == "unknown":
                mark, m_attr = "!", cl["warn"]
            elif p.get("enabled", True):
                mark, m_attr = "●", cl["right"]
            else:
                mark, m_attr = "○", cl["dim"]
            cur = i == self.sel
            sel = cur and self.focus == "left"
            scr.addnstr(3 + i, 0, "▸" if cur else " ", 1,
                        cl["sel"] if sel else cl["dim"])
            scr.addnstr(3 + i, 1, "%s " % mark, 2, m_attr)
            name = p["name"][:max(4, left_w - 7)]
            scr.addnstr(3 + i, 3, name, len(name),
                        cl["sel"] if sel else cl["accent"])
            cnt = "%3d" % len(p.get("models") or {})
            cnt_x = 4 + len(name)
            if cnt_x + len(cnt) <= left_w:
                scr.addnstr(3 + i, cnt_x, cnt, len(cnt),
                            cl["sel"] if sel else curses.A_NORMAL)

        # divider between left and middle panes
        for y in range(2, h - 2):
            scr.addch(y, mid_x - 1, "│", curses.A_DIM)

        # middle pane: provider summary + model table
        p = self.cur()
        if p is None:
            scr.addnstr(3, mid_x + 1, "no providers yet — press 'a' to add one",
                        mid_w - 2, cl["dim"])
        else:
            nk = len(p.get("keys") or [])
            summ = "%s · %s · %s · %d keys%s" % (
                p["name"], p.get("flavor", "unknown"),
                p.get("rotation", "fill_first").replace("_", "-"), nk,
                "" if insp_w else " · " + p.get("base_url", ""))
            scr.addnstr(2, mid_x + 1, summ[: mid_w - 2], mid_w - 2, cl["accent"])

            models = self.models(p)
            total = len(p.get("models") or {})
            name_w = max(14, min(40, mid_w - 26))
            o_name = 3
            o_rsn = o_name + name_w + 1
            o_vis = o_rsn + 5
            o_av = o_vis + 5
            o_ctx = o_av + 6

            y = 3
            head_attr = ((curses.A_BOLD | curses.A_UNDERLINE)
                         if self.focus == "right" else curses.A_UNDERLINE)
            scr.addnstr(y, mid_x + o_name, "model", 5, head_attr)
            for off, lab in ((o_rsn, "rsn"), (o_vis, "vis"), (o_av, "avail"), (o_ctx, "ctx")):
                scr.addnstr(y, mid_x + off, lab, 5, head_attr)
            y += 1
            try:
                for i in range(mid_w - 1):
                    scr.addch(y, mid_x + i, "─", curses.A_DIM)
            except curses.error:
                pass
            y += 1

            body_h = max(1, h - 2 - y)
            if not models:
                if self.filter:
                    scr.addnstr(y, mid_x + 1, "no match for /%s (%d models)" % (self.filter, total),
                                mid_w - 2, cl["dim"])
                else:
                    scr.addnstr(y, mid_x + 1, "(none — press r to detect, or m to add by hand)",
                                mid_w - 2, cl["dim"])
            else:
                # keep the cursor inside the visible window
                if self.msel < self.mtop:
                    self.mtop = self.msel
                elif self.msel >= self.mtop + body_h:
                    self.mtop = self.msel - body_h + 1
                self.mtop = max(0, min(self.mtop, max(0, len(models) - body_h)))

                for row, (mid, meta) in enumerate(models[self.mtop:self.mtop + body_h]):
                    yy = y + row
                    idx = self.mtop + row
                    cur = idx == self.msel
                    sel = cur and self.focus == "right"
                    on = meta.get("enabled", True)
                    name_s = ("%-*s" % (name_w, mid[:name_w]))[: mid_w - o_name - 1]
                    scr.addnstr(yy, mid_x, "▸" if cur else " ", 1,
                                cl["sel"] if sel else cl["dim"])
                    scr.addnstr(yy, mid_x + 1,
                                "●" if on else "○", 1,
                                cl["sel"] if sel else
                                (cl["right"] if on else cl["dim"]))
                    scr.addnstr(yy, mid_x + o_name, name_s, mid_w - o_name - 1,
                                cl["sel"] if sel else
                                (curses.A_DIM if not on else curses.A_NORMAL))
                    # capability glyphs and ctx are drawn ON the selection bar:
                    # the bar must not blank the data for the row you're on.
                    for off, val in ((o_rsn, meta.get("reasoning")),
                                     (o_vis, meta.get("vision")),
                                     (o_av, meta.get("available"))):
                        g, ga = _glyph(val, cl)
                        scr.addnstr(yy, mid_x + off, g, 1,
                                    cl["sel"] if sel else
                                    (cl["dim"] if not on else ga))
                    ctx_s = "%-5s" % _ctx(meta.get("context"))
                    scr.addnstr(yy, mid_x + o_ctx, ctx_s, 5,
                                cl["sel"] if sel else
                                (cl["dim"] if not on else curses.A_NORMAL))

                foot = "%d/%d" % (self.msel + 1, len(models))
                if self.filter:
                    foot += "  /%s" % self.filter
                if len(models) < total:
                    foot += "  (of %d)" % total
                scr.addnstr(h - 2, mid_x + max(0, mid_w - len(foot) - 1), foot, len(foot), cl["right"])

        # right pane: provider + selected-model inspection (layout B)
        if insp_w:
            for y in range(2, h - 2):
                scr.addch(y, insp_x - 1, "│", curses.A_DIM)
            scr.addnstr(2, insp_x + 1, " inspection", insp_w - 1, curses.A_BOLD)
            y = 3
            if p is None:
                scr.addnstr(y, insp_x + 1, "(no provider selected)", insp_w - 2, cl["accent"])
            else:
                for k, v in (("url", p.get("base_url", "")),
                             ("flavor", p.get("flavor", "unknown")),
                             ("rot", p.get("rotation", "fill_first").replace("_", "-")),
                             ("keys", "%d: %s" % (len(p.get("keys") or []),
                                                  ", ".join(k.get("label", "k")
                                                            for k in p.get("keys") or [])))):
                    scr.addnstr(y, insp_x + 1, "%-7s" % k, 7, cl["right"])
                    scr.addnstr(y, insp_x + 9, v[: insp_w - 10], max(0, insp_w - 10))
                    y += 1
                    if y >= h - 3:
                        y = h - 3  # clamp: short windows must not write past the pane
                y += 1
                mid, meta = self.cur_model()
                if not mid:
                    scr.addnstr(y, insp_x + 1, "(no models yet)", insp_w - 2, cl["accent"])
                else:
                    meta = dict(meta or {})
                    on = meta.get("enabled", True)
                    scr.addnstr(y, insp_x + 1, " %s" % mid[: insp_w - 3],
                                insp_w - 2, curses.A_BOLD)
                    y += 1
                    rows = []
                    rows.append(("on",
                                 "%s %s" % ("●" if on else "○",
                                            "enabled" if on else "disabled"),
                                 cl["ok"] if on else cl["bad"]))
                    ctx = meta.get("context")
                    rows.append(("ctx", "%s (%s)" % (
                        _ctx(ctx), "listing" if meta.get("context_source") == "listing"
                        else "manual" if ctx is not None else "unset"), curses.A_NORMAL))
                    for k, key in (("rsn", "reasoning"), ("vis", "vision"), ("avail", "available")):
                        v = meta.get(key)
                        g, ga = _glyph(v, cl)
                        note = meta.get(key + "_note") or ""
                        rows.append((k, "%s  %s" % (g, note), ga if v is not None else cl["dim"]))
                    chk = meta.get("checked")
                    if chk:
                        rows.append(("check", chk, cl["accent"]))
                    for k, v, a in rows:
                        if y >= h - 3:
                            break
                        scr.addnstr(y, insp_x + 1, "%-7s" % k, 7, cl["right"])
                        scr.addnstr(y, insp_x + 9, v[: insp_w - 10], max(0, insp_w - 10), a)
                        y += 1
                y = min(y, h - 3)
                y += 1
                # full binding list, one key per line
                binds = [("a", "add"), ("e", "edit"), ("d", "del"), ("K", "keys"),
                         ("r", "refresh"), ("t", "on/off"), ("T", "all-models"),
                         ("j/k", "move"), ("ENTER", "inspect"), ("c", "context"),
                         ("x", "reset"), ("m", "add"), ("/", "filter"), ("A", "avail-all"),
                         ("R", "revive"), ("l", "logs"), ("tab", "switch"),
                         ("?", "help"), ("q", "quit")]
                if y < h - 2:
                    scr.addnstr(y, insp_x + 1, "keys", insp_w - 2, curses.A_NORMAL)
                    y += 1
                for key, desc in binds:
                    if y >= h - 2:
                        break
                    scr.addnstr(y, insp_x + 1, "%s: %s" % (key, desc),
                                insp_w - 2, cl["accent"])
                    y += 1

        # status line: spinner while busy, last message otherwise
        if self.busy:
            spin = SPIN[int(time.time() * 10) % len(SPIN)]
            scr.addnstr(h - 2, 0, ("%s %s" % (spin, self.busy))[:w - 1], w - 1, cl["warn"])
        elif self.msg:
            scr.addnstr(h - 2, 0, self.msg[:w - 1], w - 1, curses.A_BOLD)
        # bindings are drawn in the right pane; keep the bottom row clean
        scr.addnstr(h - 1, 0, " " * (w - 1), w - 1)
        scr.refresh()

    # ---------- actions ----------

    def run_detect(self, scr, p):
        keys = [k["key"] for k in p.get("keys") or [] if k.get("enabled", True)]
        if not keys:
            self.msg = "no enabled keys"
            return

        self.busy = "detecting flavor..."
        self.draw(scr)
        res = D.discover(p["base_url"], keys[0],
                         flavor=None if p.get("flavor") == "unknown" else p.get("flavor"))
        self.busy = None
        p["flavor"] = res["flavor"]
        p["base_url"] = res["base_url"]
        p["models"] = C.merge_models(p.get("models"), res["models"])
        self.save()
        if res["flavor"] == "unknown":
            # a failure explanation is worth a modal; it scrolls off the status line
            self.show_error(scr, "detection failed: %s" % p["name"], res["note"])
            self.msg = "%s: detection failed — %s" % (p["name"], res["note"])
        else:
            self.msg = "%s: %s (%s)" % (p["name"], res["note"], res["flavor"])

    def show_error(self, scr, title, body):
        """Wrap and display a failure reason; any key dismisses it."""
        h, w = scr.getmaxyx()
        bw = min(w - 4, 72)
        lines = textwrap.wrap(body, bw - 4) or ["(no detail)"]
        hint = "any key to close  ·  check the url, then the key"
        bh = len(lines) + 5
        top, left = max(0, (h - bh) // 2), max(0, (w - bw) // 2)
        win = curses.newwin(bh, bw, top, left)
        round_box(win, title)
        for i, line in enumerate(lines):
            win.addnstr(2 + i, 2, line, bw - 4)
        win.addnstr(bh - 2, 2, hint, bw - 4, curses.A_DIM)
        win.refresh()
        win.getch()

    def add_provider(self, scr):
        name = self.prompt(scr, "name: ")
        if not name:
            return
        if C.find(self.cfg, name):
            self.msg = "name already exists"
            return
        url = C.clean_url(self.prompt(scr, "base url (e.g. https://x.com/v1): "))
        if not url:
            return
        keys = []
        while True:
            k = C.clean_key(self.prompt(scr, "api key %d (blank to finish): " % (len(keys) + 1)))
            if not k:
                break
            if k in keys:
                self.msg = "that key is already in the list"
                continue
            keys.append(k)
        if not keys:
            self.msg = "need at least one key"
            return
        rot = self.prompt(scr, "rotation [F]ill-first / [r]ound-robin: ").lower()
        rotation = "round_robin" if rot.startswith("r") else "fill_first"
        p = C.new_provider(name, url, keys, rotation)
        self.cfg["providers"].append(p)
        self.save()
        self.sel = len(self.provs()) - 1
        self.run_detect(scr, self.cur())

    def edit_provider(self, scr):
        p = self.cur()
        if not p:
            return
        url = self.prompt(scr, "base url [%s]: " % p["base_url"])
        if url:
            p["base_url"] = C.clean_url(url)
        rot = self.prompt(scr, "rotation [f]ill-first/[r]ound-robin [%s]: " % p["rotation"]).lower()
        if rot.startswith("r"):
            p["rotation"] = "round_robin"
        elif rot.startswith("f"):
            p["rotation"] = "fill_first"
        fl = self.prompt(scr, "flavor openai/anthropic/auto [%s]: " % p["flavor"]).lower()
        if fl.startswith("o"):
            p["flavor"] = "openai"
        elif fl.startswith("an"):
            p["flavor"] = "anthropic"
        elif fl.startswith("au"):
            p["flavor"] = "unknown"
        self.save()
        self.msg = "saved"

    def edit_keys(self, scr):
        p = self.cur()
        if not p:
            return
        while True:
            labels = ", ".join("%d:%s%s" % (i + 1, k.get("label"), "" if k.get("enabled", True) else "(off)")
                               for i, k in enumerate(p.get("keys") or []))
            cmd = self.prompt(scr, "keys [%s] — [a]dd [d]el N [t]oggle N [q]uit: " % labels)
            if not cmd or cmd.startswith("q"):
                break
            parts = cmd.split()
            op = parts[0][0].lower()
            if op == "a":
                k = C.clean_key(self.prompt(scr, "new key: "))
                if k:
                    p["keys"].append({"key": k, "label": "k%d" % (len(p["keys"]) + 1), "enabled": True})
            elif op in ("d", "t") and len(parts) > 1 and parts[1].isdigit():
                i = int(parts[1]) - 1
                if 0 <= i < len(p["keys"]):
                    if op == "d":
                        p["keys"].pop(i)
                    else:
                        p["keys"][i]["enabled"] = not p["keys"][i].get("enabled", True)
            self.save()

    def inspect(self, scr):
        """Probe the highlighted model: reasoning, vision, availability."""
        p = self.cur()
        if not p:
            return
        mid, meta = self.cur_model()
        if not mid:
            self.msg = "no model selected (tab to the right pane)"
            return
        keys = [k["key"] for k in p.get("keys") or [] if k.get("enabled", True)]
        if not keys:
            self.msg = "no enabled keys"
            return

        def prog(stage):
            self.busy = "inspecting %s — %s ..." % (mid[:32], stage)
            self.draw(scr)

        res = D.inspect_model(p["base_url"], mid, keys[0], p.get("flavor", "openai"),
                              listing_item=D.stated_facts(meta),
                              progress=prog,
                              ask=lambda q: self.confirm(scr, q))
        self.busy = None

        entry = D.merge_inspection(meta, res)
        p.setdefault("models", {})[mid] = entry
        entry["checked"] = _now()
        self.save()
        self.show_report(scr, mid, res, entry)

    def show_report(self, scr, mid, res, entry=None):
        """Centered box with the inspection result; any key dismisses it."""
        entry = entry or {}
        ctx = entry.get("context")
        if ctx is None:
            ctx_s = "not set"
        else:
            src = entry.get("context_source")
            ctx_s = "%s tokens (%s)" % (_ctx(ctx), "listing" if src == "listing" else "manual")
        lines = [
            ("model", mid),
            ("context", ctx_s),
            ("reasoning", "%s — %s" % (_cap(res["reasoning"]), res["reasoning_note"])),
            ("vision", "%s — %s" % (_cap(res["vision"]), res["vision_note"])),
            ("available", "%s — %s" % (_cap(res.get("available")), res.get("available_note", ""))),
        ]
        h, w = scr.getmaxyx()
        bw = min(w - 4, max(46, max(len("%-10s %s" % kv) for kv in lines) + 4))
        bh = len(lines) + 4
        top, left = max(0, (h - bh) // 2), max(0, (w - bw) // 2)
        win = curses.newwin(bh, bw, top, left)
        round_box(win, " inspection ")
        for i, (k, v) in enumerate(lines):
            if k in ("reasoning", "vision", "available"):
                cap, note = v.split(" — ", 1)
                g, ga = _glyph({"yes": True, "no": False, "-": None}[cap], self.cl)
                win.addnstr(1 + i, 2, "%-10s" % k, 10, curses.A_DIM)
                win.addnstr(1 + i, 12, g, 1, ga)
                win.addnstr(1 + i, 14, note, bw - 16)
            else:
                win.addnstr(1 + i, 2, "%-10s %s" % (k, v), bw - 4)
        win.addnstr(bh - 2, 2, "any key to close", bw - 4, curses.A_DIM)
        win.refresh()
        win.getch()
        self.msg = "%s: reasoning=%s vision=%s avail=%s" % (
            mid, _cap(res["reasoning"]), _cap(res["vision"]), _cap(res.get("available")))

    def add_model(self, scr):
        p = self.cur()
        if not p:
            return
        mid = self.prompt(scr, "model id: ")
        if not mid:
            return
        p.setdefault("models", {})[mid] = {"source": "manual"}
        self.save()
        self.msg = "added %s" % mid

    def set_context(self, scr):
        """Set/clear the manual context window for the highlighted model."""
        p = self.cur()
        if not p:
            return
        mid, meta = self.cur_model()
        if not mid:
            self.msg = "no model selected (tab to the right pane)"
            return
        meta = meta or {}
        cur = meta.get("context")
        cur_s = _ctx(cur) if cur is not None else "unset"
        val = self.prompt(scr, "context for %s [%s] (e.g. 128000, 128k; blank=clear): "
                          % (mid[:20], cur_s))
        val = val.strip().lower().replace("_", "")
        if not val:
            # blank = clear the override
            if cur is not None:
                meta.pop("context", None)
                meta.pop("context_source", None)
                self.save()
                self.msg = "%s: context cleared" % mid
            else:
                self.msg = "%s: context was already unset" % mid
            return
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
        if n is None or n <= 0:
            self.msg = "invalid context size: %r" % val
            return
        p.setdefault("models", {})[mid] = meta or {}
        meta = p["models"][mid]
        meta["context"] = n
        meta["context_source"] = "manual"
        self.save()
        self.msg = "%s: context set to %s tokens" % (mid, _ctx(n))

    def reset_model(self, scr):
        """Clear probed capability verdicts and any manual context override.

        reasoning/vision/availability and their notes go back to '-'; a
        manually-set context is dropped too. A context that came from the
        provider's listing survives — that's provider data, not a probe
        result or a user choice.
        """
        p = self.cur()
        if not p:
            return
        mid, meta = self.cur_model()
        if not mid:
            self.msg = "no model selected (tab to the right pane)"
            return
        if not meta or not any(k in meta for k in
                                ("reasoning", "vision", "available", "checked",
                                 "context")):
            self.msg = "%s: nothing probed to reset" % mid
            return
        if not self.confirm(scr, "reset probed values for %s?" % mid[:24]):
            return
        for k in ("reasoning", "reasoning_note", "vision", "vision_note",
                  "available", "available_note", "checked"):
            meta.pop(k, None)
        if meta.get("context_source") == "manual":
            meta.pop("context", None)
            meta.pop("context_source", None)
        self.save()
        self.msg = "%s: probed values reset" % mid

    def toggle_model(self, scr):
        """Turn the highlighted model on/off at the gateway level."""
        p = self.cur()
        if not p:
            return
        mid, meta = self.cur_model()
        if not mid:
            self.msg = "no model selected (shift-tab or left-arrow to providers)"
            return
        meta = p.setdefault("models", {}).setdefault(mid, {})
        cur = meta.get("enabled", True)
        meta["enabled"] = not cur
        self.save()
        self.msg = "%s/%s: %s" % (p["name"], mid[:28],
                                  "enabled" if meta["enabled"] else "disabled")

    def toggle_all_models(self, scr):
        """Toggle every model of the selected provider on or off."""
        p = self.cur()
        if not p:
            return
        models = p.get("models") or {}
        if not models:
            self.msg = "%s has no models yet" % p["name"]
            return
        any_off = any(not m.get("enabled", True) for m in models.values())
        # if anything is off, go all-on; if everything is on, go all-off
        on = any_off
        q = "enable all %d models of %s?" if on else "disable all %d models of %s?"
        if not self.confirm(scr, q % (len(models), p["name"])):
            return
        for m in models.values():
            m["enabled"] = on
        self.save()
        self.msg = "%s: %d models %s" % (p["name"], len(models),
                                         "enabled" if on else "disabled")

    def open_logs(self, scr):
        """'l' — spawn a new terminal window showing the live gateway log."""
        repo = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        script = os.path.join(repo, "scripts", "logwatch.sh")
        if not os.path.exists(script):
            self.msg = "logwatch not found: %s" % script
            return
        for term in ("ghostty", "kitty", "konsole", "foot", "alacritty", "xterm"):
            exe = shutil.which(term)
            if not exe:
                continue
            try:
                subprocess.Popen([exe, "-e", "bash", script],
                                 start_new_session=True,
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                self.msg = "opened log terminal (%s)" % term
                return
            except OSError:
                continue
        self.msg = "no terminal emulator found (install ghostty/kitty/konsole)"

    def show_help(self, scr):
        """Full keymap overlay (the bottom bar shows only the active pane's)."""
        h, w = scr.getmaxyx()
        rows = [
            ("left pane", "j/k  move · a:add e:edit d:del K:keys"),
            ("left pane", "r:refresh t:on/off T:all-models"),
            ("right", "j/k  move · ENTER:inspect t:on/off T:all-models"),
            ("right", "A:avail-all (one call per model) · c:context"),
            ("right", "x:reset m:add /:filter"),
            ("both", "TAB/←→  switch pane · R:revive keys · ?:this help"),
            ("both", "l:live-log terminal · q or ESC  quit"),
        ]
        bw = min(w - 4, 62)
        bh = len(rows) + 4
        top, left = max(0, (h - bh) // 2), max(0, (w - bw) // 2)
        win = curses.newwin(bh, bw, top, left)
        round_box(win, " keys ")
        for i, (scope, txt) in enumerate(rows):
            win.addnstr(1 + i, 2, "%-6s" % scope, 6, curses.A_DIM)
            win.addnstr(1 + i, 8, txt, bw - 10)
        win.addnstr(bh - 2, 2, "any key to close", bw - 4, curses.A_DIM)
        win.refresh()
        win.getch()

    def probe_all_avail(self, scr):
        """Availability-only pass over every model of the current provider.

        One minimal chat call (~1 token) per model, no reasoning/vision
        probing: it refreshes the avail column + note for the whole list
        in seconds, so dead or misrouted models show up at a glance.
        """
        p = self.cur()
        if not p:
            return
        models = p.get("models") or {}
        if not models:
            self.msg = "%s has no models yet" % p["name"]
            return
        keys = [k["key"] for k in p.get("keys") or [] if k.get("enabled", True)]
        if not keys:
            self.msg = "no enabled keys"
            return
        key, flavor = keys[0], p.get("flavor", "openai")
        total, ok, dead = len(models), 0, 0
        self.busy = "availability 0/%d" % total
        self.draw(scr)
        # NOTE: save() reloads self.cfg, which would replace the dict we're
        # iterating — mutate in place and persist exactly once at the end.
        for i, (mid, meta) in enumerate(sorted(models.items())):
            self.busy = "availability %d/%d — %s" % (i + 1, total, mid[:40])
            self.draw(scr)
            try:
                avail, note = D.probe_availability(p["base_url"], mid, key, flavor)
            except Exception as e:  # noqa: BLE001 — a probe must never kill the pass
                avail, note = None, "probe crashed: %s" % e
            meta["available"] = avail
            meta["available_note"] = note
            meta["checked"] = _now()
            if avail is True:
                ok += 1
            elif avail is False:
                dead += 1
        self.save()
        self.busy = None
        self.msg = "%s: %d ok, %d dead, %d inconclusive (%d models)" % (
            p["name"], ok, dead, total - ok - dead, total)

    def revive(self, scr):
        """Ask the running server to clear dead/cooldown key state.

        Key health lives in the server process, so pressing this in the TUI
        has to go over HTTP; there was previously no way at all to un-kill a
        key that a transient 401/403 had marked dead.
        """
        listen = self.cfg.get("listen") or {}
        base = "http://%s:%s" % (listen.get("host", "127.0.0.1"), listen.get("port", C.DEFAULT_PORT))
        p = self.cur()
        body = {"provider": p["name"]} if p else {}
        r = H.post(base + "/v1/revive", body, timeout=3)
        if not r.ok:
            self.msg = "server not running (%s)" % (r.status or r.error)
            return
        data = r.json() or {}
        who = data.get("provider", "all")
        if not data.get("revived"):
            self.msg = "server has no key state for %s yet" % who
        else:
            self.msg = "revived keys: %s" % who

    # ---------- main loop ----------

    def loop(self, scr):
        curses.curs_set(0)
        scr.keypad(True)
        self._init_colors()
        self._healthz()  # prime the cached gateway liveness before first draw
        while True:
            self.cfg = self._load_or_keep(self.cfg) if not self.busy else self.cfg
            self.draw(scr)
            # animate the braille spinner while a probe runs; block otherwise
            scr.timeout(150 if self.busy else -1)
            try:
                ch = scr.getch()
            except KeyboardInterrupt:
                break
            if ch == -1:  # timer tick, no key: just redraw the spinner
                continue
            self.msg = ""
            n = len(self.provs())
            if ch in (ord("q"), 27):
                break
            elif ch in (ord("j"), curses.KEY_DOWN):
                if self.focus == "left":
                    self.sel = min(self.sel + 1, max(0, n - 1))
                    self.msel = 0
                    self.mtop = 0
                else:
                    self.msel = min(self.msel + 1, max(0, len(self.models()) - 1))
            elif ch in (ord("k"), curses.KEY_UP):
                if self.focus == "left":
                    self.sel = max(0, self.sel - 1)
                    self.msel = 0
                    self.mtop = 0
                else:
                    self.msel = max(0, self.msel - 1)
            elif ch in (curses.KEY_NPAGE, ord(" ")) and self.focus == "right":
                self.msel = min(self.msel + 15, max(0, len(self.models()) - 1))
            elif ch == curses.KEY_PPAGE and self.focus == "right":
                self.msel = max(0, self.msel - 15)
            elif ch == curses.KEY_HOME and self.focus == "right":
                self.msel = 0
            elif ch == curses.KEY_END and self.focus == "right":
                self.msel = max(0, len(self.models()) - 1)
            elif ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                self.focus = "left" if ch == curses.KEY_LEFT else "right"
            elif ch in (curses.KEY_ENTER, 10, 13):
                if self.focus == "left":
                    self.focus = "right"
                else:
                    self.inspect(scr)
            elif ch == ord("/"):
                self.focus = "right"
                self.filter = self.prompt(scr, "filter models: ")
                self.msel = 0
                self.mtop = 0
            elif ch == ord("\t"):
                self.focus = "right" if self.focus == "left" else "left"
            elif ch == ord("a"):
                self.add_provider(scr)
            elif ch == ord("e"):
                self.edit_provider(scr)
            elif ch == ord("K"):
                self.edit_keys(scr)
            elif ch == ord("m"):
                self.add_model(scr)
            elif ch == ord("c"):
                self.set_context(scr)
            elif ch == ord("x"):
                self.reset_model(scr)
            elif ch == ord("r"):
                p = self.cur()
                if p:
                    self.run_detect(scr, p)
            elif ch == ord("t"):
                if self.focus == "left":
                    p = self.cur()
                    if p:
                        p["enabled"] = not p.get("enabled", True)
                        self.save()
                else:
                    self.toggle_model(scr)
            elif ch == ord("T"):
                self.toggle_all_models(scr)
            elif ch == ord("A"):
                self.probe_all_avail(scr)
            elif ch == ord("R"):
                self.revive(scr)
            elif ch == ord("l"):
                self.open_logs(scr)
            elif ch == ord("?"):
                self.show_help(scr)
            elif ch == ord("d"):
                p = self.cur()
                if p and self.confirm(scr, "delete %s?" % p["name"]):
                    self.cfg["providers"].remove(p)
                    self.save()
                    self.sel = max(0, self.sel - 1)


def main(path=None):
    curses.wrapper(Tui(path).loop)
