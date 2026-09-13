#!/usr/bin/env python3
"""Headless TUI smoke test: drives the curses UI inside a pty and asserts
the panes render, keys don't crash it, and narrow terminals survive.

Not a full UI test — catches crashes, layout errors, and dead keys.

    python3 tests/tui_smoke.py
"""

import os
import pty
import select
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cgw import config as C  # noqa: E402

TRACEBACK = "Traceback (most recent call last)"


def drive(path, keys, settle=0.6, cols=110, lines=50):
    """Run the TUI in a pty, send keys, return everything it drew."""
    env = dict(os.environ, TERM="xterm-256color", LINES=str(lines), COLUMNS=str(cols))
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(sys.executable, [sys.executable, os.path.join(ROOT, "cg"),
                                    "--config", path, "tui"], env)
    out = b""
    time.sleep(settle)
    for k in keys:
        os.write(fd, k)
        time.sleep(0.25)
        while select.select([fd], [], [], 0.15)[0]:
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
    try:
        os.write(fd, b"q")
    except OSError:
        pass
    deadline = time.time() + 3
    while time.time() < deadline:
        r = select.select([fd], [], [], 0.2)[0]
        if not r:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return out.decode("utf-8", "replace")


# Every binding the right pane advertises. A key that raises takes the whole
# UI down, so each one gets pressed against a config that actually has a
# provider and models to act on.
KEY_SWEEP = [
    ("help", [b"?", b"q"]),
    ("filter", [b"/", b"m", b"\r", b"q"]),
    ("filter-esc", [b"/", b"zz", b"\x1b", b"q"]),
    ("logs", [b"l", b"\x1b", b"q"]),
    ("usage", [b"u", b"\r", b"\x1b", b"q"]),
    ("toggle model", [b" ", b"q"]),
    ("toggle all", [b"a", b"q"]),
    ("add model", [b"m", b"\x1b", b"q"]),
    ("context", [b"c", b"\x1b", b"q"]),
    ("reset model", [b"x", b"\x1b", b"q"]),
    ("backup", [b"b", b"q"]),
    ("restore", [b"B", b"\x1b", b"q"]),
    ("keys", [b"K", b"\x1b", b"q"]),
    ("new provider", [b"a", b"\x1b", b"\x1b", b"q"]),
    ("revive", [b"R", b"q"]),
    ("detect", [b"r", b"\x1b", b"q"]),
    ("inspect", [b"\r", b"\x1b", b"q"]),
    ("restart (declined)", [b"S", b"n", b"q"]),
    ("tab", [b"\t", b"\t", b"q"]),
    ("arrows", [b"\x1b[B", b"\x1b[B", b"\x1b[A", b"q"]),
    ("page keys", [b"\x1b[6~", b"\x1b[5~", b"q"]),
]


def main():
    if not shutil.which("infocmp"):
        print("SKIP: no terminfo available")
        return 0

    d = tempfile.mkdtemp()
    path = os.path.join(d, "config.json")
    cfg = C.default_config()
    p = C.new_provider("demo", "http://127.0.0.1:1/v1", ["aaa", "bbb"], "round_robin", "openai")
    p["models"] = {"model-one": {"reasoning": True, "vision": False},
                   "model-two": {}}
    cfg["providers"].append(p)
    C.save(cfg, path)

    screen = drive(path, [b"\t", b"\x1b[B", b"\t"])

    checks = [
        ("title", "CG"),
        ("provider name", "demo"),
        ("flavor row", "openai"),
        ("rotation row", "round-robin"),
        ("key list", "k1"),
        ("model listing", "model-one"),
        ("capability column", "rsn"),
        ("capability value", "✓"),
        ("bindings (right pane)", "a: add"),
        ("bindings (right pane)", "b: backup"),
        ("bindings (right pane)", "S: restart"),
        ("bindings (right pane)", "ENTER: inspect"),
        ("bindings (right pane)", "c: context"),
        ("bindings (right pane)", "l: logs"),
    ]
    failed = [name for name, needle in checks if needle not in screen]
    for name, needle in checks:
        print("%-16s %s" % ("ok" if needle in screen else "MISSING", name))

    # a provider added by hand must survive a restart of the TUI
    reloaded = C.load(path)
    assert reloaded["providers"][0]["rotation"] == "round_robin"

    # Narrow / short viewports used to kill the UI outright: a column header
    # placed past the right edge raises _curses.error out of draw(). Every
    # size must render something and exit cleanly.
    for label, cols, lines in (("narrow", 46, 12), ("tiny", 30, 8),
                               ("wide-short", 200, 8), ("tall-thin", 24, 60)):
        text = drive(path, [], cols=cols, lines=lines)
        broke = TRACEBACK in text
        print("%-16s %s (%dx%d)" % ("ok" if not broke else "CRASHED", label, cols, lines))
        if broke:
            failed.append(label)
            print(text[text.index(TRACEBACK):][:900])

    # every advertised binding, driven for real
    for label, keys in KEY_SWEEP:
        text = drive(path, keys)
        broke = TRACEBACK in text
        print("%-16s %s" % ("ok" if not broke else "CRASHED", "key: " + label))
        if broke:
            failed.append("key:" + label)
            print(text[text.index(TRACEBACK):][:900])

    if failed:
        print("\nFAILED: %s" % ", ".join(failed))
        print("--- captured screen ---")
        print(screen[-3000:])
        return 1
    print("\nTUI smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
