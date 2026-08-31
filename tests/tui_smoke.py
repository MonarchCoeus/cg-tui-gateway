#!/usr/bin/env python3
"""Headless TUI smoke test: drives the curses UI inside a pty and asserts
the panes render. Not a full UI test — catches crashes and layout errors.

    python3 tests/tui_smoke.py
"""

import os
import pty
import select
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cgw import config as C  # noqa: E402


def drive(path, keys, settle=0.6):
    """Run the TUI in a pty, send keys, return everything it drew."""
    env = dict(os.environ, TERM="xterm-256color", LINES="50", COLUMNS="110")
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

    if failed:
        print("\nFAILED: %s" % ", ".join(failed))
        print("--- captured screen ---")
        print(screen[-3000:])
        return 1
    print("\nTUI smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
