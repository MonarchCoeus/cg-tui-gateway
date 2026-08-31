#!/usr/bin/env python3
"""Regression test: the TUI prompt must never truncate what was typed.

The original bug: curses.getstr() echoes into the window and stops at the
right margin, so a 50-char API key entered after a 29-char label on an
80-column terminal came back 49 chars long. A silently shortened key is
indistinguishable from a rejected one, which cost an hour of blaming a
provider for a UI bug.

This drives the real prompt() inside a real pty across a range of terminal
widths and label lengths, including values far wider than the screen, and
asserts an exact round-trip every time.

    python3 tests/prompt_widths.py
"""

import os
import pty
import select
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Values chosen to straddle the old failure boundary: 48/50 are real key
# lengths, 200 is far past any terminal width, and the empty string checks
# that "blank to finish" still works.
CASES = [
    ("sk-nry-gv7Qfs13o4PVDyCra8SRo7l_JcQtlH9W6Ucl0Od9xmk", "api key 1 (blank to finish): "),
    ("dahl_7BLEwLGBW53fGUnpDdQw7g8LTXYBU8Taa", "api key 1 (blank to finish): "),
    ("x" * 200, "api key 1 (blank to finish): "),
    ("sk-" + "a" * 60, "base url (e.g. https://x.com/v1): "),
    ("short", "name: "),
]
WIDTHS = (40, 60, 80, 100, 120, 200)

CHILD = r'''
import curses, os, sys
sys.path.insert(0, %(root)r)
from cgw.tui import Tui

def run(scr):
    scr.keypad(True)
    return Tui.prompt(Tui.__new__(Tui), scr, %(label)r)

got = curses.wrapper(run)
# marker-delimited so pty echo and redraws can't be mistaken for the result
sys.stdout.write("\n<<<%%d|%%s>>>\n" %% (len(got), got))
sys.stdout.flush()
'''


def run_case(value, label, width):
    """Type `value` into a real prompt at `width` columns; return what it got."""
    src = CHILD % {"root": ROOT, "label": label}
    env = dict(os.environ, TERM="xterm-256color", LINES="24", COLUMNS=str(width))
    pid, fd = pty.fork()
    if pid == 0:
        os.execvpe(sys.executable, [sys.executable, "-c", src], env)

    # size the pty itself: COLUMNS alone does not resize a tty
    import fcntl
    import struct
    import termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 24, width, 0, 0))

    time.sleep(0.5)
    os.write(fd, value.encode() + b"\n")

    out = b""
    deadline = time.time() + 6
    while time.time() < deadline:
        if not select.select([fd], [], [], 0.2)[0]:
            if b">>>" in out:
                break
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        out += chunk
        if b">>>" in out:
            break
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass

    text = out.decode("utf-8", "replace")
    start = text.rfind("<<<")
    end = text.rfind(">>>")
    if start < 0 or end < 0:
        return None, text
    payload = text[start + 3:end]
    n, _, got = payload.partition("|")
    try:
        return (int(n), got)
    except ValueError:
        return None, text


def main():
    if not shutil.which("infocmp"):
        print("SKIP: no terminfo available")
        return 0

    failures = []
    for value, label in CASES:
        for width in WIDTHS:
            room = width - len(label) - 2
            n, got = run_case(value, label, width)
            if n is None:
                failures.append((value, label, width, "no result captured"))
                print("  ERROR  w=%-4d room=%-4d len=%-4d %s" %
                      (width, room, len(value), label.strip()))
                continue
            ok = got == value and n == len(value)
            if not ok:
                failures.append((value, label, width,
                                 "sent %d got %d" % (len(value), n)))
            print("  %-6s w=%-4d room=%-4d sent=%-4d got=%-4d %s" %
                  ("ok" if ok else "FAIL", width, room, len(value), n,
                   "" if ok else "LOST %r" % value[n:]))

    print("")
    if failures:
        print("FAILED %d/%d cases" % (len(failures), len(CASES) * len(WIDTHS)))
        for value, label, width, why in failures:
            print("  w=%d label=%r len=%d: %s" % (width, label, len(value), why))
        return 1
    print("prompt round-trips exactly at every width (%d cases)"
          % (len(CASES) * len(WIDTHS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
