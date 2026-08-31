#!/usr/bin/env bash
# CG — AI TUI Gateway installer for Linux / macOS.
# Downloads the latest release tarball, installs to ~/.local/share/cg-tui-gateway
# and symlinks the `cg` command into ~/.local/bin.
#
# Overrides (for testing/mirrors):
#   CG_INSTALL_BASE   base URL of the release assets (default: GitHub releases/latest)
#   CG_INSTALL_DIR    install directory        (default: ~/.local/share/cg-tui-gateway)
#   CG_INSTALL_BIN    where to link `cg`       (default: ~/.local/bin)
set -euo pipefail

REPO=MonarchCoeus/cg-tui-gateway
BASE="${CG_INSTALL_BASE:-https://github.com/${REPO}/releases/latest/download}"
DEST="${CG_INSTALL_DIR:-$HOME/.local/share/cg-tui-gateway}"
BIN="${CG_INSTALL_BIN:-$HOME/.local/bin}"

echo "==> CG — AI TUI Gateway installer"

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 (3.9 or newer) is required — install it first:"
    echo "  Linux:  apt install python3 / pacman -S python / dnf install python3"
    echo "  macOS:  brew install python  (or https://python.org)"
    exit 1
fi
if [ "$(python3 -c 'import sys; print(1 if sys.version_info[:2] >= (3, 9) else 0)')" != "1" ]; then
    echo "error: python 3.9+ required (found $(python3 -V))"
    exit 1
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
echo "==> downloading $BASE/cg-tui-gateway.tar.gz"
TARBALL="$WORK/cg-tui-gateway.tar.gz"
if ! curl -fsSL "$BASE/cg-tui-gateway.tar.gz" -o "$TARBALL"; then
    if command -v wget >/dev/null 2>&1; then
        wget -q "$BASE/cg-tui-gateway.tar.gz" -O "$TARBALL"
    else
        echo "error: download failed (need curl or wget)"
        exit 1
    fi
fi
echo "==> extracting"
tar -xzf "$TARBALL" -C "$WORK"

mkdir -p "$DEST" "$BIN"
cp -R "$WORK"/cg-tui-gateway/. "$DEST"/
ln -sf "$DEST/cg" "$BIN/cg"

echo "==> installed to $DEST (command: $BIN/cg)"
if case ":$PATH:" in *":$BIN:"*) false ;; *) true ;; esac; then
    echo "==> add to your PATH once (then 'cg' works everywhere):"
    echo "    export PATH=\"$BIN:\$PATH\""
else
    echo "==> run with:  cg tui   (or:  cg serve)"
fi
