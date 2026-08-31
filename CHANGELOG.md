# Changelog

All notable changes to CG — AI TUI Gateway. Format follows [Keep a Changelog](https://keepachangelog.com/), versions follow [SemVer](https://semver.org/).

## [1.0.0] — 2026-08-31

Initial public release.

### Added

- Local OpenAI-compatible gateway (stdlib-only Python, nothing to install)
- `cg` CLI: `add` / `list` / `detect` / `inspect` / `revive` / `toggle` / `context` / `reset` / `serve` / `status` / `tui`
- 3-pane curses TUI: providers · models · inspection, with full keymap, filter, and live-log terminal (`l`)
- Key rotation per provider: fill-first and round-robin, with retry-across-keys, exponential backoff, and permanent dead-key tracking (401/403)
- Capability probing on demand: availability, reasoning (evidence-based), vision (token-differential); results cached with provenance
- Context sizes read from provider listings (`source: listing`) or manual override (`source: manual`); never silently guessed
- OpenAI ⇄ Anthropic translation for Anthropic-native providers (system hoisting, `stop_sequences`, thinking mapping), incl. SSE streaming
- HTTP API: `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/logs`, `/healthz`, `/v1/revive`
- Live per-request log endpoint `/v1/logs` (model, key label, status, latency)
- OSC-11-aware selection tint: the TUI reads the terminal's real background so the selection bar matches the accent in light and dark themes
- Cross-OS one-line installers (`install.sh`, `install.ps1`) + GitHub release workflow
- Tests: 145 unit tests against a local fake upstream (no network) + headless TUI smoke; CI on Linux/macOS/Windows

### Security

- Config `~/.config/cg/config.json` mode 0600; keys stored in plaintext by design (documented)
- Binds `127.0.0.1` only; no inbound auth (documented) — do not expose publicly without adding auth
