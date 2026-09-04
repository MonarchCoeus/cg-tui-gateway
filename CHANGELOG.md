# Changelog

All notable changes to CG — AI TUI Gateway. Format follows [Keep a Changelog](https://keepachangelog.com/), versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added

- Live model search: `/` in the TUI now filters as you type (Enter keeps, ESC clears)
- Token accounting: every successful request logs tokens in/out/cached to `usage.jsonl`; new `cg stats [--window W] [--by model|provider]` table with windows 15min–30d plus session (this gateway run); `/v1/logs` entries carry the same counts
- TUI `u` key: time/session scope picker — time windows 15min–30d, or a searchable Hermes-session browser (id + first-message title column, full title in the result) with backspace to go back a level; `cg stats --session ID` does the same on the CLI
- Probe cancellation: backspace (or ESC) aborts an in-flight model inspect, discarding partial results; the availability sweep stops too, keeping whatever it finished
- F5 refresh: re-reads the config from disk and re-checks the gateway without restarting the TUI (R stays revive)
- Tool calls through /responses-only models: `tools`/`tool_choice` forwarded upstream, `function_call` items folded back to OpenAI `tool_calls` (both plain and streamed) — agentic clients no longer stall waiting for calls that never arrive
- Audit-fix batch: tool definitions, history, and streamed tool calls on the Anthropic path; vision images through /responses (`input_image`); HF vision heuristic no longer flags T5/BART-style text models; empty `modalities: []` reads unknown instead of text-only; usage math unified; streamed Responses cache hits kept; relay error statuses logged truthfully; usage-file writes serialized; vision-probe thread failures read inconclusive

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
