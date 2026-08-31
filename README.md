# CG — Coeus Gateway

A local LLM gateway in plain Python. No pip installs, no database, no cloud.

Give it a provider name, a base URL, and one or more API keys. It works out the
rest and serves everything to your clients as one OpenAI-compatible endpoint.

    ./cg add myprovider https://example.com/v1 sk-key1 sk-key2
    ./cg serve

Point Hermes (or anything OpenAI-compatible) at `http://127.0.0.1:20185/v1`.

## Install

One-liner for newbies — the latest release tarball, no git needed:

    curl -fsSL https://github.com/MonarchCoeus/cg-tui-gateway/releases/latest/download/install.sh | bash

Windows (PowerShell):

    irm https://github.com/MonarchCoeus/cg-tui-gateway/releases/latest/download/install.ps1 | iex

The installers put CG under `~/.local/share/cg-tui-gateway` (Linux/macOS) or
`%LOCALAPPDATA%\cg-tui-gateway` (Windows), symlink/link a `cg` command on your
PATH, and check your Python (3.9+ required). Nothing else to install: no pip
packages, no database, no cloud.

Developers can also clone the repo itself:

    git clone https://github.com/MonarchCoeus/cg-tui-gateway.git
    cd cg-tui-gateway
    ./cg                    # launches the TUI (or: ./cg serve)

## Quick start

    ./cg add myprovider https://example.com/v1 sk-key1 sk-key2 --round-robin
    ./cg serve

Then point any OpenAI-compatible client at `http://127.0.0.1:20185/v1` and use
models as `myprovider/modelname`. The first `add` already detects the
provider's models and whether it's OpenAI- or Anthropic-shaped.

## Why

LiteLLM wants Postgres and migrations. OmniRoute and 9router guess context
sizes and then serve the guess as fact. CG does one job: rotate keys across
providers and tell the truth about what it knows — and **not** pretend to know
what it doesn't.

## Requirements

Python 3.9+ standard library. That's it. Nothing to install.

## Commands

    ./cg add NAME URL KEY [KEY...]   add a provider and detect its models
        --round-robin                rotate keys evenly (default: fill-first)
    ./cg list                        show providers and models
    ./cg detect NAME                 re-run detection on an existing provider
    ./cg inspect NAME MODEL          probe a model (avail / reasoning / vision)
    ./cg revive [NAME]               clear dead/cooldown key state live
    ./cg toggle NAME                 enable/disable a provider
    ./cg context NAME MODEL SIZE     set/clear a model's context window
    ./cg reset NAME MODEL            discard a model's probed verdicts
    ./cg serve [--host H] [--port N] run the gateway
    ./cg status                      key health from a running gateway
    ./cg tui                         interactive manager (also the default)

Every command takes `--config PATH` if you want a config other than
`~/.config/cg/config.json`.

## The TUI

    ./cg tui

Three panes: providers on the left, that provider's models in the middle, and
an inspection pane on the right (provider/model details plus the full keymap).
`TAB` (or `←→`) switches panes. Edits are saved immediately and the running
gateway picks them up on the next request — no restart needed.

    a  add provider          x  reset a model's probed verdicts
    e  edit provider         m  add a model by hand
    d  delete provider       /  filter models
    K  edit keys             A  probe availability for all models
    r  re-detect             R  revive dead/benched keys
    t  enable/disable        l  open a live log terminal (new window)
    T  toggle all models     ?  full keymap overlay
    j/k  move      ENTER  inspect the highlighted model
    c  set/clear its context window      tab  switch pane
    q  quit

The `l` key spawns your terminal emulator running `scripts/logwatch.sh`, which
polls the gateway's `/v1/logs` (see below) every second and shows the last
requests with the key that served each one — a running proof of rotation.

The selection bar is a tint of the accent blue blended over the terminal's
real background (queried via OSC 11), so it reads as the same blue as the
keymap text in both light and dark terminals.

## How detection works

Flavor: CG asks `{url}/models` with a Bearer token. If that fails it retries
with Anthropic's `x-api-key` header. If your URL lacks `/v1`, it tries adding
it — and if you included `/v1` but the provider doesn't use it, it tries
removing it.

Models: whatever the provider's `/models` returns.

Context size: **read from the provider's listing when it volunteers one**
(`context_window`, `max_model_len`, etc.). CG never probes for it and never
defaults it: a provider that doesn't advertise a size has no context shown
(`-`) until you set it manually with `cg context NAME MODEL SIZE` or press
`c` on a model in the TUI. Whatever the provider lists is stored with source
`listing`; a manual override replaces it. The value is reported in
`/v1/models` so the client (e.g. Hermes) knows without probing. What CG
*does* verify, on demand:

- **availability** — one minimal chat call. Does this model answer at all?
- **reasoning** — a positive result needs a non-empty trace, never just a key
  being present.
- **vision** — a token-differential probe: a real vision model costs hundreds
  of extra input tokens for an image, a router that silently drops it costs
  nothing.

`ENTER` on a model (or `./cg inspect NAME MODEL`) runs all three probes and
stores the verdicts. Until then the model shows `-` for unchecked.

## Key rotation

Per provider, pick one:

- **fill-first** — hammer key 1 until it stops working, then key 2. Best for
  free tiers where you want to exhaust one quota before touching the next.
- **round-robin** — spread requests evenly across keys.

When a request fails, CG retries it on the next healthy key before returning
anything to the client, so rate limits are invisible to your client instead of
surfacing as errors.

Failure handling:

- **429 / 5xx / timeout** — key benched with exponential backoff, 60s doubling
  to a 15 minute cap. Recovers by itself. A success resets the counter.
- **401 / 403** — key marked `dead` and skipped entirely. It's not busy, it's
  wrong, so retrying wastes time. Fix or remove it; disabling and re-enabling
  in the TUI also clears the flag.

Up to 5 keys are tried per request. Key health lives in memory only, so
restarting the gateway gives every key a fresh chance.

## Model naming

Models are served as `providername/modelname`, so two providers offering the
same model don't collide. A bare model name also works if exactly one provider
offers it.

## Anthropic-native providers

If a provider's real API is Anthropic-shaped, CG translates on the way out:
system messages hoisted to the top-level `system` field, consecutive same-role
turns merged, `stop` promoted to `stop_sequences`, `max_tokens` filled in when
your client omits it, and `reasoning_effort` (low/medium/high/xhigh/max)
mapped to an Anthropic `thinking` block so reasoning levels apply there too.
Responses and SSE streams are converted back to OpenAI shape. Your client
only ever speaks OpenAI.

## Config

`~/.config/cg/config.json`, mode 0600, written atomically.

```json
{
  "version": 1,
  "listen": { "host": "127.0.0.1", "port": 20185 },
  "providers": [
    {
      "name": "myprovider",
      "base_url": "https://example.com/v1",
      "flavor": "openai",
      "keys": [ { "key": "sk-...", "label": "k1", "enabled": true } ],
      "rotation": "fill_first",
      "enabled": true,
      "models": { "some-model": { "reasoning": true, "vision": false } }
    }
  ]
}
```

Keys are stored in plaintext. That's deliberate — one file, no env-var
indirection, no secret manager. The file is 0600 and the directory is 0700.

## Security

CG has **no inbound authentication** and binds `127.0.0.1` only. Any process on
this machine can send requests through it and use your upstream keys. Do not
bind `0.0.0.0` or put it behind a public reverse proxy without adding auth
first.

## Endpoints

    GET  /v1/models            model list (includes context when known)
    POST /v1/chat/completions   streaming and non-streaming
    POST /v1/completions
    POST /v1/embeddings
    GET  /healthz               providers, key health, rotation
    GET  /v1/logs?n=20          recent per-request log (model, key, status, ms)
    POST /v1/revive             clear dead/cooldown key state

## Tests

    python3 tests/run_tests.py    145 tests against a local fake upstream
    python3 tests/tui_smoke.py    renders the TUI in a pty and checks output

No network access, no real API calls, no credits spent. The fake upstream in
`tests/fake_upstream.py` simulates providers that volunteer capability flags,
ones that report nothing, one with sibling metadata, an Anthropic-native one,
and two with broken keys.

## Layout

    cg                    CLI entry point
    cgw/config.py         JSON config load/save
    cgw/http.py           urllib wrapper (sends a curl User-Agent; some
                          providers 403 Python's default)
    cgw/detect.py         flavor, model list, capability probes
    cgw/keyring.py        rotation, cooldowns, dead-key tracking
    cgw/translate.py      OpenAI <-> Anthropic conversion
    cgw/server.py         the gateway itself
    cgw/tui.py            curses interface
    scripts/logwatch.sh   live log viewer (opened by the TUI's l key)

## License

MIT — see [LICENSE](LICENSE).
