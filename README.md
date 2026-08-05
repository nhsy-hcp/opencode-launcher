# opencode-launcher

[![CI](https://github.com/nhsy-hcp/opencode-launcher/actions/workflows/ci.yml/badge.svg)](https://github.com/nhsy-hcp/opencode-launcher/actions/workflows/ci.yml)

Interactive TUI to pick a model and launch [opencode](https://opencode.ai).

Fetches available models from enabled providers in parallel, lets you browse/filter them, writes the chosen model into `~/.config/opencode/opencode.json`, and exec's `opencode --model <provider>/<model>`.

## Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/)
- [`opencode`](https://opencode.ai/docs/) in `$PATH`

## Usage

```bash
uv run opencode-launcher.py                          # interactive picker
uv run opencode-launcher.py --model litellm/gpt-4o  # skip picker, launch directly
uv run opencode-launcher.py --dry-run                # print command, don't execute
uv run opencode-launcher.py --no-cache               # bypass cache, fetch fresh models
uv run opencode-launcher.py --clear-history          # wipe history & cache, then exit
uv run opencode-launcher.py --clear-cache            # wipe model cache, then exit
uv run opencode-launcher.py --litellm-url URL        # override LiteLLM base URL
uv run opencode-launcher.py --timeout 30             # request timeout in seconds
```

Extra arguments after `--` are forwarded to `opencode`:

```bash
uv run opencode-launcher.py -- --workspace /my/project
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LITELLM_URL` | `http://localhost:4000` | LiteLLM proxy base URL |
| `LITELLM_MASTER_KEY` | `litellm` | LiteLLM API key |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `VLLM_URL` | `http://localhost:8000` | vLLM base URL |
| `NVIDIA_API_KEY` | — | Required when NVIDIA provider is enabled |
| `OPENROUTER_API_KEY` | — | Required when OpenRouter provider is enabled |
| `ICA_API_KEY` | — | Required when ICA provider is enabled |
| `REQUEST_TIMEOUT` | `10` | HTTP request timeout (seconds) |

## Providers

Toggle providers by editing the `PROVIDERS` dict at the top of the script:

```python
PROVIDERS = {
    "litellm": True,
    "ollama": False,
    "nvidia": False,
    "openrouter": False,
    "vllm": False,
    "ica": True,
}
```

## Development

```bash
# create venv and install dev deps
uv venv .venv
uv pip install pytest pytest-cov ruff

# run tests
task test

# run tests with coverage (fail under 80%)
task test:coverage

# lint
task lint

# full CI pipeline
task test:ci
```

## Cache

Models are cached for 24 hours in `~/.config/opencode-launcher/cache.json`. Recent selections are stored in `~/.config/opencode-launcher/history.json` (last 10 entries).
