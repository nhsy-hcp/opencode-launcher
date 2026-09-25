# AGENTS.md

Developer and AI-agent reference for this project.

## Project Overview

`opencode-launcher.py` is a self-contained `uv`-runnable Python script (inline script metadata, no package build). It presents an interactive TUI for selecting an LLM provider/model and launching `opencode`.

## Runtime

| Tool | Purpose |
|---|---|
| `uv` | Dependency management and script execution |
| `ruff` | Linting and formatting (replaces black, isort, flake8) |
| `pytest` + `pytest-cov` | Test runner with coverage enforcement |
| `pre-commit` | Git hook automation |
| `task` | Task runner (`Taskfile.yml`) |

## Key Commands

```bash
task lint          # ruff check + ruff format --check
task test          # pytest (no coverage)
task test:coverage # pytest with coverage — fails if < 80%
task test:ci       # run CI workflow locally via act (requires act + Docker)
task test:ica      # live ICA smoke test: /models + /chat/completions (requires ICA_API_KEY)
task clean         # remove .pytest_cache, .coverage, .ruff_cache, .tmp
```

## Code Conventions

- **Python ≥ 3.11**, single-file script (`opencode-launcher.py`)
- Line length: **200 chars** (configured in `pyproject.toml`)
- No comments unless explicitly requested
- No new dependencies without updating the inline `# dependencies = [...]` block in the script header
- All file writes use `_atomic_write()` — never write config files directly

## Testing

- Tests live in `tests/` — `conftest.py` + `test_opencode_launcher.py`
- The script is imported via `importlib.util` (hyphen in filename prevents normal import)
- Use `unittest.mock` for HTTP mocking — no extra mock libraries
- Coverage minimum: **80%** — enforced by `pytest-cov` and `pyproject.toml`
- TUI functions (`pick_*`, `show_*`, `header`, `abort`) are excluded from unit tests; test logic-layer functions only

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR to `main`:
1. `ruff check` + `ruff format --check`
2. `pytest` (with coverage, fail under 80%)

## Pre-commit Hooks

Install once with `pre-commit install`. Hooks:
- `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-json`, `check-merge-conflict`, `detect-private-key`
- `ruff` (lint + fix) and `ruff-format`
- `gitleaks` secret scan

## Environment Variables

See `README.md` for the full table. Never hardcode secrets — use `.env` locally (git-ignored).

## File Layout

```
opencode-launcher.py   # main script
scripts/
  test-ica.sh          # live ICA endpoint smoke test
tests/
  conftest.py          # session-scoped launcher fixture + tmp_launcher
  test_opencode_launcher.py
pyproject.toml         # pytest + ruff + coverage config
Taskfile.yml           # task runner
.pre-commit-config.yaml
.github/workflows/ci.yml
.gitignore
README.md
AGENTS.md              # this file
```
