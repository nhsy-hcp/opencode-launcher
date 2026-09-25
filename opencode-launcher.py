#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "questionary",
#     "rich",
#     "requests",
# ]
# ///
"""
opencode-launcher.py
────────────────────
Interactive TUI to pick a model and launch opencode.

Usage:
  ./opencode-launcher.py                             # default: LiteLLM proxy
  ./opencode-launcher.py --litellm-url URL           # override LiteLLM URL
  ./opencode-launcher.py --model litellm/my-model        # skip picker, launch directly
  ./opencode-launcher.py --dry-run                   # print command, don't execute
  ./opencode-launcher.py --no-cache                  # bypass cache, fetch fresh models
  ./opencode-launcher.py --clear-history             # wipe history/cache and exit
  ./opencode-launcher.py --clear-cache              # wipe model cache and exit

Environment variables:
  LITELLM_URL          LiteLLM proxy base URL (overrides default below)
  LITELLM_MASTER_KEY   LiteLLM API key (default: "litellm")
  OLLAMA_HOST          Ollama base URL (default: http://localhost:11434)
  NVIDIA_API_KEY       NVIDIA NIM API key (required if nvidia provider enabled)
  OPENROUTER_API_KEY   OpenRouter API key (required if openrouter provider enabled)
  ICA_API_KEY          ICA API key (required if ica provider enabled)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import questionary
import requests
from questionary import Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ──────────────────────────────────────────────────────────────────────────────
# Provider Feature Flags
# Set True/False to enable/disable each source. Edit this dict to toggle.
# ──────────────────────────────────────────────────────────────────────────────
PROVIDERS = {
    "litellm": True,  # LiteLLM proxy  —  see LITELLM_URL below
    "ollama": False,  # Local Ollama   —  requires OLLAMA_HOST or default localhost:11434
    "nvidia": False,  # NVIDIA NIM     —  requires NVIDIA_API_KEY env var
    "openrouter": False,  # OpenRouter     —  requires OPENROUTER_API_KEY env var
    "vllm": False,  # Remote vLLM    —  requires VLLM_URL env var or terraform directory
    "ica": True,  # ICA            —  requires ICA_API_KEY env var
}

# ──────────────────────────────────────────────────────────────────────────────
# URLs & defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_LITELLM_URL = "http://localhost:4000"
DEFAULT_LITELLM_KEY = "litellm"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def get_vllm_url() -> str:
    return os.getenv("VLLM_URL", "http://localhost:8000")


NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODELS_URL = f"{NVIDIA_API_BASE}/models"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_API_BASE}/models"
ICA_API_BASE = "https://api.servicesessentials.ibm.com/v1"
ICA_MODELS_URL = f"{ICA_API_BASE}/models"

OPENCODE_CONFIG_PATH = Path.home() / ".config" / "opencode" / "opencode.json"
LAUNCHER_DIR = Path.home() / ".config" / "opencode-launcher"
HISTORY_FILE = LAUNCHER_DIR / "history.json"
CACHE_FILE = LAUNCHER_DIR / "cache.json"

CACHE_TTL = 24 * 3600  # 24 h
MAX_HISTORY = 10
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
BYTES_PER_GB = 1e9

# ──────────────────────────────────────────────────────────────────────────────
# NVIDIA popular/exclude filters (used when nvidia is enabled)
# ──────────────────────────────────────────────────────────────────────────────
NVIDIA_POPULAR_PROVIDERS = {
    "meta",
    "deepseek-ai",
    "mistralai",
    "google",
    "qwen",
    "nvidia",
    "microsoft",
    "openai",
    "moonshotai",
}
NVIDIA_EXCLUDE_PATTERNS = [
    "embed",
    "retriever",
    "guard",
    "safety",
    "pii",
    "parse",
    "riva",
    "nvclip",
    "vila",
    "neva",
    "kosmos",
    "reward",
    "arctic-embed",
    "bge-",
    "nv-embed",
    "synthetic",
]

# ──────────────────────────────────────────────────────────────────────────────
# OpenRouter popular/exclude filters (used when openrouter is enabled)
# ──────────────────────────────────────────────────────────────────────────────
OPENROUTER_INCLUDE_PROVIDERS = {
    "anthropic",
    "openai",
    "google",
    "meta-llama",
    "mistralai",
    "deepseek",
    "qwen",
    "nvidia",
    "microsoft",
    "x-ai",
}
OPENROUTER_EXCLUDE_PATTERNS = [
    "embed",
    "ocr",
    "guard",
    "tts",
    "stt",
    "whisper",
    "imagen",
    "flux",
    "stability",
    "image-",
]
OPENROUTER_MIN_CONTEXT = 128_000

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
# TUI style
# ──────────────────────────────────────────────────────────────────────────────
STYLE = Style(
    [
        ("qmark", "fg:#76e3ea bold"),
        ("question", "bold"),
        ("answer", "fg:#76e3ea bold"),
        ("pointer", "fg:#76e3ea bold"),
        ("highlighted", "fg:#76e3ea bold"),
        ("selected", "fg:#76e3ea"),
        ("separator", "fg:#444444"),
        ("instruction", "fg:#888888 italic"),
        ("disabled", "fg:#666666 italic"),
    ]
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def header() -> None:
    console.print()
    console.print("[bold #00ffcc]◈ opencode[/bold #00ffcc] [dim]launcher[/dim]")
    active = "  ".join(f"[bold]{k}[/bold]" if v else f"[dim]{k}[/dim]" for k, v in PROVIDERS.items())
    console.print(f"[dim]providers:[/dim]  {active}")
    console.print()


def err(msg: str) -> None:
    console.print(f"[bold red]✗[/bold red] {msg}")


def ok(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] {msg}")


def abort() -> None:
    console.print()
    console.print(Panel("[dim]Aborted. Goodbye![/dim]", border_style="#666666", padding=(0, 2)))
    sys.exit(0)


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        err(f"Invalid URL: {url}")
        sys.exit(1)
    return url


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        os.unlink(tmp)
        raise


def _fetch(url: str, headers: dict, timeout: int, retries: int = 2) -> requests.Response:
    last: Exception = RuntimeError("no attempts")
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            last = exc
            if attempt < retries:
                time.sleep(2**attempt)
    raise last


# ──────────────────────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────────────────────


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        raw = HISTORY_FILE.read_text().strip()
        return json.loads(raw) if raw else []
    except (json.JSONDecodeError, OSError):
        return []


def save_history(history: list[dict]) -> None:
    _atomic_write(HISTORY_FILE, json.dumps(history[:MAX_HISTORY], indent=2) + "\n")


def record_selection(provider: str, model_id: str) -> None:
    history = load_history()
    history = [h for h in history if not (h["provider"] == provider and h["model_id"] == model_id)]
    history.insert(0, {"provider": provider, "model_id": model_id})
    save_history(history)


def filter_history(
    history: list[dict],
    litellm_models: list[dict],
    ollama_models: list[dict],
    nvidia_models: list[dict],
    openrouter_models: list[dict],
    ica_models: list[dict],
) -> list[dict]:
    available: set[str] = set()
    available |= {f"litellm/{m['id']}" for m in litellm_models}
    available |= {f"ollama/{m['name']}" for m in ollama_models}
    available |= {f"nvidia/{m['id']}" for m in nvidia_models}
    available |= {f"openrouter/{m['id']}" for m in openrouter_models}
    available |= {f"ica/{m['id']}" for m in ica_models}
    return [h for h in history if f"{h['provider']}/{h['model_id']}" in available]


def clear_history() -> None:
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────────────────────


def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}
    try:
        raw = CACHE_FILE.read_text().strip()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict) -> None:
    _atomic_write(CACHE_FILE, json.dumps(cache, indent=2) + "\n")


def _fresh(entry: dict) -> bool:
    return time.time() - entry.get("ts", 0) < CACHE_TTL


def clear_cache() -> None:
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# Fetchers
# ──────────────────────────────────────────────────────────────────────────────


def fetch_litellm_models(base_url: str, api_key: str, use_cache: bool = True, timeout: int = REQUEST_TIMEOUT, cache: dict | None = None) -> list[dict]:
    if use_cache:
        cache = cache if cache is not None else load_cache()
        if "litellm" in cache and _fresh(cache["litellm"]):
            return cache["litellm"]["data"]

    url = base_url.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = _fetch(url, headers, timeout)
        raw = resp.json().get("data", [])
    except requests.exceptions.ConnectionError:
        warn(f"LiteLLM proxy not reachable at {base_url} — skipping.")
        return []
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        err(f"LiteLLM error {code}: {exc}")
        return []
    except Exception as exc:
        warn(f"LiteLLM fetch failed: {exc} — skipping.")
        return []

    models = sorted(
        [{"id": m.get("id", ""), "provider": (m.get("id", "") or "").split("/")[0] or "unknown"} for m in raw],
        key=lambda x: x["id"].lower(),
    )
    if use_cache:
        cache = cache if cache is not None else load_cache()
        cache["litellm"] = {"ts": time.time(), "data": models}
        save_cache(cache)
    return models


def fetch_ollama_models(base_url: str, timeout: int, use_cache: bool = True, cache: dict | None = None) -> list[dict]:
    if use_cache:
        cache = cache if cache is not None else load_cache()
        if "ollama" in cache and _fresh(cache["ollama"]):
            return cache["ollama"]["data"]

    validate_url(base_url)
    url = base_url.rstrip("/") + "/api/tags"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        warn(f"Ollama not reachable at {base_url} — skipping.")
        return []
    except Exception as exc:
        warn(f"Ollama error: {exc} — skipping.")
        return []

    models = []
    for m in data.get("models", []):
        size_b = m.get("size", 0)
        models.append(
            {
                "name": m.get("name", ""),
                "size": f"{size_b / BYTES_PER_GB:.1f} GB" if size_b else "?",
                "family": m.get("details", {}).get("family", ""),
            }
        )
    models = sorted(models, key=lambda x: x["name"].lower())

    if use_cache:
        cache = cache if cache is not None else load_cache()
        cache["ollama"] = {"ts": time.time(), "data": models}
        save_cache(cache)
    return models


def fetch_nvidia_models(api_key: str, use_cache: bool = True, timeout: int = REQUEST_TIMEOUT, cache: dict | None = None) -> list[dict]:
    if use_cache:
        cache = cache if cache is not None else load_cache()
        if "nvidia" in cache and _fresh(cache["nvidia"]):
            return _filter_nvidia(cache["nvidia"]["data"])

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = _fetch(NVIDIA_MODELS_URL, headers, timeout)
        raw = resp.json().get("data", [])
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        err(f"NVIDIA error {code}: {exc}")
        return []
    except Exception as exc:
        err(f"NVIDIA fetch failed: {exc}")
        return []

    if use_cache:
        cache = cache if cache is not None else load_cache()
        cache["nvidia"] = {"ts": time.time(), "data": raw}
        save_cache(cache)
    return _filter_nvidia(raw)


def fetch_openrouter_models(api_key: str, use_cache: bool = True, timeout: int = REQUEST_TIMEOUT, cache: dict | None = None) -> list[dict]:
    if use_cache:
        cache = cache if cache is not None else load_cache()
        if "openrouter" in cache and _fresh(cache["openrouter"]):
            return _filter_openrouter(cache["openrouter"]["data"])

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = _fetch(OPENROUTER_MODELS_URL, headers, timeout)
        raw = resp.json().get("data", [])
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        err(f"OpenRouter error {code}: {exc}")
        return []
    except Exception as exc:
        err(f"OpenRouter fetch failed: {exc}")
        return []

    if use_cache:
        cache = cache if cache is not None else load_cache()
        cache["openrouter"] = {"ts": time.time(), "data": raw}
        save_cache(cache)
    return _filter_openrouter(raw)


def fetch_ica_models(api_key: str, use_cache: bool = True, timeout: int = REQUEST_TIMEOUT, cache: dict | None = None) -> list[dict]:
    if use_cache:
        cache = cache if cache is not None else load_cache()
        if "ica" in cache and _fresh(cache["ica"]):
            return cache["ica"]["data"]

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = _fetch(ICA_MODELS_URL, headers, timeout)
        raw = resp.json().get("data", [])
    except requests.exceptions.HTTPError as exc:
        code = exc.response.status_code if exc.response else "?"
        err(f"ICA error {code}: {exc}")
        return []
    except Exception as exc:
        err(f"ICA fetch failed: {exc}")
        return []

    models = sorted(
        [{"id": m.get("id", ""), "owned_by": m.get("owned_by", "")} for m in raw],
        key=lambda x: x["id"].lower(),
    )
    if use_cache:
        cache = cache if cache is not None else load_cache()
        cache["ica"] = {"ts": time.time(), "data": models}
        save_cache(cache)
    return models


def _filter_nvidia(raw: list[dict]) -> list[dict]:
    models = []
    for m in raw:
        mid = m.get("id", "")
        owned_by = m.get("owned_by", "")
        provider = mid.split("/")[0] if "/" in mid else owned_by
        if provider not in NVIDIA_POPULAR_PROVIDERS:
            continue
        if any(pat in mid.lower() for pat in NVIDIA_EXCLUDE_PATTERNS):
            continue
        models.append({"id": mid, "owned_by": owned_by})
    return sorted(models, key=lambda x: x["id"].lower())


def _filter_openrouter(raw: list[dict]) -> list[dict]:
    models = []
    for m in raw:
        mid = m.get("id", "")
        provider = mid.split("/")[0] if "/" in mid else ""
        if provider not in OPENROUTER_INCLUDE_PROVIDERS:
            continue
        if any(pat in mid.lower() for pat in OPENROUTER_EXCLUDE_PATTERNS):
            continue
        pricing = m.get("pricing", {}) or {}
        is_free = pricing.get("prompt") in ("0", 0) and pricing.get("completion") in ("0", 0)
        ctx = m.get("context_length", 0)
        if ctx < OPENROUTER_MIN_CONTEXT:
            continue
        models.append({"id": mid, "name": m.get("name", ""), "free": is_free, "context": ctx})
    return sorted(models, key=lambda x: x["id"].lower())


# ──────────────────────────────────────────────────────────────────────────────
# Display tables
# ──────────────────────────────────────────────────────────────────────────────


def _panel(title: str, count: int) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold #00ffcc]◈[/bold #00ffcc] [dim]Found[/dim] [bold]{count}[/bold] [dim]model(s)[/dim]",
            border_style="#00ffcc",
            padding=(0, 2),
            title=f"[dim]{title}[/dim]",
            subtitle="[dim]●[/dim]",
        )
    )


def _nav_hint() -> None:
    console.print("[dim]  ↑/↓ navigate  •  type to filter  •  enter select  •  q quit[/dim]")


def show_litellm_table(models: list[dict]) -> None:
    _panel("LiteLLM Proxy Models", len(models))
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold #00ffcc", expand=True)
    t.add_column("Model ID", style="#ffffff", ratio=3)
    t.add_column("Provider", style="#888888", ratio=1)
    for i, m in enumerate(models):
        row_style = "#1a1a2e" if i % 2 == 0 else ""
        t.add_row(m["id"], m["provider"], style=row_style)
    console.print(t)
    _nav_hint()


def show_ica_table(models: list[dict]) -> None:
    _panel("ICA Models", len(models))
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold #00ffcc", expand=True)
    t.add_column("Model ID", style="#ffffff", ratio=3)
    t.add_column("Owner", style="#888888", ratio=1)
    for i, m in enumerate(models):
        t.add_row(m["id"], m.get("owned_by", ""), style="#1a1a2e" if i % 2 == 0 else "")
    console.print(t)
    _nav_hint()


def show_ollama_table(models: list[dict]) -> None:
    _panel("Ollama Models", len(models))
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold #00ffcc", expand=True)
    t.add_column("Model", style="#ffffff", ratio=2)
    t.add_column("Family", style="#888888", ratio=1)
    t.add_column("Size", style="#ffaa00", justify="right", ratio=1)
    for i, m in enumerate(models):
        t.add_row(m["name"], m["family"], m["size"], style="#1a1a2e" if i % 2 == 0 else "")
    console.print(t)
    _nav_hint()


def show_nvidia_table(models: list[dict]) -> None:
    _panel("NVIDIA NIM Models", len(models))
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold #00ffcc", expand=True)
    t.add_column("Model ID", style="#ffffff", ratio=3)
    t.add_column("Owner", style="#888888", ratio=1)
    for i, m in enumerate(models):
        t.add_row(m["id"], m["owned_by"], style="#1a1a2e" if i % 2 == 0 else "")
    console.print(t)
    _nav_hint()


def show_openrouter_table(models: list[dict]) -> None:
    _panel("OpenRouter Models", len(models))
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold #00ffcc", expand=True)
    t.add_column("Model ID", style="#ffffff", ratio=3)
    t.add_column("Name", style="#888888", ratio=2)
    t.add_column("Ctx", style="#ffaa00", justify="right", ratio=1)
    t.add_column("Tier", justify="center", ratio=1)
    for i, m in enumerate(models):
        tier = "[green]free[/green]" if m["free"] else "[yellow]paid[/yellow]"
        ctx = f"{m['context'] // 1000}k" if m["context"] else "?"
        t.add_row(m["id"], m["name"], ctx, tier, style="#1a1a2e" if i % 2 == 0 else "")
    console.print(t)
    _nav_hint()


# ──────────────────────────────────────────────────────────────────────────────
# TUI pickers
# ──────────────────────────────────────────────────────────────────────────────

ICONS = {"litellm": "🔀", "ollama": "🦙", "nvidia": "⚡", "openrouter": "🌐", "vllm": "🚀", "ica": "🧠"}


def pick_provider(
    has: dict[str, bool],
    history: list[dict],
) -> tuple | None:
    """Return ('recent', provider, model_id) or ('provider', name, None)."""
    choices = []

    for h in history:
        icon = ICONS.get(h["provider"], "•")
        choices.append(
            questionary.Choice(
                f"⏱  {icon} {h['provider']:<12}  {h['model_id']}",
                value=("recent", h["provider"], h["model_id"]),
            )
        )

    if history:
        choices.append(questionary.Separator("───────"))

    order = ["litellm", "ollama", "nvidia", "openrouter", "vllm", "ica"]
    labels = {
        "litellm": "🔀  LiteLLM  (proxy)",
        "ollama": "🦙  Ollama   (local)",
        "nvidia": "⚡  NVIDIA NIM  (cloud)",
        "openrouter": "🌐  OpenRouter  (cloud)",
        "vllm": "🚀  vLLM        (remote)",
        "ica": "🧠  ICA          (cloud)",
    }
    for name in order:
        if has.get(name):
            choices.append(questionary.Choice(labels[name], value=("provider", name, None)))

    if history:
        choices.append(questionary.Separator("───────"))
        choices.append(
            questionary.Choice(
                "🗑  Clear recent history & cache",
                value=("clear-history", None, None),
            )
        )
        choices.append(
            questionary.Choice(
                "🗑  Clear model cache",
                value=("clear-cache", None, None),
            )
        )

    if not choices:
        err("No providers available.")
        return None

    if len(choices) == 1 and not history:
        return choices[0].value

    result = questionary.select(
        "Select model or provider:  (↑/↓ navigate  •  type to filter  •  q quit)",
        choices=choices,
        style=STYLE,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()
    if result is None:
        abort()
    return result


def _pick(models: list[dict], title: str, label_fn) -> str | None:
    choices = [questionary.Choice(label_fn(m), value=m["id"] if "id" in m else m["name"]) for m in models]
    choices.append(questionary.Choice("↩  Back", value="__back__"))
    result = questionary.select(
        f"{title}  (type to filter, ↑/↓ navigate):",
        choices=choices,
        style=STYLE,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()
    if result is None:
        abort()
    return result


def pick_litellm_model(models: list[dict]) -> str | None:
    return _pick(models, "Select LiteLLM model", lambda m: f"{m['id']}  [{m['provider']}]")


def pick_ollama_model(models: list[dict]) -> str | None:
    return _pick(models, "Select Ollama model", lambda m: f"{m['name']}  [{m['family']}  {m['size']}]")


def pick_nvidia_model(models: list[dict]) -> str | None:
    return _pick(models, "Select NVIDIA model", lambda m: f"{m['id']}  [{m['owned_by']}]")


def pick_openrouter_model(models: list[dict]) -> str | None:
    return _pick(models, "Select OpenRouter model", lambda m: f"{m['id']}  [{m['name']}  {'free' if m['free'] else 'paid'}]")


# ──────────────────────────────────────────────────────────────────────────────
# opencode.json helpers
# ──────────────────────────────────────────────────────────────────────────────


def _read_oc_config() -> dict:
    try:
        if OPENCODE_CONFIG_PATH.exists():
            return json.loads(OPENCODE_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _write_oc_config(config: dict) -> None:
    _atomic_write(OPENCODE_CONFIG_PATH, json.dumps(config, indent=2) + "\n")


def _ensure_oc_litellm(model_id: str, litellm_url: str, api_key: str) -> None:
    config = _read_oc_config()
    providers = config.setdefault("provider", {})
    prov = providers.setdefault(
        "litellm",
        {
            "npm": "@ai-sdk/openai-compatible",
            "name": "LiteLLM",
            "options": {"baseURL": "", "apiKey": ""},
            "models": {},
        },
    )
    opts = prov.setdefault("options", {})
    opts["baseURL"] = litellm_url.rstrip("/") + "/v1"
    opts["apiKey"] = api_key
    prov.setdefault("models", {})[model_id] = {"name": model_id}
    _write_oc_config(config)


def _ensure_oc_ollama(model_id: str, ollama_url: str) -> None:
    config = _read_oc_config()
    base_url = ollama_url.rstrip("/") + "/v1"
    prov = config.setdefault("provider", {}).setdefault(
        "ollama",
        {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Ollama",
            "options": {"baseURL": base_url},
            "models": {},
        },
    )
    prov.setdefault("options", {})["baseURL"] = base_url
    prov.setdefault("models", {}).setdefault(model_id, {"name": model_id})
    _write_oc_config(config)


def _ensure_oc_openrouter(model_id: str, api_key: str) -> None:
    config = _read_oc_config()
    prov = config.setdefault("provider", {}).setdefault(
        "openrouter",
        {
            "npm": "@ai-sdk/openai-compatible",
            "name": "OpenRouter",
            "options": {"baseURL": OPENROUTER_API_BASE, "apiKey": ""},
            "models": {},
        },
    )
    prov.setdefault("options", {})["apiKey"] = api_key
    prov.setdefault("models", {})[model_id] = {"name": model_id}
    _write_oc_config(config)


def _ensure_oc_vllm(model_id: str, vllm_url: str) -> None:
    config = _read_oc_config()
    if not vllm_url:
        err("VLLM URL is not set. Run from tf dir or set VLLM_URL.")
        sys.exit(1)
    base_url = vllm_url.rstrip("/") + "/v1"
    prov = config.setdefault("provider", {}).setdefault(
        "vllm",
        {
            "npm": "@ai-sdk/openai-compatible",
            "name": "vLLM",
            "options": {"baseURL": base_url, "apiKey": "dummy"},
            "models": {},
        },
    )
    prov.setdefault("options", {})["baseURL"] = base_url
    prov.setdefault("options", {})["apiKey"] = "dummy"
    prov.setdefault("models", {})[model_id] = {"name": model_id}
    _write_oc_config(config)


def _ensure_oc_ica(model_id: str, api_key: str) -> None:
    config = _read_oc_config()
    prov = config.setdefault("provider", {}).setdefault(
        "ica",
        {
            "npm": "@ai-sdk/openai-compatible",
            "name": "ICA",
            "options": {"baseURL": ICA_API_BASE, "apiKey": ""},
            "models": {},
        },
    )
    prov.setdefault("options", {})["baseURL"] = ICA_API_BASE
    prov.setdefault("options", {})["apiKey"] = api_key
    prov.setdefault("models", {})[model_id] = {"name": model_id}
    _write_oc_config(config)


# ──────────────────────────────────────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────────────────────────────────────


def launch(
    provider: str,
    model_id: str,
    litellm_url: str,
    litellm_key: str,
    openrouter_key: str,
    ollama_url: str,
    vllm_url: str,
    ica_key: str,
    dry_run: bool,
    extra_args: list[str],
) -> None:
    if not re.match(r"^[a-zA-Z0-9._\-:/]+$", model_id):
        err(f"Unsafe model ID: {model_id}")
        sys.exit(1)

    if provider == "litellm":
        _ensure_oc_litellm(model_id, litellm_url, litellm_key)
        model_flag = f"litellm/{model_id}"
    elif provider == "ollama":
        _ensure_oc_ollama(model_id, ollama_url)
        model_flag = f"ollama/{model_id}"
    elif provider == "openrouter":
        _ensure_oc_openrouter(model_id, openrouter_key)
        model_flag = f"openrouter/{model_id}"
    elif provider == "vllm":
        _ensure_oc_vllm(model_id, vllm_url)
        model_flag = f"vllm/{model_id}"
    elif provider == "ica":
        _ensure_oc_ica(model_id, ica_key)
        model_flag = f"ica/{model_id}"
    else:
        model_flag = f"{provider}/{model_id}"

    cmd = ["opencode", "--model", model_flag, *extra_args]

    extra_info = ""
    if provider == "litellm":
        extra_info = f"\n[bold]Base URL:[/bold]   [dim]{litellm_url}/v1[/dim]"
    elif provider == "vllm":
        extra_info = f"\n[bold]Base URL:[/bold]   [dim]{vllm_url}/v1[/dim]"

    console.print()
    console.print(
        Panel(
            f"[bold]Provider:[/bold]  [cyan]{provider}[/cyan]\n[bold]Model:[/bold]     [green]{model_id}[/green]\n[bold]Command:[/bold]   [dim]{' '.join(cmd)}[/dim]{extra_info}",
            title="[bold cyan]◈[/bold cyan] [bold]Launching opencode[/bold]",
            border_style="#00ffcc",
            padding=(0, 1),
            expand=False,
        )
    )
    console.print()

    record_selection(provider, model_id)

    if dry_run:
        warn("[dim]Dry-run — not executing.[/dim]")
        return

    try:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    except FileNotFoundError:
        err("'opencode' not found in PATH. Install: https://opencode.ai/docs/")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive model picker for opencode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--litellm-url",
        default=os.getenv("LITELLM_URL", DEFAULT_LITELLM_URL),
        metavar="URL",
        type=validate_url,
        help=f"LiteLLM base URL (default: {DEFAULT_LITELLM_URL})",
    )
    p.add_argument(
        "--model",
        metavar="PROVIDER/MODEL",
        default=None,
        help="Skip picker and launch directly (e.g. litellm/my-model)",
    )
    p.add_argument(
        "--vllm-url",
        default=get_vllm_url(),
        metavar="URL",
        help="vLLM base URL (default: inferred from terraform)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print command, don't execute")
    p.add_argument("--no-cache", action="store_true", help="Fetch fresh models, bypass cache")
    p.add_argument("--clear-history", action="store_true", help="Clear history/cache and exit")
    p.add_argument("--clear-cache", action="store_true", help="Clear model cache and exit")
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help=f"Request timeout in seconds (default: {REQUEST_TIMEOUT})",
    )
    p.add_argument(
        "opencode_args",
        nargs=argparse.REMAINDER,
        metavar="...",
        help="Extra args passed through to opencode",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    header()

    if args.clear_history:
        clear_history()
        clear_cache()
        ok("History and cache cleared.")
        sys.exit(0)

    if args.clear_cache:
        clear_cache()
        ok("Cache cleared.")
        sys.exit(0)

    litellm_key = os.getenv("LITELLM_MASTER_KEY", DEFAULT_LITELLM_KEY)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_URL)
    ica_key = os.getenv("ICA_API_KEY", "")
    timeout = args.timeout or REQUEST_TIMEOUT
    use_cache = not args.no_cache

    # ── Direct launch ─────────────────────────────────────────────────────────
    if args.model:
        provider, _, model_id = args.model.partition("/")
        if not model_id:
            err("--model must be PROVIDER/MODEL (e.g. litellm/my-model)")
            sys.exit(1)
        if provider not in PROVIDERS:
            err(f"Unknown provider '{provider}'. Valid: {', '.join(PROVIDERS)}")
            sys.exit(1)
        extra = [a for a in args.opencode_args if a != "--"]
        launch(provider, model_id, args.litellm_url, litellm_key, openrouter_key, ollama_url, args.vllm_url, ica_key, args.dry_run, extra)
        return

    # ── Fetch enabled providers (parallel) ───────────────────────────────────
    shared_cache: dict | None = load_cache() if use_cache else None
    tasks: dict[str, object] = {}
    if PROVIDERS.get("litellm"):
        tasks["litellm"] = lambda: fetch_litellm_models(args.litellm_url, litellm_key, use_cache, timeout, shared_cache)
    if PROVIDERS.get("ollama"):
        tasks["ollama"] = lambda: fetch_ollama_models(ollama_url, timeout, use_cache, shared_cache)
    if PROVIDERS.get("nvidia"):
        if not nvidia_key:
            warn("NVIDIA_API_KEY not set — skipping NVIDIA NIM.")
        else:
            tasks["nvidia"] = lambda: fetch_nvidia_models(nvidia_key, use_cache, timeout, shared_cache)
    if PROVIDERS.get("openrouter"):
        if not openrouter_key:
            warn("OPENROUTER_API_KEY not set — skipping OpenRouter.")
        else:
            tasks["openrouter"] = lambda: fetch_openrouter_models(openrouter_key, use_cache, timeout, shared_cache)
    if PROVIDERS.get("ica"):
        if not ica_key:
            warn("ICA_API_KEY not set — skipping ICA.")
        else:
            tasks["ica"] = lambda: fetch_ica_models(ica_key, use_cache, timeout, shared_cache)

    fetched: dict[str, list[dict]] = {name: [] for name in tasks}
    if tasks:
        n = len(tasks)
        label = ", ".join(tasks)
        with (
            console.status(
                f"[bold #00ffcc]Fetching models from {n} provider(s): {label}…[/bold #00ffcc]",
                spinner="dots",
            ),
            ThreadPoolExecutor(max_workers=n) as ex,
        ):
            futs = {ex.submit(fn): name for name, fn in tasks.items()}
            for fut in as_completed(futs):
                fetched[futs[fut]] = fut.result()

    # Report results in fixed display order
    litellm_models: list[dict] = fetched.get("litellm", [])
    ollama_models: list[dict] = fetched.get("ollama", [])
    nvidia_models: list[dict] = fetched.get("nvidia", [])
    openrouter_models: list[dict] = fetched.get("openrouter", [])
    ica_models: list[dict] = fetched.get("ica", [])

    if "litellm" in tasks:
        if litellm_models:
            ok(f"Found [bold]{len(litellm_models)}[/bold] LiteLLM model(s)")
        else:
            warn("No LiteLLM models returned (proxy running? key correct?)")
    if "ollama" in tasks:
        if ollama_models:
            ok(f"Found [bold]{len(ollama_models)}[/bold] Ollama model(s)")
        else:
            warn("No Ollama models returned.")
    if "nvidia" in tasks:
        if nvidia_models:
            ok(f"Found [bold]{len(nvidia_models)}[/bold] NVIDIA model(s)")
        else:
            warn("No NVIDIA models returned.")
    if "openrouter" in tasks:
        if openrouter_models:
            ok(f"Found [bold]{len(openrouter_models)}[/bold] OpenRouter model(s)")
        else:
            warn("No OpenRouter models returned.")
    if "ica" in tasks:
        if ica_models:
            ok(f"Found [bold]{len(ica_models)}[/bold] ICA model(s)")
        else:
            warn("No ICA models returned.")

    has = {
        "litellm": bool(litellm_models),
        "ollama": bool(ollama_models),
        "nvidia": bool(nvidia_models),
        "openrouter": bool(openrouter_models),
        "vllm": PROVIDERS.get("vllm", False),
        "ica": bool(ica_models),
    }

    if not any(has.values()):
        err("No models available from any provider. Exiting.")
        sys.exit(1)

    # ── History ───────────────────────────────────────────────────────────────
    history = load_history()
    history = filter_history(history, litellm_models, ollama_models, nvidia_models, openrouter_models, ica_models)

    # ── Provider / model picker loop ──────────────────────────────────────────
    while True:
        selection = pick_provider(has, history)
        if selection is None:
            sys.exit(0)

        kind, prov, mid = selection

        if kind == "clear-history":
            confirmed = questionary.confirm("Clear history and cache?", default=False, style=STYLE).ask()
            if confirmed is None:
                abort()
            if confirmed:
                clear_history()
                clear_cache()
                history = []
                ok("Cleared.")
            console.print()
            continue

        if kind == "clear-cache":
            confirmed = questionary.confirm("Clear model cache?", default=False, style=STYLE).ask()
            if confirmed is None:
                abort()
            if confirmed:
                clear_cache()
                ok("Cache cleared.")
            console.print()
            continue

        if kind == "recent":
            provider, model_id = prov, mid
            break

        provider = prov
        if provider == "litellm":
            model_id = pick_litellm_model(litellm_models)
        elif provider == "ollama":
            model_id = pick_ollama_model(ollama_models)
        elif provider == "nvidia":
            model_id = pick_nvidia_model(nvidia_models)
        elif provider == "vllm":
            model_id = "local-model"
        elif provider == "ica":
            model_id = _pick(ica_models, "Select ICA model", lambda m: f"{m['id']}  [{m['owned_by']}]")
        else:
            model_id = pick_openrouter_model(openrouter_models)

        if model_id == "__back__":
            console.print()
            continue
        break

    extra = [a for a in args.opencode_args if a != "--"]
    launch(provider, model_id, args.litellm_url, litellm_key, openrouter_key, ollama_url, args.vllm_url, ica_key, args.dry_run, extra)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Aborted.[/dim]")
        sys.exit(0)
