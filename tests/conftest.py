import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def launcher():
    """Import opencode-launcher.py as a module (hyphen-safe)."""
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("launcher", root / "opencode-launcher.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["opencode-launcher.py"]
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def tmp_launcher(tmp_path, launcher, monkeypatch):
    """Launcher with HISTORY_FILE and CACHE_FILE redirected to tmp_path."""
    monkeypatch.setattr(launcher, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(launcher, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(launcher, "LAUNCHER_DIR", tmp_path)
    return launcher
