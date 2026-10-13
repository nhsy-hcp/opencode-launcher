import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests as req

# ── validate_url ──────────────────────────────────────────────────────────────


class TestValidateUrl:
    def test_valid_http(self, launcher):
        assert launcher.validate_url("http://example.com") == "http://example.com"

    def test_valid_https(self, launcher):
        assert launcher.validate_url("https://litellm.example.co.uk") == "https://litellm.example.co.uk"

    def test_invalid_no_scheme(self, launcher):
        with pytest.raises(SystemExit):
            launcher.validate_url("not-a-url")

    def test_invalid_ftp(self, launcher):
        with pytest.raises(SystemExit):
            launcher.validate_url("ftp://example.com")

    def test_invalid_empty(self, launcher):
        with pytest.raises(SystemExit):
            launcher.validate_url("")


# ── cache TTL (_fresh) ────────────────────────────────────────────────────────


class TestFresh:
    def test_fresh_recent(self, launcher):
        entry = {"ts": time.time()}
        assert launcher._fresh(entry) is True

    def test_stale(self, launcher):
        entry = {"ts": time.time() - launcher.CACHE_TTL - 1}
        assert launcher._fresh(entry) is False

    def test_missing_ts(self, launcher):
        assert launcher._fresh({}) is False

    def test_exactly_on_boundary(self, launcher):
        entry = {"ts": time.time() - launcher.CACHE_TTL}
        assert launcher._fresh(entry) is False


# ── history ───────────────────────────────────────────────────────────────────


class TestHistory:
    def test_load_empty_when_no_file(self, tmp_launcher):
        assert tmp_launcher.load_history() == []

    def test_save_and_load_roundtrip(self, tmp_launcher):
        data = [{"provider": "litellm", "model_id": "gpt-4o"}]
        tmp_launcher.save_history(data)
        assert tmp_launcher.load_history() == data

    def test_load_corrupted_returns_empty(self, tmp_launcher):
        tmp_launcher.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_launcher.HISTORY_FILE.write_text("not json")
        assert tmp_launcher.load_history() == []

    def test_record_deduplicates(self, tmp_launcher):
        tmp_launcher.record_selection("litellm", "gpt-4o")
        tmp_launcher.record_selection("litellm", "gpt-4o")
        history = tmp_launcher.load_history()
        assert history.count({"provider": "litellm", "model_id": "gpt-4o"}) == 1

    def test_record_most_recent_first(self, tmp_launcher):
        tmp_launcher.record_selection("litellm", "model-a")
        tmp_launcher.record_selection("ollama", "model-b")
        history = tmp_launcher.load_history()
        assert history[0] == {"provider": "ollama", "model_id": "model-b"}

    def test_record_promotes_existing(self, tmp_launcher):
        tmp_launcher.record_selection("litellm", "model-a")
        tmp_launcher.record_selection("ollama", "model-b")
        tmp_launcher.record_selection("litellm", "model-a")
        history = tmp_launcher.load_history()
        assert history[0] == {"provider": "litellm", "model_id": "model-a"}

    def test_save_truncates_to_max(self, tmp_launcher):
        data = [{"provider": "p", "model_id": str(i)} for i in range(20)]
        tmp_launcher.save_history(data)
        assert len(tmp_launcher.load_history()) == tmp_launcher.MAX_HISTORY

    def test_clear_history(self, tmp_launcher):
        tmp_launcher.record_selection("litellm", "x")
        tmp_launcher.clear_history()
        assert not tmp_launcher.HISTORY_FILE.exists()


# ── cache ─────────────────────────────────────────────────────────────────────


class TestCache:
    def test_load_empty_when_no_file(self, tmp_launcher):
        assert tmp_launcher.load_cache() == {}

    def test_save_and_load_roundtrip(self, tmp_launcher):
        data = {"litellm": {"ts": 1234, "data": [{"id": "m"}]}}
        tmp_launcher.save_cache(data)
        assert tmp_launcher.load_cache() == data

    def test_load_corrupted_returns_empty(self, tmp_launcher):
        tmp_launcher.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp_launcher.CACHE_FILE.write_text("{broken")
        assert tmp_launcher.load_cache() == {}

    def test_clear_cache(self, tmp_launcher):
        tmp_launcher.save_cache({"key": "val"})
        tmp_launcher.clear_cache()
        assert not tmp_launcher.CACHE_FILE.exists()

    def test_clear_cache_noop_when_missing(self, tmp_launcher):
        tmp_launcher.clear_cache()


# ── _atomic_write ─────────────────────────────────────────────────────────────


class TestAtomicWrite:
    def test_creates_file(self, launcher, tmp_path):
        p = tmp_path / "sub" / "out.json"
        launcher._atomic_write(p, '{"ok":true}')
        assert p.read_text() == '{"ok":true}'

    def test_overwrites_existing(self, launcher, tmp_path):
        p = tmp_path / "out.json"
        p.write_text("old")
        launcher._atomic_write(p, "new")
        assert p.read_text() == "new"


# ── filter_history ────────────────────────────────────────────────────────────


class TestFilterHistory:
    def test_keeps_matching_entries(self, launcher):
        history = [{"provider": "litellm", "model_id": "gpt-4o"}]
        litellm = [{"id": "gpt-4o", "provider": "openai"}]
        result = launcher.filter_history(history, litellm, [], [], [], [])
        assert result == history

    def test_removes_unavailable(self, launcher):
        history = [{"provider": "litellm", "model_id": "gone-model"}]
        result = launcher.filter_history(history, [], [], [], [], [])
        assert result == []

    def test_mixed(self, launcher):
        history = [
            {"provider": "litellm", "model_id": "present"},
            {"provider": "ollama", "model_id": "missing"},
        ]
        litellm = [{"id": "present", "provider": "x"}]
        result = launcher.filter_history(history, litellm, [], [], [], [])
        assert result == [{"provider": "litellm", "model_id": "present"}]


# ── _filter_nvidia ────────────────────────────────────────────────────────────


class TestFilterNvidia:
    def _model(self, mid, owned_by="meta"):
        return {"id": mid, "owned_by": owned_by}

    def test_keeps_popular_provider(self, launcher):
        raw = [self._model("meta/llama-3-8b")]
        assert len(launcher._filter_nvidia(raw)) == 1

    def test_excludes_unknown_provider(self, launcher):
        raw = [self._model("unknown-co/some-model", owned_by="unknown-co")]
        assert launcher._filter_nvidia(raw) == []

    def test_excludes_embed_pattern(self, launcher):
        raw = [self._model("nvidia/nv-embed-v2", owned_by="nvidia")]
        assert launcher._filter_nvidia(raw) == []

    def test_sorted_output(self, launcher):
        raw = [self._model("nvidia/zzz"), self._model("meta/aaa")]
        ids = [m["id"] for m in launcher._filter_nvidia(raw)]
        assert ids == sorted(ids, key=str.lower)

    def test_empty_input(self, launcher):
        assert launcher._filter_nvidia([]) == []


# ── _filter_openrouter ────────────────────────────────────────────────────────


class TestFilterOpenrouter:
    def _model(self, mid, name="Test", ctx=200_000, prompt="0.001", completion="0.002"):
        return {"id": mid, "name": name, "context_length": ctx, "pricing": {"prompt": prompt, "completion": completion}}

    def test_keeps_valid_model(self, launcher):
        raw = [self._model("anthropic/claude-3-5-sonnet")]
        assert len(launcher._filter_openrouter(raw)) == 1

    def test_excludes_unknown_provider(self, launcher):
        raw = [self._model("random-corp/model")]
        assert launcher._filter_openrouter(raw) == []

    def test_excludes_embed_pattern(self, launcher):
        raw = [self._model("openai/text-embed-3")]
        assert launcher._filter_openrouter(raw) == []

    def test_excludes_short_context(self, launcher):
        raw = [self._model("anthropic/claude-haiku", ctx=10_000)]
        assert launcher._filter_openrouter(raw) == []

    def test_free_flag_set(self, launcher):
        raw = [self._model("openai/gpt-4o-free", prompt="0", completion="0")]
        result = launcher._filter_openrouter(raw)
        assert result[0]["free"] is True

    def test_paid_flag_set(self, launcher):
        raw = [self._model("openai/gpt-4o")]
        result = launcher._filter_openrouter(raw)
        assert result[0]["free"] is False

    def test_sorted_output(self, launcher):
        raw = [self._model("openai/zzz"), self._model("anthropic/aaa")]
        ids = [m["id"] for m in launcher._filter_openrouter(raw)]
        assert ids == sorted(ids, key=str.lower)


# ── fetch_litellm_models (mocked) ─────────────────────────────────────────────


class TestFetchLitellm:
    def test_returns_sorted_models(self, tmp_launcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-3"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            models = tmp_launcher.fetch_litellm_models("http://litellm.local", "key", use_cache=False)
        ids = [m["id"] for m in models]
        assert ids == sorted(ids, key=str.lower)

    def test_connection_error_returns_empty(self, tmp_launcher):
        import requests as req

        with patch("requests.get", side_effect=req.exceptions.ConnectionError):
            models = tmp_launcher.fetch_litellm_models("http://litellm.local", "key", use_cache=False)
        assert models == []

    def test_uses_cache_when_fresh(self, tmp_launcher):
        cached = [{"id": "cached/model", "provider": "cached"}]
        tmp_launcher.save_cache({"litellm": {"ts": time.time(), "data": cached}})
        models = tmp_launcher.fetch_litellm_models("http://litellm.local", "key", use_cache=True)
        assert models == cached

    def test_provider_extracted_from_id(self, tmp_launcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "openai/gpt-4o"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            models = tmp_launcher.fetch_litellm_models("http://litellm.local", "key", use_cache=False)
        assert models[0]["provider"] == "openai"


# ── fetch_ica_models (mocked) ─────────────────────────────────────────────────


class TestFetchIca:
    def test_returns_sorted_models(self, tmp_launcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": [{"id": "ibm/granite-13b", "owned_by": "ibm"}, {"id": "abc/model", "owned_by": "abc"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            models = tmp_launcher.fetch_ica_models("key", use_cache=False)
        ids = [m["id"] for m in models]
        assert ids == sorted(ids, key=str.lower)

    def test_http_error_returns_empty(self, tmp_launcher):
        import requests as req

        mock_resp = MagicMock()
        mock_resp.status_code = 401
        with patch("requests.get", side_effect=req.exceptions.HTTPError(response=mock_resp)):
            models = tmp_launcher.fetch_ica_models("badkey", use_cache=False)
        assert models == []

    def test_uses_cache_when_fresh(self, tmp_launcher):
        cached = [{"id": "ibm/granite", "owned_by": "ibm"}]
        tmp_launcher.save_cache({"ica": {"ts": time.time(), "data": cached}})
        models = tmp_launcher.fetch_ica_models("key", use_cache=True)
        assert models == cached


# ── _fetch retries ────────────────────────────────────────────────────────────


class TestFetchHelper:
    def test_returns_on_success(self, launcher):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            resp = launcher._fetch("http://example.com", {}, 5, retries=0)
        assert resp is mock_resp

    def test_retries_on_failure_then_succeeds(self, launcher):
        good = MagicMock()
        good.raise_for_status.return_value = None
        with patch("requests.get", side_effect=[req.exceptions.ConnectionError(), good]), patch("time.sleep"):
            resp = launcher._fetch("http://example.com", {}, 5, retries=1)
        assert resp is good

    def test_raises_after_all_retries(self, launcher):
        with patch("requests.get", side_effect=req.exceptions.ConnectionError("down")), patch("time.sleep"), pytest.raises(req.exceptions.ConnectionError):
            launcher._fetch("http://example.com", {}, 5, retries=1)


# ── fetch_ollama_models (mocked) ──────────────────────────────────────────────


class TestFetchOllama:
    def _mock_resp(self, models_data):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": models_data}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    def test_returns_sorted_models(self, tmp_launcher):
        data = [
            {"name": "llama3:8b", "size": 4_000_000_000, "details": {"family": "llama"}},
            {"name": "codellama:7b", "size": 3_500_000_000, "details": {"family": "llama"}},
        ]
        with patch("requests.get", return_value=self._mock_resp(data)):
            models = tmp_launcher.fetch_ollama_models("http://localhost:11434", 5, use_cache=False)
        ids = [m["name"] for m in models]
        assert ids == sorted(ids, key=str.lower)

    def test_size_formatted_as_gb(self, tmp_launcher):
        data = [{"name": "llama3:8b", "size": 4_000_000_000, "details": {"family": "llama"}}]
        with patch("requests.get", return_value=self._mock_resp(data)):
            models = tmp_launcher.fetch_ollama_models("http://localhost:11434", 5, use_cache=False)
        assert "GB" in models[0]["size"]

    def test_connection_error_returns_empty(self, tmp_launcher):
        with patch("requests.get", side_effect=req.exceptions.ConnectionError()):
            models = tmp_launcher.fetch_ollama_models("http://localhost:11434", 5, use_cache=False)
        assert models == []

    def test_uses_cache_when_fresh(self, tmp_launcher):
        cached = [{"name": "cached-model", "size": "4.0 GB", "family": "llama"}]
        tmp_launcher.save_cache({"ollama": {"ts": time.time(), "data": cached}})
        models = tmp_launcher.fetch_ollama_models("http://localhost:11434", 5, use_cache=True)
        assert models == cached


# ── fetch_nvidia_models (mocked) ──────────────────────────────────────────────


class TestFetchNvidia:
    def _raw(self):
        return [
            {"id": "meta/llama-3-8b", "owned_by": "meta"},
            {"id": "nvidia/nv-embed-v2", "owned_by": "nvidia"},
        ]

    def test_filters_applied(self, tmp_launcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": self._raw()}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            models = tmp_launcher.fetch_nvidia_models("key", use_cache=False)
        ids = [m["id"] for m in models]
        assert "meta/llama-3-8b" in ids
        assert "nvidia/nv-embed-v2" not in ids

    def test_http_error_returns_empty(self, tmp_launcher):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("requests.get", side_effect=req.exceptions.HTTPError(response=mock_resp)):
            models = tmp_launcher.fetch_nvidia_models("badkey", use_cache=False)
        assert models == []

    def test_uses_cache_when_fresh(self, tmp_launcher):
        raw = [{"id": "meta/llama-3-8b", "owned_by": "meta"}]
        tmp_launcher.save_cache({"nvidia": {"ts": time.time(), "data": raw}})
        models = tmp_launcher.fetch_nvidia_models("key", use_cache=True)
        assert any(m["id"] == "meta/llama-3-8b" for m in models)


# ── fetch_openrouter_models (mocked) ──────────────────────────────────────────


class TestFetchOpenrouter:
    def _raw(self):
        return [
            {"id": "anthropic/claude-3-5-sonnet", "name": "Claude", "context_length": 200_000, "pricing": {"prompt": "0.003", "completion": "0.015"}},
            {"id": "openai/text-embed-3", "name": "Embed", "context_length": 200_000, "pricing": {"prompt": "0.001", "completion": "0.001"}},
        ]

    def test_filters_applied(self, tmp_launcher):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": self._raw()}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.get", return_value=mock_resp):
            models = tmp_launcher.fetch_openrouter_models("key", use_cache=False)
        ids = [m["id"] for m in models]
        assert "anthropic/claude-3-5-sonnet" in ids
        assert "openai/text-embed-3" not in ids

    def test_uses_cache_when_fresh(self, tmp_launcher):
        cached = [{"id": "anthropic/claude-3-5-sonnet", "name": "Claude", "free": False, "context": 200_000}]
        tmp_launcher.save_cache({"openrouter": {"ts": time.time(), "data": []}})
        with patch.object(tmp_launcher, "_filter_openrouter", return_value=cached):
            models = tmp_launcher.fetch_openrouter_models("key", use_cache=True)
        assert models == cached


# ── opencode.json helpers ─────────────────────────────────────────────────────


class TestOcConfigHelpers:
    def test_read_returns_empty_when_missing(self, launcher, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher, "OPENCODE_CONFIG_PATH", tmp_path / "nonexistent.json")
        assert launcher._read_oc_config() == {}

    def test_read_returns_empty_on_corrupt(self, launcher, tmp_path, monkeypatch):
        p = tmp_path / "opencode.json"
        p.write_text("{bad json")
        monkeypatch.setattr(launcher, "OPENCODE_CONFIG_PATH", p)
        assert launcher._read_oc_config() == {}

    def test_write_and_read_roundtrip(self, launcher, tmp_path, monkeypatch):
        p = tmp_path / "opencode.json"
        monkeypatch.setattr(launcher, "OPENCODE_CONFIG_PATH", p)
        launcher._write_oc_config({"foo": "bar"})
        assert launcher._read_oc_config() == {"foo": "bar"}

    def test_ensure_provider_writes_v2_shape(self, launcher):
        cfg = {}
        launcher._ensure_oc_provider(cfg, "ica", "claude-sonnet-5", "ICA", launcher.ICA_API_BASE, "{env:ICA_OPENCODE_API_KEY}")
        prov = cfg["providers"]["ica"]
        assert prov["name"] == "ICA"
        assert prov["package"] == launcher.OPENCODE_PACKAGE
        assert prov["settings"] == {"baseURL": "https://api.servicesessentials.ibm.com/v1", "apiKey": "{env:ICA_OPENCODE_API_KEY}"}
        assert prov["models"] == {"claude-sonnet-5": {"name": "claude-sonnet-5"}}
        assert "provider" not in cfg

    def test_ensure_provider_without_api_key(self, launcher):
        cfg = {}
        launcher._ensure_oc_provider(cfg, "ollama", "llama3:8b", "Ollama", "http://localhost:11434/v1")
        assert cfg["providers"]["ollama"]["settings"] == {"baseURL": "http://localhost:11434/v1"}

    def test_ensure_provider_migrates_legacy_entry(self, launcher):
        cfg = {
            "provider": {
                "ica": {"npm": "@ai-sdk/openai-compatible", "options": {"apiKey": "secret"}, "models": {"old-model": {"name": "Old"}}},
                "llamacpp": {"npm": "@ai-sdk/openai-compatible"},
            }
        }
        launcher._ensure_oc_provider(cfg, "ica", "new-model", "ICA", launcher.ICA_API_BASE, "{env:ICA_OPENCODE_API_KEY}")
        assert cfg["provider"] == {"llamacpp": {"npm": "@ai-sdk/openai-compatible"}}
        assert cfg["providers"]["ica"]["models"] == {"old-model": {"name": "Old"}, "new-model": {"name": "new-model"}}
        assert "secret" not in json.dumps(cfg)

    def test_ensure_provider_removes_empty_legacy_block(self, launcher):
        cfg = {"provider": {"ica": {"models": {}}}}
        launcher._ensure_oc_provider(cfg, "ica", "m", "ICA", launcher.ICA_API_BASE)
        assert "provider" not in cfg

    def test_ensure_provider_preserves_existing_v2_fields(self, launcher):
        cfg = {"providers": {"ica": {"name": "Custom", "settings": {"headers": {"X-A": "1"}}, "models": {"m": {"name": "Mine"}}}}}
        launcher._ensure_oc_provider(cfg, "ica", "m", "ICA", launcher.ICA_API_BASE)
        prov = cfg["providers"]["ica"]
        assert prov["name"] == "Custom"
        assert prov["settings"]["headers"] == {"X-A": "1"}
        assert prov["models"]["m"] == {"name": "Mine"}

    def test_set_model(self, launcher):
        cfg = {}
        launcher._set_oc_model(cfg, "litellm", "anthropic/claude-sonnet-4-5")
        assert cfg["model"] == "litellm/anthropic/claude-sonnet-4-5"

    def _spec(self, launcher, provider, **overrides):
        kwargs = {
            "litellm_url": "https://litellm.example.com/",
            "litellm_key": "litellm",
            "openrouter_key": "or-key",
            "ollama_url": "http://localhost:11434",
            "vllm_url": "https://vllm.example.com",
            "ica_key": "ica-key",
        }
        kwargs.update(overrides)
        return launcher._oc_provider_spec(provider, **kwargs)

    def test_spec_litellm_uses_env_reference_when_set(self, launcher, monkeypatch):
        monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-real")
        assert self._spec(launcher, "litellm") == ("LiteLLM", "https://litellm.example.com/v1", "{env:LITELLM_MASTER_KEY}")

    def test_spec_litellm_falls_back_to_default_key(self, launcher, monkeypatch):
        monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
        assert self._spec(launcher, "litellm") == ("LiteLLM", "https://litellm.example.com/v1", "litellm")

    def test_spec_ollama(self, launcher):
        assert self._spec(launcher, "ollama") == ("Ollama", "http://localhost:11434/v1", None)

    def test_spec_openrouter(self, launcher):
        assert self._spec(launcher, "openrouter") == ("OpenRouter", launcher.OPENROUTER_API_BASE, "{env:OPENROUTER_API_KEY}")
        assert self._spec(launcher, "openrouter", openrouter_key="")[2] is None

    def test_spec_vllm(self, launcher):
        assert self._spec(launcher, "vllm") == ("vLLM", "https://vllm.example.com/v1", "dummy")

    def test_spec_vllm_exits_without_url(self, launcher):
        with pytest.raises(SystemExit):
            self._spec(launcher, "vllm", vllm_url="")

    def test_spec_ica(self, launcher):
        assert self._spec(launcher, "ica") == ("ICA", "https://api.servicesessentials.ibm.com/v1", "{env:ICA_OPENCODE_API_KEY}")
        assert self._spec(launcher, "ica", ica_key="")[2] is None

    def test_spec_unknown_provider(self, launcher):
        assert self._spec(launcher, "nvidia") is None


# ── launch (dry-run) ──────────────────────────────────────────────────────────


class TestLaunch:
    def _base_kwargs(self, launcher, tmp_path, monkeypatch):
        p = tmp_path / "opencode.json"
        monkeypatch.setattr(launcher, "OPENCODE_CONFIG_PATH", p)
        monkeypatch.setattr(launcher, "HISTORY_FILE", tmp_path / "history.json")
        monkeypatch.setattr(launcher, "CACHE_FILE", tmp_path / "cache.json")
        monkeypatch.setattr(launcher, "LAUNCHER_DIR", tmp_path)
        return {
            "litellm_url": "https://litellm.example.com",
            "litellm_key": "key",
            "openrouter_key": "",
            "ollama_url": "http://localhost:11434",
            "vllm_url": "https://vllm.example.com",
            "ica_key": "",
            "dry_run": True,
            "extra_args": [],
        }

    def test_dry_run_litellm_does_not_exec(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        with patch("subprocess.run") as mock_run:
            launcher.launch("litellm", "gpt-4o", **kwargs)
        mock_run.assert_not_called()

    def test_dry_run_ollama_does_not_exec(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        with patch("subprocess.run") as mock_run:
            launcher.launch("ollama", "llama3:8b", **kwargs)
        mock_run.assert_not_called()

    def test_dry_run_vllm_does_not_exec(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        with patch("subprocess.run") as mock_run:
            launcher.launch("vllm", "local-model", **kwargs)
        mock_run.assert_not_called()

    def test_dry_run_openrouter_does_not_exec(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        kwargs["openrouter_key"] = "or-key"
        with patch("subprocess.run") as mock_run:
            launcher.launch("openrouter", "anthropic/claude-3-5-sonnet", **kwargs)
        mock_run.assert_not_called()

    def test_dry_run_ica_does_not_exec(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        kwargs["ica_key"] = "ica-key"
        with patch("subprocess.run") as mock_run:
            launcher.launch("ica", "ibm/granite-13b", **kwargs)
        mock_run.assert_not_called()

    def test_unsafe_model_id_exits(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        with pytest.raises(SystemExit):
            launcher.launch("litellm", "bad model; rm -rf /", **kwargs)

    def test_records_selection_on_dry_run(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        launcher.launch("litellm", "gpt-4o", **kwargs)
        history = launcher.load_history()
        assert history[0] == {"provider": "litellm", "model_id": "gpt-4o"}

    def test_unknown_provider_uses_passthrough(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        with patch("subprocess.run") as mock_run:
            launcher.launch("custom", "my-model", **kwargs)
        mock_run.assert_not_called()

    def test_opencode_not_found_exits(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        kwargs["dry_run"] = False
        with patch("subprocess.run", side_effect=FileNotFoundError), pytest.raises(SystemExit):
            launcher.launch("litellm", "gpt-4o", **kwargs)

    def test_live_run_exits_with_returncode(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        kwargs["dry_run"] = False
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result) as mock_run, pytest.raises(SystemExit) as exc:
            launcher.launch("litellm", "gpt-4o", **kwargs)
        assert exc.value.code == 0
        assert [c.args[0] for c in mock_run.call_args_list] == [["opencode", "reload"], ["opencode"]]

    def test_live_run_forwards_extra_args(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        kwargs["dry_run"] = False
        kwargs["extra_args"] = ["--continue"]
        with patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run, pytest.raises(SystemExit):
            launcher.launch("litellm", "gpt-4o", **kwargs)
        assert mock_run.call_args_list[-1].args[0] == ["opencode", "--continue"]

    def test_writes_default_model_and_provider(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        kwargs["ica_key"] = "ica-secret"
        launcher.launch("ica", "claude-sonnet-5", **kwargs)
        raw = (tmp_path / "opencode.json").read_text()
        cfg = json.loads(raw)
        assert cfg["model"] == "ica/claude-sonnet-5"
        assert cfg["providers"]["ica"]["settings"]["apiKey"] == "{env:ICA_OPENCODE_API_KEY}"
        assert "ica-secret" not in raw

    def test_passthrough_provider_only_sets_model(self, launcher, tmp_path, monkeypatch):
        kwargs = self._base_kwargs(launcher, tmp_path, monkeypatch)
        launcher.launch("nvidia", "meta/llama-3.1-70b-instruct", **kwargs)
        cfg = json.loads((tmp_path / "opencode.json").read_text())
        assert cfg == {"model": "nvidia/meta/llama-3.1-70b-instruct"}


# ── parse_args ────────────────────────────────────────────────────────────────


class TestParseArgs:
    def test_defaults(self, launcher, monkeypatch):
        monkeypatch.delenv("LITELLM_URL", raising=False)
        monkeypatch.delenv("VLLM_URL", raising=False)
        import sys

        with patch.object(sys, "argv", ["opencode-launcher.py"]):
            args = launcher.parse_args()
        assert args.dry_run is False
        assert args.no_cache is False
        assert args.clear_history is False
        assert args.clear_cache is False
        assert args.model is None
        assert args.timeout is None

    def test_dry_run_flag(self, launcher, monkeypatch):
        import sys

        with patch.object(sys, "argv", ["opencode-launcher.py", "--dry-run"]):
            args = launcher.parse_args()
        assert args.dry_run is True

    def test_no_cache_flag(self, launcher, monkeypatch):
        import sys

        with patch.object(sys, "argv", ["opencode-launcher.py", "--no-cache"]):
            args = launcher.parse_args()
        assert args.no_cache is True

    def test_model_flag(self, launcher):
        import sys

        with patch.object(sys, "argv", ["opencode-launcher.py", "--model", "litellm/gpt-4o"]):
            args = launcher.parse_args()
        assert args.model == "litellm/gpt-4o"

    def test_timeout_flag(self, launcher):
        import sys

        with patch.object(sys, "argv", ["opencode-launcher.py", "--timeout", "30"]):
            args = launcher.parse_args()
        assert args.timeout == 30

    def test_invalid_litellm_url_exits(self, launcher):
        import sys

        with patch.object(sys, "argv", ["opencode-launcher.py", "--litellm-url", "not-a-url"]), pytest.raises(SystemExit):
            launcher.parse_args()


# ── console helpers (err / ok / warn) ─────────────────────────────────────────


class TestConsoleHelpers:
    def test_err_prints(self, launcher, capsys):
        with patch.object(launcher.console, "print") as mock_print:
            launcher.err("something went wrong")
        mock_print.assert_called_once()

    def test_ok_prints(self, launcher):
        with patch.object(launcher.console, "print") as mock_print:
            launcher.ok("all good")
        mock_print.assert_called_once()

    def test_warn_prints(self, launcher):
        with patch.object(launcher.console, "print") as mock_print:
            launcher.warn("watch out")
        mock_print.assert_called_once()


# ── get_vllm_url ──────────────────────────────────────────────────────────────


class TestGetVllmUrl:
    def test_returns_env_var(self, launcher, monkeypatch):
        monkeypatch.setenv("VLLM_URL", "https://my-vllm.example.com")
        assert launcher.get_vllm_url() == "https://my-vllm.example.com"

    def test_returns_default(self, launcher, monkeypatch):
        monkeypatch.delenv("VLLM_URL", raising=False)
        assert launcher.get_vllm_url() == "http://localhost:8000"


# ── main() non-TUI branches ───────────────────────────────────────────────────


class TestMain:
    def _patch_argv(self, monkeypatch, args):
        import sys

        monkeypatch.setattr(sys, "argv", ["opencode-launcher.py"] + args)

    def test_clear_history_flag_exits(self, tmp_launcher, monkeypatch):
        self._patch_argv(monkeypatch, ["--clear-history"])
        with patch.object(tmp_launcher, "header"), pytest.raises(SystemExit) as exc:
            tmp_launcher.main()
        assert exc.value.code == 0
        assert not tmp_launcher.HISTORY_FILE.exists()

    def test_clear_cache_flag_exits(self, tmp_launcher, monkeypatch):
        tmp_launcher.save_cache({"key": "val"})
        self._patch_argv(monkeypatch, ["--clear-cache"])
        with patch.object(tmp_launcher, "header"), pytest.raises(SystemExit) as exc:
            tmp_launcher.main()
        assert exc.value.code == 0
        assert not tmp_launcher.CACHE_FILE.exists()

    def test_model_flag_no_model_id_exits(self, tmp_launcher, monkeypatch):
        self._patch_argv(monkeypatch, ["--model", "litellm"])
        with patch.object(tmp_launcher, "header"), pytest.raises(SystemExit):
            tmp_launcher.main()

    def test_model_flag_unknown_provider_exits(self, tmp_launcher, monkeypatch):
        self._patch_argv(monkeypatch, ["--model", "bogus/my-model"])
        with patch.object(tmp_launcher, "header"), pytest.raises(SystemExit):
            tmp_launcher.main()

    def test_model_flag_direct_launch_dry_run(self, tmp_launcher, monkeypatch, tmp_path):
        monkeypatch.setattr(tmp_launcher, "OPENCODE_CONFIG_PATH", tmp_path / "opencode.json")
        self._patch_argv(monkeypatch, ["--model", "litellm/gpt-4o", "--dry-run"])
        with patch.object(tmp_launcher, "header"), patch("subprocess.run") as mock_run:
            tmp_launcher.main()
        mock_run.assert_not_called()
