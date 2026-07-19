# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""get_llm_client must route local / OpenAI-compatible small models to the
OllamaClient so PACEvolve++ can drive an advisor served on a local GPU."""

from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from types import SimpleNamespace

import llm_utils


class _StubClient(llm_utils.LLMClient):
    def count_tokens(self, text):
        return 1

    def generate(self, prompt, generation_config):
        return "stub"


def test_routes_ollama_by_client_type(monkeypatch):
    monkeypatch.setattr(llm_utils, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(llm_utils, "OllamaClient", _StubClient, raising=False)
    monkeypatch.setattr(llm_utils, "_CLIENT_CACHE", {})
    client = llm_utils.get_llm_client(
        "gpt-oss-20b", {"llm": {"client_type": "ollama", "name": "gpt-oss-20b"}})
    assert isinstance(client, _StubClient)


def test_routes_ollama_by_name(monkeypatch):
    monkeypatch.setattr(llm_utils, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(llm_utils, "OllamaClient", _StubClient, raising=False)
    monkeypatch.setattr(llm_utils, "_CLIENT_CACHE", {})
    client = llm_utils.get_llm_client("ollama-qwen", {"llm": {"name": "ollama-qwen"}})
    assert isinstance(client, _StubClient)


def test_same_name_different_base_url_not_cached_together(monkeypatch):
    # Two roles sharing a model name but different endpoints must get distinct clients.
    monkeypatch.setattr(llm_utils, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(llm_utils, "OllamaClient", _StubClient, raising=False)
    monkeypatch.setattr(llm_utils, "_CLIENT_CACHE", {})
    a = llm_utils.get_llm_client("qwen", {"llm": {"client_type": "ollama", "name": "qwen",
                                                  "base_url": "http://a:8000/v1"}})
    b = llm_utils.get_llm_client("qwen", {"llm": {"client_type": "ollama", "name": "qwen",
                                                  "base_url": "http://b:8000/v1"}})
    assert a is not b


def test_same_name_different_client_type_not_cached_together(monkeypatch):
    monkeypatch.setattr(llm_utils, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(llm_utils, "OpenAIClient", _StubClient, raising=False)
    monkeypatch.setattr(llm_utils, "OllamaClient", _StubClient, raising=False)
    monkeypatch.setattr(llm_utils, "_CLIENT_CACHE", {})
    a = llm_utils.get_llm_client(
        "neutral-model", {"llm": {"client_type": "openai", "name": "neutral-model"}})
    b = llm_utils.get_llm_client(
        "neutral-model", {"llm": {"client_type": "ollama", "name": "neutral-model"}})
    assert a is not b


def test_gemini_name_still_routes_gemini(monkeypatch):
    monkeypatch.setattr(llm_utils, "GEMINI_AVAILABLE", True)
    monkeypatch.setattr(llm_utils, "GeminiClient", _StubClient, raising=False)
    monkeypatch.setattr(llm_utils, "_CLIENT_CACHE", {})
    client = llm_utils.get_llm_client("gemini-2.5-pro", {"llm": {"name": "gemini-2.5-pro"}})
    assert isinstance(client, _StubClient)


def _openrouter_client(monkeypatch, response, config=None):
    calls = []

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return response

    fake_openai = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    monkeypatch.setattr(llm_utils, "OpenAI", lambda **kwargs: fake_openai)
    client_config = {"name": "google/gemini-3.1-pro-preview"}
    client_config.update(config or {})
    return llm_utils.OpenRouterClient(client_config), calls


def test_openrouter_requests_usage_provider_order_and_logs_cache(
    monkeypatch, caplog
):
    response = SimpleNamespace(
        model="google/gemini-3.1-pro-preview",
        usage={
            "prompt_tokens": 1200,
            "completion_tokens": 45,
            "prompt_tokens_details": {"cached_tokens": 900},
        },
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
    )
    monkeypatch.setenv(
        "OPENROUTER_PROVIDER_ORDER",
        " google-vertex, google-ai-studio , ",
    )
    client, calls = _openrouter_client(
        monkeypatch, response, {"reasoning": {"effort": "low"}}
    )

    with caplog.at_level(logging.INFO, logger="controller"):
        assert client.generate("prompt", {}) == "done"

    assert calls[0]["extra_body"] == {
        "usage": {"include": True},
        "reasoning": {"effort": "low"},
        "provider": {
            "order": ["google-vertex", "google-ai-studio"],
            "allow_fallbacks": True,
        },
    }
    assert (
        "OpenRouter usage: model=google/gemini-3.1-pro-preview "
        "prompt_tokens=1200 completion_tokens=45 cached_tokens=900"
        in caplog.text
    )


def test_openrouter_omits_provider_when_order_is_unset(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
    )
    monkeypatch.delenv("OPENROUTER_PROVIDER_ORDER", raising=False)
    client, calls = _openrouter_client(monkeypatch, response)

    assert client.generate("prompt", {}) == "done"
    assert calls[0]["extra_body"] == {"usage": {"include": True}}


def test_transcript_file_writes_are_serialized_across_instances(monkeypatch):
    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    class SlowFile:
        def __enter__(self):
            return self

        def __exit__(self, *unused_args):
            return False

        def write(self, content):
            del content
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(
                    state["max_active"], state["active"]
                )
            time.sleep(0.01)
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: SlowFile())
    transcripts = [llm_utils.Transcript("shared.jsonl") for _ in range(8)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda transcript: transcript.append(
                    llm_utils.ContentChunk("entry", "user")
                ),
                transcripts,
            )
        )

    assert state["max_active"] == 1
