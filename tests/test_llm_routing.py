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
