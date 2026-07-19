# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping, MutableSequence
import dataclasses
import json
import re
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Union

# --- Logger Setup ---
logger = logging.getLogger("controller")

# Shared pool for hard-timeouting LLM calls. OpenRouter can trickle keepalive bytes
# that reset httpx's read-timeout, so the SDK-level timeout may never fire on a slow
# reasoning generation. We run each call here and enforce a wall-clock deadline via
# future.result(timeout=...). Sized well above the driver's worker count so a few
# leaked (timed-out but still-running) calls don't starve the pool.
import concurrent.futures as _futures
_LLM_EXECUTOR = _futures.ThreadPoolExecutor(max_workers=96, thread_name_prefix="llmcall")
_TRANSCRIPT_FILE_LOCK = threading.Lock()

@dataclasses.dataclass
class ContentChunk:
    content: str
    role: str
    tags: list[str] = dataclasses.field(default_factory=list)
    hidden: bool = False

    def format(self):
        return f"<start_of_turn>{self.role}\n{self.content}<end_of_turn>\n"


class Transcript(MutableSequence):
    def __init__(self, log_filename: str | None = None):
        self._log_filename = log_filename
        self._list = list()

    def _log_to_file(self, v):
        if isinstance(v, ContentChunk) and self._log_filename:
            json_str = dataclasses.asdict(v)
            with _TRANSCRIPT_FILE_LOCK:
                with open(self._log_filename, 'a') as f:
                    f.write(json.dumps(json_str) + "\n")

    def hide_by_tag(self, tags: list[str]):
        for chunk in self._list:
            if any(tag in chunk.tags for tag in tags):
                chunk.hidden = True

    def unhide_all_tags(self):
        for chunk in self._list:
            chunk.hidden = False

    def format(self):
        output_text = []
        for chunk in self._list:
            if chunk.hidden:
                continue
            output_text.append(chunk.format())
        return "".join(output_text)

    def log_debug_message(self, message: str):
        c = ContentChunk(
            content=message,
            role="info",
            tags=["debug_message"]
        )
        self._log_to_file(c)

    def __len__(self): return len(self._list)
    def __getitem__(self, i): return self._list[i]
    def __delitem__(self, i): del self._list[i]
    def __str__(self): return str(self._list)
    def __setitem__(self, i, v):
        self._list[i] = v

    def insert(self, i, v):
        if i == len(self._list):
            self._log_to_file(v)
        self._list.insert(i, v)

# --- LLM Abstraction ---
class LLMClient(ABC):
    """Abstract Base Class for Language Model Clients."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass

    @abstractmethod
    def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
        pass

# --- Gemini Client ---
try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

if GEMINI_AVAILABLE:
    class GeminiClient(LLMClient):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "gemini-3-pro-preview")
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found in environment.")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)

        def count_tokens(self, text: str) -> int:
            return self.model.count_tokens(text).total_tokens

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            # Map common keys to Gemini specific keys
            gen_config = genai_types.GenerationConfig(**generation_config)
            response = self.model.generate_content(prompt, generation_config=gen_config)
            return response.text

# --- OpenAI Client ---
try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if OPENAI_AVAILABLE:
    class OpenAIClient(LLMClient):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "gpt-5.2-pro")
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        def count_tokens(self, text: str) -> int:
            """Returns the number of tokens in a text string."""
            encoding = tiktoken.encoding_for_model(self.model_name)
            num_tokens = len(encoding.encode(text))
            return num_tokens

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            # Convert generation_config to OpenAI format
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": generation_config.get("temperature", 1.0),
                "top_p": generation_config.get("top_p", 0.95),
                "max_tokens": generation_config.get("max_output_tokens", 4096)
            }
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
    
    
    class OllamaClient(LLMClient):
        """Ollama local LLM via OpenAI-compatible API (http://localhost:11434/v1)."""

        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "llama3.2")
            base_url = self.config.get("base_url", "http://localhost:11434/v1")
            self.client = OpenAI(
                base_url=base_url,
                api_key=self.config.get("api_key", "ollama"),
            )

        def count_tokens(self, text: str) -> int:
            """Approximate token count (~4 chars/token for most models)."""
            return max(1, len(text) // 4)

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=generation_config.get("temperature", 1.0),
                max_tokens=generation_config.get("max_output_tokens", 16384),
            )
            return response.choices[0].message.content or ""


    class OpenRouterClient(LLMClient):
        """Frontier open-weight models via OpenRouter's OpenAI-compatible API.

        Reads the key from `api_key` in the llm config or the OPENROUTER_API_KEY
        env var; base URL from `base_url`/OPENROUTER_BASE_URL. OpenRouter returns
        any chain-of-thought in `message.reasoning` and the final answer in
        `message.content`, so returning `.content` gives the clean fenced code
        block / Idea-ID line the pipeline parses (reasoning is stripped for free).
        """

        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name")
            base_url = self.config.get("base_url") or os.environ.get(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
            api_key = self.config.get("api_key") or os.environ.get(
                "OPENROUTER_API_KEY"
            )
            if not api_key:
                logger.warning("OPENROUTER_API_KEY not found in environment.")
            # Per-request timeout so a slow/hung reasoning generation fails fast
            # instead of stalling a worker; the caller's retry loop then moves on.
            self.timeout = float(self.config.get("request_timeout", 240))
            self.client = OpenAI(
                base_url=base_url, api_key=api_key,
                timeout=self.timeout, max_retries=1,
            )
            # Optional provider-agnostic reasoning control, e.g. {"effort": "low"}
            # or {"max_tokens": 4000}. Passed through OpenRouter's `reasoning` field.
            self.reasoning = self.config.get("reasoning")
            self.extra_headers = {
                "HTTP-Referer": self.config.get(
                    "referer", "https://github.com/yudduy/pacevolve-pp"
                ),
                "X-Title": self.config.get("title", "PACEvolve++ RFG"),
            }

        def count_tokens(self, text: str) -> int:
            # OpenRouter model names aren't in tiktoken's registry; approximate
            # at ~4 chars/token (same convention as OllamaClient).
            return max(1, len(text) // 4)

        @staticmethod
        def _field(value, name):
            """Read SDK objects, dictionaries, and Pydantic extra fields."""
            if value is None:
                return None
            if isinstance(value, Mapping):
                return value.get(name)
            field_value = getattr(value, name, None)
            if field_value is not None:
                return field_value
            model_extra = getattr(value, "model_extra", None)
            if isinstance(model_extra, Mapping):
                return model_extra.get(name)
            return None

        def _log_usage(self, response) -> None:
            usage = self._field(response, "usage")
            prompt_tokens = self._field(usage, "prompt_tokens")
            completion_tokens = self._field(usage, "completion_tokens")
            details = self._field(usage, "prompt_tokens_details")
            cache_fields = []
            for name in (
                "cached_tokens",
                "cache_read_tokens",
                "cache_read_input_tokens",
            ):
                value = self._field(details, name)
                if value is None:
                    value = self._field(usage, name)
                if value is not None:
                    cache_fields.append(f"{name}={value}")
            cache_suffix = f" {' '.join(cache_fields)}" if cache_fields else ""
            logger.info(
                "OpenRouter usage: model=%s prompt_tokens=%s "
                "completion_tokens=%s%s",
                self._field(response, "model") or self.model_name,
                prompt_tokens,
                completion_tokens,
                cache_suffix,
            )

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": generation_config.get("temperature", 1.0),
                "top_p": generation_config.get("top_p", 0.95),
                "max_tokens": generation_config.get("max_output_tokens", 8192),
                "extra_headers": self.extra_headers,
            }
            extra_body = {"usage": {"include": True}}
            if self.reasoning is not None:
                extra_body["reasoning"] = self.reasoning
            provider_order = [
                provider.strip()
                for provider in os.environ.get(
                    "OPENROUTER_PROVIDER_ORDER", ""
                ).split(",")
                if provider.strip()
            ]
            if provider_order:
                extra_body["provider"] = {
                    "order": provider_order,
                    "allow_fallbacks": True,
                }
            params["extra_body"] = extra_body
            response = self.client.chat.completions.create(**params)
            self._log_usage(response)
            return response.choices[0].message.content or ""

# --- Anthropic Client ---
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

if ANTHROPIC_AVAILABLE:
    class AnthropicClient(LLMClient):
        def __init__(self, config: Dict[str, Any]):
            super().__init__(config)
            self.model_name = self.config.get("name", "claude-opus-4-6")
            self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def count_tokens(self, text: str) -> int:
            """
            Calls the Anthropic API to get a precise token count for the input string.
            """
            response = self.client.messages.count_tokens(
                model=self.model_name,
                messages=[{"role": "user", "content": text}]
            )
            return response.input_tokens

        def generate(self, prompt: str, generation_config: Dict[str, Any]) -> str:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=generation_config.get("max_output_tokens", 4096),
                temperature=generation_config.get("temperature", 1.0),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

# --- Client Factory and Cache ---

_CLIENT_CACHE: Dict[tuple, LLMClient] = {}

def get_llm_client(llm_name: str, config: Dict[str, Any]) -> LLMClient:
    """Singleton pattern to retrieve or create LLM clients."""
    llm_config = config.get('llm', {})
    # Cache by (name, base_url, client_type) so two roles that share a model
    # name don't collide when either their endpoint or client type differs.
    cache_key = (
        llm_name,
        llm_config.get('base_url'),
        llm_config.get('client_type'),
    )
    if cache_key in _CLIENT_CACHE:
        return _CLIENT_CACHE[cache_key]

    client_type = llm_config.get('client_type', 'gemini')
    
    # Override client_type based on model name if user just switched the name.
    # Explicit local endpoints may serve models with arbitrary provider names.
    if llm_config.get('client_type') not in ('ollama', 'openrouter'):
        if 'gpt' in llm_name or 'openai' in llm_name:
            client_type = 'openai'
        elif 'claude' in llm_name or 'anthropic' in llm_name:
            client_type = 'anthropic'
        elif 'gemini' in llm_name:
            client_type = 'gemini'
        elif 'ollama' in llm_name:
            client_type = 'ollama'

    client = None
    if client_type == 'gemini' and GEMINI_AVAILABLE:
        client = GeminiClient(llm_config)
    elif client_type == 'openai' and OPENAI_AVAILABLE:
        client = OpenAIClient(llm_config)
    elif client_type == 'anthropic' and ANTHROPIC_AVAILABLE:
        client = AnthropicClient(llm_config)
    elif client_type == 'ollama' and OPENAI_AVAILABLE:
        # OpenAI-compatible local endpoint (e.g. Ollama or a vLLM server); the
        # base_url is read from the llm config, enabling small-model advisors.
        client = OllamaClient(llm_config)
    elif client_type == 'openrouter' and OPENAI_AVAILABLE:
        # Frontier open-weight models via OpenRouter's OpenAI-compatible API.
        client = OpenRouterClient(llm_config)
    else:
        raise ValueError(f"Unsupported or missing client type: {client_type}")

    _CLIENT_CACHE[cache_key] = client
    logger.info(f"Initialized LLM Client: {client_type} for model {llm_name}")
    return client

# --- Completion Function ---

def generate_completion(
    llm_name_or_client: Union[str, LLMClient],
    transcript: Transcript,
    config: dict,
):
    """
    Generates a completion.
    Accepts either a string (model name) or an instantiated LLMClient.
    If a string is passed, it looks up/creates the client using the config.
    """
    
    # 1. Resolve the Client
    if isinstance(llm_name_or_client, str):
        llm_client = get_llm_client(llm_name_or_client, config)
    else:
        llm_client = llm_name_or_client

    prompt = transcript.format()
    llm_config = config.get('llm', {})

    # 2. Token Counting (Safe)
    try:
        token_count = llm_client.count_tokens(prompt)
    except Exception as e:
        logger.warning(f"Could not count tokens: {e}")
        token_count = 0

    # 3. Truncation Logic
    max_context = llm_config.get('context_size', 100000) # Default high for modern models
    if token_count > max_context:
        logger.warning(f"Prompt too long ({token_count}), truncating...")
        
    final_prompt = prompt 

    generation_config = {
        "temperature": llm_config.get("temperature", 1.0),
        "top_p": llm_config.get("top_p", 0.95),
        "max_output_tokens": llm_config.get("max_output_tokens", 4096)
    }

    # 5. Execution with Retry + hard wall-clock timeout per call
    max_tries = llm_config.get('max_try_count', 3)
    hard_timeout = float(llm_config.get('request_timeout', 240)) + 30
    output_text = None

    for i in range(max_tries):
        try:
            if getattr(llm_client, "generate_in_caller_thread", False):
                # BackendLLMClient must generate and expose last_generation in
                # this rollout thread so its thread-local token capture is read
                # by the matching sample. Tinker enforces its own call timeout.
                output_text = llm_client.generate(final_prompt, generation_config)
            else:
                fut = _LLM_EXECUTOR.submit(
                    llm_client.generate, final_prompt, generation_config
                )
                output_text = fut.result(timeout=hard_timeout)
            break
        except _futures.TimeoutError:
            logger.error(
                f"LLM generate hard-timeout ({hard_timeout:.0f}s) on try "
                f"{i + 1}/{max_tries}"
            )
            if i == max_tries - 1:
                return None
        except Exception as e:
            logger.error(f"LLM generate call failed on try {i + 1}/{max_tries}: {e}")
            if i == max_tries - 1:
                return None

    if output_text is None:
        return None

    # Cleanup tags
    output_text = output_text.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()
    return output_text


def extract_code_blocks(markdown_string: str) -> list[str]:
    if not markdown_string:
        return []
    code_block_pattern = re.compile(
        r'^[ \t]*(?P<fence>`{3,})(?:[a-zA-Z0-9_+\.-]+)?[ \t]*\r?\n'
        r'(?P<code>.*?)'
        r'\r?\n[ \t]*(?P=fence)[ \t]*$',
        re.DOTALL | re.MULTILINE,
    )
    return [
        match.group('code')
        for match in code_block_pattern.finditer(markdown_string)
    ]
