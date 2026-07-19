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

"""Guards for the Tinker backend wiring, the token-capture seam, and the
reward-variance fix.

The real Tinker path can only run on FarmShare (needs tinker/transformers), so
these tests cover the parts that must hold locally: the module conforms to the
PolicyBackend contract, the carrier fields exist, the token-capture seam rejects
stale/mismatched generations and is a no-op for name-string advisors, and the
``y_min`` knob and ``alpha_r`` guard behave.
"""

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
import threading

import numpy as np

import advantages
import llm_utils
import rl_rewards
import rl_trainer
import run_advisor_rl
import tinker_backend


def _config(y_min=None):
    reward = {"scale_c": 5.0, "alpha_r": 1.0}
    if y_min is not None:
        reward["y_min"] = y_min
    return {
        "evaluation": {
            "metric_direction": "max",
            "init_score": 0.601,
            "target_score": 0.667,
        },
        "rl": {"reward": reward},
    }


# --- reward shaping ---------------------------------------------------------

def test_from_config_defaults_y_min_to_baseline():
    cfg = rl_rewards.RewardShapingConfig.from_config(_config())
    assert cfg.y_min == 0.601  # min(init, target)
    assert cfg.y_max == 0.667


def test_from_config_reads_y_min_override_including_zero():
    cfg = rl_rewards.RewardShapingConfig.from_config(_config(y_min=0.0))
    assert cfg.y_min == 0.0  # 0.0 is a valid override, not treated as "unset"


def test_lowered_y_min_defeats_the_constant_reward_trap():
    # A group clustered at/below the 0.601 seed. With the default y_min=baseline
    # every candidate shapes to 0 -> constant group -> phase mix SKIPS (no
    # gradient). Lowering y_min below baseline keeps the rewards distinct.
    scores = [0.58, 0.595, 0.60, 0.601]
    default_cfg = rl_rewards.RewardShapingConfig.from_config(_config())
    lowered_cfg = rl_rewards.RewardShapingConfig.from_config(_config(y_min=0.0))

    default_rewards = rl_rewards.shape_rewards(scores, default_cfg)
    lowered_rewards = rl_rewards.shape_rewards(scores, lowered_cfg)

    assert np.all(default_rewards == 0.0)
    assert advantages.phase_adaptive_mix(
        default_rewards, k=2, alpha_t=0.0
    ).skip_update

    assert np.all(np.diff(lowered_rewards) > 0)
    lowered_mix = advantages.phase_adaptive_mix(lowered_rewards, k=2, alpha_t=0.0)
    assert not lowered_mix.skip_update
    assert lowered_mix.advantages is not None


def test_alpha_r_must_be_positive():
    # alpha_r=0 would make 0**0==1 hand the worst candidates the max reward.
    import pytest

    with pytest.raises(ValueError):
        rl_rewards.RewardShapingConfig(
            metric_direction="max", y_init=0.6, y_target=0.667, alpha_r=0.0
        )


# --- backend contract -------------------------------------------------------

def test_tinker_backend_conforms_to_policy_backend():
    # Module imports without tinker/transformers (heavy imports live in __init__).
    assert issubclass(
        tinker_backend.TinkerPolicyBackend, rl_trainer.PolicyBackend
    )
    for method in ("generate", "update", "sync_weights"):
        assert callable(getattr(tinker_backend.TinkerPolicyBackend, method))


def test_carrier_fields_exist_for_prompt_tokens():
    generation = rl_trainer.GenerationResult(text="x")
    sample = rl_trainer.RolloutSample(island_id=0)
    assert hasattr(generation, "prompt_token_ids")
    assert hasattr(sample, "prompt_token_ids")


# --- token-capture seam (the load-bearing wiring for gradient correctness) ---

class _FakeBackend:
    def __init__(self, generation):
        self.last_generation = generation


class _FakeAdvisor:
    def __init__(self, generation):
        self.backend = _FakeBackend(generation)


def _generation(text):
    return rl_trainer.GenerationResult(
        text=text,
        token_ids=np.array([11, 12, 13], dtype=np.int64),
        logprobs=np.array([-0.1, -0.2, -0.3]),
        prompt_token_ids=np.array([9, 8], dtype=np.int64),
    )


def test_capture_accepts_matching_generation():
    advisor = _FakeAdvisor(_generation("Idea ID: 2\nrationale"))
    tokens, logprobs, prompt = run_advisor_rl._capture_advisor_tokens(
        advisor, "Idea ID: 2\nrationale"
    )
    assert tokens is not None and logprobs is not None and prompt is not None
    assert list(tokens) == [11, 12, 13]


def test_capture_rejects_stale_mismatched_generation():
    # A leaked/timed-out generation whose text != the returned response must be
    # rejected so its tokens are not paired with the wrong reward.
    advisor = _FakeAdvisor(_generation("Idea ID: 5 leaked"))
    assert run_advisor_rl._capture_advisor_tokens(
        advisor, "Idea ID: 2 real"
    ) == (None, None, None)


def test_capture_none_when_slot_empty():
    advisor = _FakeAdvisor(None)
    assert run_advisor_rl._capture_advisor_tokens(advisor, "x") == (None, None, None)


def test_capture_is_noop_for_name_string_advisor():
    # Mock/baseline path: advisor is a model-name string with no `.backend`.
    assert run_advisor_rl._capture_advisor_tokens("qwen/qwen3-8b", "x") == (
        None,
        None,
        None,
    )


def test_reset_clears_slot_and_is_safe_for_strings():
    advisor = _FakeAdvisor(_generation("y"))
    run_advisor_rl._reset_advisor_capture(advisor)
    assert advisor.backend.last_generation is None
    run_advisor_rl._reset_advisor_capture("qwen/qwen3-8b")  # must not raise


def test_parallel_mock_capture_is_thread_local():
    barrier = threading.Barrier(2)

    class ThreadedMockBackend(rl_trainer.MockPolicyBackend):
        def generate(self, prompt, generation_config):
            del generation_config
            token = 11 if "alpha" in prompt else 22
            generation = rl_trainer.GenerationResult(
                text=prompt,
                token_ids=np.array([token], dtype=np.int64),
                logprobs=np.array([-token / 100.0]),
                prompt_token_ids=np.array([token + 100], dtype=np.int64),
            )
            self.last_generation = generation
            barrier.wait()
            return generation

    backend = ThreadedMockBackend({})
    advisor = rl_trainer.BackendLLMClient(backend, {"llm": {}})

    def generate_and_capture(label):
        transcript = llm_utils.Transcript()
        transcript.append(llm_utils.ContentChunk(label, "user"))
        raw = llm_utils.generate_completion(
            advisor,
            transcript,
            {"llm": {"max_try_count": 1}},
        )
        return run_advisor_rl._capture_advisor_tokens(advisor, raw)

    with ThreadPoolExecutor(max_workers=2) as executor:
        alpha, beta = executor.map(generate_and_capture, ("alpha", "beta"))

    assert alpha[0].tolist() == [11]
    assert alpha[2].tolist() == [111]
    assert beta[0].tolist() == [22]
    assert beta[2].tolist() == [122]


def test_backend_client_retries_timeout_in_caller_thread():
    caller_thread = threading.get_ident()

    class RetryBackend(rl_trainer.MockPolicyBackend):
        def __init__(self):
            super().__init__({})
            self.calls = []

        def generate(self, prompt, generation_config):
            self.calls.append(threading.get_ident())
            if len(self.calls) == 1:
                raise FuturesTimeoutError
            return super().generate(prompt, generation_config)

    backend = RetryBackend()
    advisor = rl_trainer.BackendLLMClient(backend, {"llm": {}})
    transcript = llm_utils.Transcript()
    transcript.append(llm_utils.ContentChunk("retry", "user"))

    raw = llm_utils.generate_completion(
        advisor,
        transcript,
        {"llm": {"max_try_count": 2, "request_timeout": 0.01}},
    )

    assert raw == "mock response"
    assert backend.calls == [caller_thread, caller_thread]
    assert run_advisor_rl._capture_advisor_tokens(advisor, raw)[0].tolist() == [
        0,
        1,
        2,
        3,
    ]


def test_clean_text_matches_generate_completion_postprocessing():
    assert run_advisor_rl._clean_text("  <start_of_turn>hi<end_of_turn>  ") == "hi"
    assert run_advisor_rl._clean_text(None) is None


def test_resolve_base_url_defaults_to_none(monkeypatch):
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    assert tinker_backend._resolve_base_url({}) is None


def test_resolve_base_url_reads_config_key(monkeypatch):
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    assert tinker_backend._resolve_base_url(
        {"tinker_base_url": "http://127.0.0.1:8000"}
    ) == "http://127.0.0.1:8000"


def test_resolve_base_url_env_overrides_config(monkeypatch):
    monkeypatch.setenv("TINKER_BASE_URL", "http://envhost:9")
    assert tinker_backend._resolve_base_url(
        {"tinker_base_url": "http://cfg:8"}
    ) == "http://envhost:9"


def test_resolve_base_url_blank_values_fall_through(monkeypatch):
    monkeypatch.setenv("TINKER_BASE_URL", "")
    assert tinker_backend._resolve_base_url({"tinker_base_url": ""}) is None
    assert tinker_backend._resolve_base_url(
        {"tinker_base_url": "http://cfg:8"}
    ) == "http://cfg:8"


def test_generation_timeout_preserves_generate_completion_deadline():
    config = {
        "llm": {"request_timeout": 200},
        "advisor_llm": {"request_timeout": 100},
    }
    assert tinker_backend._resolve_generation_timeout(config, 600) == 130
    assert tinker_backend._resolve_generation_timeout(config, 90) == 90


class _StubTok:
    """apply_chat_template return-shape stub (transformers version drift)."""

    def __init__(self, ret):
        self._ret = ret

    def apply_chat_template(self, *a, **k):
        return self._ret


def _render_with(ret):
    backend = tinker_backend.TinkerPolicyBackend.__new__(
        tinker_backend.TinkerPolicyBackend
    )
    backend._hf_tok = _StubTok(ret)
    backend.enable_thinking = True
    return backend._render_ids("hi")


def test_render_ids_plain_list():
    assert _render_with([6, 7]) == [6, 7]


def test_render_ids_dict_return():
    # BatchEncoding-style: iterating yields keys -> int('input_ids') crashed.
    assert _render_with({"input_ids": [1, 2, 3]}) == [1, 2, 3]


def test_render_ids_batched_return():
    assert _render_with([[4, 5]]) == [4, 5]


class _StubTensor:
    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


class _StubFBResult:
    def __init__(self, metrics=None, loss_fn_outputs=None):
        self.metrics = metrics if metrics is not None else {}
        self.loss_fn_outputs = loss_fn_outputs or []


def test_fb_loss_prefers_managed_tinker_metrics():
    fb = _StubFBResult(metrics={"loss": 1.25})
    assert tinker_backend._resolve_fb_loss(fb) == 1.25


def test_fb_loss_aggregates_skyrl_elementwise_outputs():
    # skyrl-tx: metrics={} and per-token losses in loss_fn_outputs.
    fb = _StubFBResult(
        loss_fn_outputs=[
            {"elementwise_loss": _StubTensor([1.0, 2.0])},
            {"elementwise_loss": {"data": [3.0]}},
        ]
    )
    assert tinker_backend._resolve_fb_loss(fb) == 2.0


def test_fb_loss_nan_when_no_signal():
    import math

    assert math.isnan(tinker_backend._resolve_fb_loss(_StubFBResult()))
