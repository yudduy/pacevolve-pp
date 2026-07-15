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

"""Spec for workflows/rl_trainer.py — rollout dataclasses, mock policy backend,
and the AdvisorTrainer that maps rewards to phase-adaptive advantages and drives
one gradient step + weight sync per rollout (skipping when a branch collapses)."""

import numpy as np
import pytest

import advantages as A
from rl_trainer import (
    AdvisorTrainer,
    AdvisorTrainerConfig,
    BackendLLMClient,
    GenerationResult,
    MockPolicyBackend,
    RolloutGroup,
    RolloutSample,
    TrainStepResult,
)


def _group(rewards, step=0, tokens=2):
    samples = [
        RolloutSample(island_id=i, reward=float(r), response_mask=np.ones(tokens))
        for i, r in enumerate(rewards)
    ]
    return RolloutGroup(step=step, samples=samples)


def test_rollout_group_rewards():
    np.testing.assert_allclose(_group([1.0, 2.0, 3.0]).rewards(), [1.0, 2.0, 3.0])


def test_train_step_updates_and_syncs():
    be = MockPolicyBackend({})
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="pacevolve++", top_k=4, total_steps=1000), be)
    res = tr.train_step(_group([0, 1, 2, 3, 4, 5, 6, 7], step=0))
    assert isinstance(res, TrainStepResult)
    assert not res.skipped
    assert len(be.update_calls) == 1
    assert be.sync_calls == 1
    assert res.advantages.shape == (8,)
    assert np.isfinite(res.metrics["loss"])
    assert res.alpha == pytest.approx(A.alpha_schedule(0, 1000))


def test_train_step_skips_on_constant_rewards():
    be = MockPolicyBackend({})
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="pacevolve++", top_k=2, total_steps=10), be)
    res = tr.train_step(_group([5, 5, 5, 5], step=3))
    assert res.skipped
    assert res.advantages is None
    assert len(be.update_calls) == 0
    assert be.sync_calls == 0


def test_objective_dispatch_grpo():
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="grpo", total_steps=100, eps_num=1e-8),
                        MockPolicyBackend({}))
    R = np.arange(8.0)
    np.testing.assert_allclose(tr.compute_advantages(R, 5).advantages, A.grpo_advantage(R, 1e-8))


def test_objective_dispatch_entropic():
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="entropic", entropic_gamma=0.4, total_steps=100),
                        MockPolicyBackend({}))
    R = np.arange(8.0)
    np.testing.assert_allclose(tr.compute_advantages(R, 5).advantages, A.entropic_advantage(R, 0.4))


def test_objective_dispatch_maxk_is_alpha_one():
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="maxk", top_k=4, total_steps=100),
                        MockPolicyBackend({}))
    R = np.arange(8.0)
    ref = A.phase_adaptive_mix(R, 4, 1.0)
    np.testing.assert_allclose(tr.compute_advantages(R, 5).advantages, ref.advantages)


def test_objective_dispatch_pace_uses_alpha_schedule():
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="pacevolve++", top_k=4, total_steps=100),
                        MockPolicyBackend({}))
    R = np.arange(8.0)
    ref = A.phase_adaptive_mix(R, 4, A.alpha_schedule(50, 100))
    np.testing.assert_allclose(tr.compute_advantages(R, 50).advantages, ref.advantages)


def test_objective_none_skips_without_update():
    be = MockPolicyBackend({})
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="none", total_steps=10), be)
    res = tr.train_step(_group([0, 1, 2, 3], step=0))
    assert res.skipped
    assert len(be.update_calls) == 0 and be.sync_calls == 0


def test_grpo_skips_on_constant_rewards():
    be = MockPolicyBackend({})
    tr = AdvisorTrainer(AdvisorTrainerConfig(objective="grpo", total_steps=10), be)
    res = tr.train_step(_group([2, 2, 2, 2], step=0))
    assert res.skipped
    assert len(be.update_calls) == 0


def test_config_from_config():
    config = {"rl": {"objective": "maxk", "n_samples": 16, "top_k": 8, "total_steps": 500,
                     "eps_num": 1e-7, "eps_skip": 1e-5, "entropic_gamma": 0.5,
                     "clip_eps_lo": 0.1, "clip_eps_hi": 0.3}}
    cfg = AdvisorTrainerConfig.from_config(config)
    assert cfg.objective == "maxk" and cfg.n_samples == 16 and cfg.top_k == 8
    assert cfg.total_steps == 500 and cfg.entropic_gamma == 0.5
    assert cfg.clip.eps_lo == 0.1 and cfg.clip.eps_hi == 0.3


def test_config_from_none_rl_section_uses_defaults():
    cfg = AdvisorTrainerConfig.from_config({"rl": None})
    assert cfg == AdvisorTrainerConfig()


def test_config_from_config_coerces_yaml_numeric_strings():
    config = {
        "rl": {
            "n_samples": "8",
            "top_k": "4",
            "total_steps": "1000",
            "eps_num": "1e-8",
            "eps_skip": "1e-6",
            "entropic_gamma": "0.5",
            "clip_eps_lo": "0.2",
            "clip_eps_hi": "0.28",
        }
    }
    cfg = AdvisorTrainerConfig.from_config(config)
    assert (cfg.n_samples, cfg.top_k, cfg.total_steps) == (8, 4, 1000)
    assert cfg.eps_num == pytest.approx(1e-8)
    assert cfg.eps_skip == pytest.approx(1e-6)
    assert cfg.entropic_gamma == pytest.approx(0.5)
    assert cfg.clip.eps_lo == pytest.approx(0.2)
    assert cfg.clip.eps_hi == pytest.approx(0.28)
    assert isinstance(cfg.eps_num, float)


def test_bad_objective_raises_at_construction():
    with pytest.raises(ValueError, match="objective must be one of"):
        AdvisorTrainer(
            AdvisorTrainerConfig(objective="unknown"),
            MockPolicyBackend({}),
        )


def test_oversized_subset_enumeration_raises_at_construction():
    with pytest.raises(ValueError, match="lower n_samples or top_k"):
        AdvisorTrainer(
            AdvisorTrainerConfig(
                objective="pacevolve++", n_samples=24, top_k=12
            ),
            MockPolicyBackend({}),
        )


def test_backend_llm_client_generates_and_counts():
    class Echo(MockPolicyBackend):
        def generate(self, prompt, generation_config):
            return GenerationResult(text="ECHO:" + prompt[:3], token_ids=np.arange(4),
                                    logprobs=np.zeros(4))

    client = BackendLLMClient(Echo({}), {"llm": {}})
    assert client.generate("hello world", {}).startswith("ECHO:")
    assert client.count_tokens("abcdefgh") >= 1
