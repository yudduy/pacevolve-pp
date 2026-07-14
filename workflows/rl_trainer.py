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

"""Policy backend and trainer scaffold for PACEvolve++."""

import abc
import dataclasses
import logging

import numpy as np

import advantages
import llm_utils
import rl_loss


logger = logging.getLogger("controller")


@dataclasses.dataclass
class GenerationResult:
    text: str
    token_ids: np.ndarray | None = None
    logprobs: np.ndarray | None = None


@dataclasses.dataclass
class RolloutSample:
    island_id: int
    prompt_text: str = ""
    response_text: str = ""
    token_ids: np.ndarray | None = None
    old_logprobs: np.ndarray | None = None
    response_mask: np.ndarray | None = None
    idea_id: int = -1
    exp_description: str = ""
    raw_score: float | None = None
    reward: float = 0.0
    eval_success: bool = False


@dataclasses.dataclass
class RolloutGroup:
    step: int
    samples: list

    def rewards(self) -> np.ndarray:
        return np.array([sample.reward for sample in self.samples], dtype=float)


@dataclasses.dataclass
class TrainStepResult:
    step: int
    skipped: bool
    skip_reason: str
    alpha: float
    advantages: np.ndarray | None
    metrics: dict


class PolicyBackend(abc.ABC):
    def __init__(self, config: dict):
        self.config = config

    @abc.abstractmethod
    def generate(
        self, prompt: str, generation_config: dict
    ) -> GenerationResult:
        pass

    @abc.abstractmethod
    def update(
        self,
        group: RolloutGroup,
        advantages: np.ndarray,
        clip,
    ) -> dict:
        pass

    @abc.abstractmethod
    def sync_weights(self) -> None:
        pass


class MockPolicyBackend(PolicyBackend):
    """Deterministic in-memory policy backend for tests."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.update_calls: list[tuple[int, np.ndarray]] = []
        self.sync_calls = 0
        self.last_advantages: np.ndarray | None = None

    def generate(
        self, prompt: str, generation_config: dict
    ) -> GenerationResult:
        del prompt, generation_config
        num_tokens = 4
        return GenerationResult(
            text="mock response",
            token_ids=np.arange(num_tokens),
            logprobs=np.zeros(num_tokens),
        )

    def update(
        self,
        group: RolloutGroup,
        advantages: np.ndarray,
        clip,
    ) -> dict:
        masks = [
            np.asarray(sample.response_mask, dtype=float).reshape(-1)
            if sample.response_mask is not None
            else np.ones(1, dtype=float)
            for sample in group.samples
        ]
        max_length = max((len(mask) for mask in masks), default=1)
        mask = np.zeros((len(masks), max_length), dtype=float)
        for row, sample_mask in enumerate(masks):
            mask[row, : len(sample_mask)] = sample_mask

        ratios = np.ones_like(mask)
        result = rl_loss.clipped_surrogate_loss(
            ratios, advantages, mask, clip
        )
        advantages_copy = np.asarray(advantages, dtype=float).copy()
        self.update_calls.append((group.step, advantages_copy))
        self.last_advantages = advantages_copy
        return {
            "loss": result.loss,
            "num_valid_tokens": result.num_valid_tokens,
            "clip_fraction": result.clip_fraction,
        }

    def sync_weights(self) -> None:
        self.sync_calls += 1


class BackendLLMClient(llm_utils.LLMClient):
    """Adapt a policy backend to the advisor's LLM client interface."""

    def __init__(self, backend: PolicyBackend, config: dict):
        super().__init__(config)
        self.backend = backend

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def generate(self, prompt: str, generation_config: dict) -> str:
        return self.backend.generate(prompt, generation_config).text


try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class TorchPolicyBackend(PolicyBackend):
        """Real policy-gradient backend.

        AdamW uses lr=1e-6, wd=0.1, and betas=(0.9, 0.98), with one gradient
        step per rollout. ``update`` mirrors
        ``rl_loss.clipped_surrogate_loss`` on token log probabilities. This is
        not implemented in the scaffold.
        """

        def generate(self, prompt, generation_config):
            raise NotImplementedError(
                "TorchPolicyBackend is a documented seam"
            )

        def update(self, group, advantages, clip):
            raise NotImplementedError(
                "TorchPolicyBackend is a documented seam"
            )

        def sync_weights(self):
            raise NotImplementedError(
                "TorchPolicyBackend is a documented seam"
            )


@dataclasses.dataclass
class AdvisorTrainerConfig:
    objective: str = "pacevolve++"
    n_samples: int = 8
    top_k: int = 4
    total_steps: int = 1000
    eps_num: float = 1e-8
    eps_skip: float = 1e-6
    entropic_gamma: float = 1.0
    clip: rl_loss.ClipConfig = dataclasses.field(
        default_factory=rl_loss.ClipConfig
    )

    @classmethod
    def from_config(cls, config: dict) -> "AdvisorTrainerConfig":
        rl = config.get("rl", {})
        return cls(
            objective=rl.get("objective", "pacevolve++"),
            n_samples=rl.get("n_samples", 8),
            top_k=rl.get("top_k", 4),
            total_steps=rl.get("total_steps", 1000),
            eps_num=rl.get("eps_num", 1e-8),
            eps_skip=rl.get("eps_skip", 1e-6),
            entropic_gamma=rl.get("entropic_gamma", 1.0),
            clip=rl_loss.ClipConfig(
                eps_lo=rl.get("clip_eps_lo", 0.2),
                eps_hi=rl.get("clip_eps_hi", 0.28),
            ),
        )


class AdvisorTrainer:
    def __init__(
        self, config: AdvisorTrainerConfig, backend: PolicyBackend
    ):
        self.config = config
        self.backend = backend

    def compute_advantages(self, rewards, step) -> advantages.MixResult:
        rewards = np.asarray(rewards, dtype=float)
        alpha = advantages.alpha_schedule(step, self.config.total_steps)

        if self.config.objective == "pacevolve++":
            return advantages.phase_adaptive_mix(
                rewards,
                self.config.top_k,
                alpha,
                self.config.eps_num,
                self.config.eps_skip,
            )
        if self.config.objective == "maxk":
            return advantages.phase_adaptive_mix(
                rewards,
                self.config.top_k,
                1.0,
                self.config.eps_num,
                self.config.eps_skip,
            )
        if self.config.objective in {"grpo", "entropic"}:
            if (
                not np.all(np.isfinite(rewards))
                or advantages.branch_std_collapsed(
                    rewards, self.config.eps_skip
                )
            ):
                return advantages.MixResult(
                    None, True, "collapsed", alpha, None, None
                )

            if self.config.objective == "grpo":
                advantage_values = advantages.grpo_advantage(
                    rewards, self.config.eps_num
                )
            else:
                advantage_values = advantages.entropic_advantage(
                    rewards,
                    self.config.entropic_gamma,
                    self.config.eps_num,
                )
            return advantages.MixResult(
                advantage_values, False, "", alpha, None, None
            )
        if self.config.objective == "none":
            return advantages.MixResult(
                None, True, "no-rl", alpha, None, None
            )
        raise ValueError(f"unknown objective: {self.config.objective}")

    def train_step(self, group: RolloutGroup) -> TrainStepResult:
        mix = self.compute_advantages(group.rewards(), group.step)
        if mix.skip_update:
            return TrainStepResult(
                group.step,
                True,
                mix.skip_reason,
                mix.alpha,
                None,
                {},
            )

        metrics = self.backend.update(
            group, mix.advantages, self.config.clip
        )
        self.backend.sync_weights()
        return TrainStepResult(
            group.step,
            False,
            "",
            mix.alpha,
            mix.advantages,
            metrics,
        )
