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

"""Progress-normalized reward shaping for reinforcement learning."""

import dataclasses
import logging
import math

import numpy as np

logger = logging.getLogger("controller")


@dataclasses.dataclass
class RewardShapingConfig:
    """Configuration for progress-normalized rewards."""

    metric_direction: str
    y_init: float
    y_target: float
    y_min: float | None = None
    y_max: float | None = None
    scale_c: float = 5.0
    alpha_r: float = 1.0
    failure_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.y_min is None:
            self.y_min = min(self.y_init, self.y_target)
        if self.y_max is None:
            self.y_max = max(self.y_init, self.y_target)
        if self.metric_direction not in {"max", "min"}:
            raise ValueError("metric_direction must be 'max' or 'min'")
        if not self.y_min < self.y_max:
            raise ValueError("y_min must be less than y_max")

    @classmethod
    def from_config(cls, config: dict) -> "RewardShapingConfig":
        """Build reward settings from an experiment configuration."""
        evaluation = config["evaluation"]
        reward = config.get("rl", {}).get("reward", {})
        return cls(
            metric_direction=evaluation["metric_direction"],
            y_init=evaluation["init_score"],
            y_target=evaluation["target_score"],
            scale_c=reward.get("scale_c", 5.0),
            alpha_r=reward.get("alpha_r", 1.0),
        )


def normalized_progress(y: float, cfg: RewardShapingConfig) -> float:
    """Return score progress clamped to the unit interval."""
    denominator = cfg.y_max - cfg.y_min
    if cfg.metric_direction == "max":
        numerator = y - cfg.y_min
    else:
        numerator = cfg.y_max - y
    return max(0.0, min(numerator / denominator, 1.0))


def shape_reward(y, cfg: RewardShapingConfig) -> float:
    """Return the shaped reward for one score."""
    if y is None:
        return cfg.failure_reward
    try:
        score = float(y)
    except (TypeError, ValueError, OverflowError):
        return cfg.failure_reward
    if not math.isfinite(score):
        return cfg.failure_reward
    return cfg.scale_c * (normalized_progress(score, cfg) ** cfg.alpha_r)


def shape_rewards(ys, cfg: RewardShapingConfig) -> np.ndarray:
    """Return shaped rewards for an iterable of scores."""
    return np.array([shape_reward(y, cfg) for y in ys], dtype=float)
