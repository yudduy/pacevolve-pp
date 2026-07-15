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

"""Spec for workflows/rl_rewards.py — progress-normalized reward shaping (Eq. 5).

R_RL(y) = c * u(y)^alpha_r, with
  u(y) = clamp((y - ymin)/(ymax - ymin), 0, 1)        [maximize]
  u(y) = clamp((ymax - y)/(ymax - ymin), 0, 1)        [minimize]
Failure / unparseable / non-finite score -> failure_reward (-1.0).
Default (c=5, alpha_r=1) is the paper's "linearly scaled progress reward on [0,5]".
Bounds default from evaluation.{init_score,target_score,metric_direction}.
"""

import math

import numpy as np
import pytest

from rl_rewards import (
    RewardShapingConfig,
    normalized_progress,
    shape_reward,
    shape_rewards,
)


# --- normalized_progress: maximize ---------------------------------------

def test_progress_max_endpoints_and_midpoint():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5)
    assert cfg.y_min == 1 and cfg.y_max == 5
    assert normalized_progress(1, cfg) == pytest.approx(0.0)
    assert normalized_progress(5, cfg) == pytest.approx(1.0)
    assert normalized_progress(3, cfg) == pytest.approx(0.5)


def test_progress_max_clamps_outside_bounds():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5)
    assert normalized_progress(10, cfg) == pytest.approx(1.0)
    assert normalized_progress(-4, cfg) == pytest.approx(0.0)


# --- normalized_progress: minimize (lower score is better) ---------------

def test_progress_min_endpoints():
    # llmsr-like: init -2 (worst), target -10 (best); ymin=-10, ymax=-2.
    cfg = RewardShapingConfig(metric_direction="min", y_init=-2, y_target=-10)
    assert cfg.y_min == -10 and cfg.y_max == -2
    assert normalized_progress(-2, cfg) == pytest.approx(0.0)   # worst
    assert normalized_progress(-10, cfg) == pytest.approx(1.0)  # best
    assert normalized_progress(-6, cfg) == pytest.approx(0.5)


def test_progress_min_clamps_outside_bounds():
    cfg = RewardShapingConfig(metric_direction="min", y_init=-2, y_target=-10)
    assert normalized_progress(-20, cfg) == pytest.approx(1.0)
    assert normalized_progress(5, cfg) == pytest.approx(0.0)


# --- shape_reward: [0,5] scaling and shaping -----------------------------

def test_shape_reward_default_linear_0_to_5():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5)
    assert shape_reward(1, cfg) == pytest.approx(0.0)
    assert shape_reward(3, cfg) == pytest.approx(2.5)
    assert shape_reward(5, cfg) == pytest.approx(5.0)


def test_shape_reward_min_direction():
    cfg = RewardShapingConfig(metric_direction="min", y_init=-2, y_target=-10)
    assert shape_reward(-10, cfg) == pytest.approx(5.0)
    assert shape_reward(-2, cfg) == pytest.approx(0.0)


def test_shape_reward_custom_scale_and_exponent():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5,
                              scale_c=1.0, alpha_r=1.0)
    assert shape_reward(3, cfg) == pytest.approx(0.5)
    cfg2 = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5,
                               scale_c=1.0, alpha_r=2.0)
    assert shape_reward(3, cfg2) == pytest.approx(0.25)


# --- shape_reward: failure branch ----------------------------------------

@pytest.mark.parametrize(
    "bad",
    [None, float("nan"), float("inf"), float("-inf"), "abc", [1, 2]],
)
def test_shape_reward_failure_maps_to_minus_one(bad):
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5)
    assert shape_reward(bad, cfg) == -1.0


def test_custom_failure_reward():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5,
                              failure_reward=-2.5)
    assert shape_reward(None, cfg) == -2.5


# --- bounds resolution ----------------------------------------------------

def test_bounds_default_independent_of_init_target_order():
    a = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5)
    b = RewardShapingConfig(metric_direction="max", y_init=5, y_target=1)
    assert (a.y_min, a.y_max) == (b.y_min, b.y_max) == (1, 5)


def test_explicit_bounds_override_defaults():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5,
                              y_min=0, y_max=10)
    assert (cfg.y_min, cfg.y_max) == (0, 10)
    assert shape_reward(5, cfg) == pytest.approx(2.5)  # 5*(5-0)/(10-0)


def test_degenerate_bounds_raise():
    with pytest.raises(ValueError):
        RewardShapingConfig(metric_direction="max", y_init=3, y_target=3)


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        RewardShapingConfig(metric_direction="sideways", y_init=1, y_target=5)


# --- from_config: read existing task YAML keys ---------------------------

def test_from_config_maximize():
    config = {
        "evaluation": {"init_score": 1, "target_score": 3, "metric_direction": "max"},
        "rl": {"reward": {"scale_c": 5.0, "alpha_r": 1.0}},
    }
    cfg = RewardShapingConfig.from_config(config)
    assert cfg.metric_direction == "max"
    assert (cfg.y_min, cfg.y_max) == (1, 3)
    assert cfg.scale_c == 5.0 and cfg.alpha_r == 1.0


def test_from_config_minimize_and_reward_defaults():
    config = {
        "evaluation": {"init_score": -2, "target_score": -10, "metric_direction": "min"},
    }
    cfg = RewardShapingConfig.from_config(config)
    assert cfg.metric_direction == "min"
    assert (cfg.y_min, cfg.y_max) == (-10, -2)
    assert cfg.scale_c == 5.0 and cfg.alpha_r == 1.0  # documented defaults


def test_from_config_accepts_none_rl_section():
    config = {
        "evaluation": {
            "init_score": "1.0",
            "target_score": "3.0",
            "metric_direction": "max",
        },
        "rl": None,
    }
    cfg = RewardShapingConfig.from_config(config)
    assert (cfg.y_init, cfg.y_target) == (1.0, 3.0)
    assert cfg.scale_c == 5.0 and cfg.alpha_r == 1.0


def test_from_config_coerces_reward_numeric_strings():
    config = {
        "evaluation": {
            "init_score": "1.0",
            "target_score": "3.0",
            "metric_direction": "max",
        },
        "rl": {"reward": {"scale_c": "5.0", "alpha_r": "2.0"}},
    }
    cfg = RewardShapingConfig.from_config(config)
    assert cfg.scale_c == 5.0 and isinstance(cfg.scale_c, float)
    assert cfg.alpha_r == 2.0 and isinstance(cfg.alpha_r, float)


# --- shape_rewards: vectorized -------------------------------------------

def test_shape_rewards_vectorized_with_failures():
    cfg = RewardShapingConfig(metric_direction="max", y_init=1, y_target=5)
    out = shape_rewards([1, 3, 5, None, float("nan")], cfg)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, [0.0, 2.5, 5.0, -1.0, -1.0])
