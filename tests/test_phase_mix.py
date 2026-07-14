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

"""Spec for phase_adaptive_mix (Eq. 4): standardize each branch within the
rollout group, then A_mix = (1-alpha)*A_group_std + alpha*A_topk_std; skip the
update when an active branch's std collapses."""

import numpy as np
import pytest

from advantages import (
    group_relative_advantage,
    phase_adaptive_mix,
    sloo_weights,
    standardize,
)


def test_alpha_zero_is_standardized_group_branch():
    R = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    res = phase_adaptive_mix(R, k=2, alpha_t=0.0)
    assert not res.skip_update
    assert res.alpha == pytest.approx(0.0)
    np.testing.assert_allclose(res.advantages, standardize(group_relative_advantage(R)))


def test_alpha_one_is_standardized_topk_branch():
    R = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    res = phase_adaptive_mix(R, k=2, alpha_t=1.0)
    assert not res.skip_update
    np.testing.assert_allclose(res.advantages, standardize(sloo_weights(R, 2)))


def test_alpha_half_is_linear_combination():
    R = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    k = 3
    g = standardize(group_relative_advantage(R))
    t = standardize(sloo_weights(R, k))
    res = phase_adaptive_mix(R, k=k, alpha_t=0.5)
    np.testing.assert_allclose(res.advantages, 0.5 * g + 0.5 * t)


def test_constant_rewards_skip_at_all_alpha():
    R = np.array([4.0, 4.0, 4.0, 4.0])
    for alpha in (0.0, 0.5, 1.0):
        res = phase_adaptive_mix(R, k=2, alpha_t=alpha)
        assert res.skip_update
        assert res.advantages is None
        assert res.skip_reason


def test_nan_reward_skips():
    R = np.array([1.0, 2.0, np.nan, 4.0])
    res = phase_adaptive_mix(R, k=2, alpha_t=0.5)
    assert res.skip_update
    assert res.advantages is None


def test_mixresult_reports_alpha_and_branches():
    R = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    res = phase_adaptive_mix(R, k=2, alpha_t=0.3)
    assert res.alpha == pytest.approx(0.3)
    assert res.a_group_std is not None
    assert res.a_topk_std is not None
    assert res.advantages.shape == R.shape
