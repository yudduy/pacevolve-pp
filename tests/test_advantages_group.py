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

"""Spec for the group-relative / GRPO / standardize / schedule pieces of
workflows/advantages.py."""

import numpy as np
import pytest

from advantages import (
    alpha_schedule,
    branch_std_collapsed,
    group_relative_advantage,
    grpo_advantage,
    standardize,
)


def test_group_relative_is_centered_raw():
    R = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    A = group_relative_advantage(R)
    np.testing.assert_allclose(A, R - R.mean())
    assert A.sum() == pytest.approx(0.0)


def test_group_relative_has_no_std_division():
    # A fixture with std != 1 so raw group-relative != GRPO.
    R = np.array([0.0, 10.0, 20.0, 30.0])
    assert not np.allclose(group_relative_advantage(R), grpo_advantage(R))


def test_grpo_matches_population_zscore():
    R = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    eps = 1e-8
    expected = (R - R.mean()) / (R.std(ddof=0) + eps)
    np.testing.assert_allclose(grpo_advantage(R, eps_num=eps), expected)


def test_grpo_constant_rewards_are_finite_zeros():
    R = np.array([2.0, 2.0, 2.0])
    A = grpo_advantage(R)
    assert np.all(np.isfinite(A))
    np.testing.assert_allclose(A, 0.0)


def test_standardize_zero_mean_and_norm_bound():
    rng = np.random.default_rng(0)
    eps = 1e-8
    for _ in range(200):
        n = rng.integers(2, 12)
        B = rng.normal(size=n) * rng.uniform(0.1, 5.0)
        Z = standardize(B, eps_num=eps)
        assert Z.mean() == pytest.approx(0.0, abs=1e-9)
        sigma = B.std(ddof=0)
        expected_sq = n * sigma**2 / (sigma + eps) ** 2
        assert float((Z**2).sum()) == pytest.approx(expected_sq, rel=1e-6, abs=1e-9)
        assert float((Z**2).sum()) <= n + 1e-9


def test_standardize_constant_is_zeros():
    Z = standardize(np.array([5.0, 5.0, 5.0]))
    assert np.all(np.isfinite(Z))
    np.testing.assert_allclose(Z, 0.0)


def test_branch_std_collapsed():
    assert branch_std_collapsed(np.array([1.0, 1.0, 1.0]), eps_skip=1e-6)
    assert branch_std_collapsed(np.array([1.0, np.nan, 2.0]), eps_skip=1e-6)
    assert not branch_std_collapsed(np.array([0.0, 1.0, 2.0]), eps_skip=1e-6)
    # eps_skip controls the threshold
    assert branch_std_collapsed(np.array([0.0, 1.0, 2.0]), eps_skip=1e9)


def test_alpha_schedule_linear_0_to_1():
    assert alpha_schedule(0, 1000) == pytest.approx(0.0)
    assert alpha_schedule(999, 1000) == pytest.approx(1.0)
    assert alpha_schedule(500, 1000) == pytest.approx(500 / 999)


def test_alpha_schedule_clamps_and_degenerate_total():
    assert alpha_schedule(-5, 1000) == pytest.approx(0.0)
    assert alpha_schedule(5000, 1000) == pytest.approx(1.0)
    assert alpha_schedule(0, 1) == pytest.approx(0.0)
    assert alpha_schedule(0, 0) == pytest.approx(0.0)
