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

"""Spec for the TTT-Discover entropic baseline in advantages.py.

beta is chosen so KL(q_beta || uniform) = gamma, q_beta(i) ∝ exp(beta*R_i);
leave-one-out advantage A_i = exp(beta*(R_i - Rmax))/(Z_-i + eps) - 1, with
Z_-i = mean_{j!=i} exp(beta*(R_j - Rmax)).
"""

import numpy as np
import pytest

from advantages import entropic_advantage, solve_entropic_beta


def _kl_uniform(R, beta):
    R = np.asarray(R, float)
    n = len(R)
    z = beta * (R - R.max())
    q = np.exp(z)
    q = q / q.sum()
    # KL(q || uniform) = log n + sum q log q
    nz = q > 0
    return np.log(n) + float(np.sum(q[nz] * np.log(q[nz])))


def test_solve_beta_hits_target_kl():
    rng = np.random.default_rng(20)
    for _ in range(30):
        n = int(rng.integers(3, 10))
        R = rng.normal(size=n)
        if len(np.unique(R)) != n:
            continue
        gamma = float(rng.uniform(0.05, 0.9 * np.log(n)))
        beta = solve_entropic_beta(R, gamma, tol=1e-10, max_iter=300)
        assert beta >= 0.0
        assert _kl_uniform(R, beta) == pytest.approx(gamma, abs=1e-6)


def test_kl_monotonic_in_beta():
    rng = np.random.default_rng(21)
    R = rng.normal(size=6)
    betas = np.linspace(0, 20, 50)
    kls = [_kl_uniform(R, b) for b in betas]
    assert all(b <= a + 1e-9 for a, b in zip(kls[1:], kls[:-1]))  # non-decreasing


def test_beta_zero_gives_uniform_and_zero_advantage():
    R = np.array([1.0, 2.0, 3.0, 4.0])
    # gamma = 0 -> beta = 0 -> uniform q -> advantages all ~0
    beta = solve_entropic_beta(R, 0.0)
    assert beta == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(entropic_advantage(R, 0.0), 0.0, atol=1e-9)


def test_constant_rewards_beta_zero():
    R = np.array([2.0, 2.0, 2.0, 2.0])
    assert solve_entropic_beta(R, 0.5) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(entropic_advantage(R, 0.5), 0.0, atol=1e-9)


def test_entropic_advantage_matches_definition():
    R = np.array([0.0, 1.0, 2.0])
    gamma = 0.3
    beta = solve_entropic_beta(R, gamma)
    eps = 1e-8
    z = np.exp(beta * (R - R.max()))
    n = len(R)
    expected = np.empty(n)
    for i in range(n):
        z_minus_i = (z.sum() - z[i]) / (n - 1)
        expected[i] = z[i] / (z_minus_i + eps) - 1.0
    np.testing.assert_allclose(entropic_advantage(R, gamma, eps_num=eps), expected, rtol=1e-7)


def test_large_reward_spread_no_overflow():
    R = np.array([0.0, 500.0, 1000.0])  # shifted exponents must avoid overflow
    A = entropic_advantage(R, 0.5)
    assert np.all(np.isfinite(A))
