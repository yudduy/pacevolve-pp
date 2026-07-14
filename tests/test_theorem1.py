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

"""Theorem 1 / Appendix F: scale-conditioned credit assignment.

For g_i(delta) = c + delta*r_i (fixed ranking, compression scale delta>0):
  A_G_i(delta)      = delta*(r_i - r_bar)          (Eq. 7)
  ||A_G(delta)||^2  = N * delta^2 * sigma_r^2       (Eq. 8)
  w_SLOO_i(delta)   = delta * w_SLOO_i(1)           (Eq. 9)
  ||w_SLOO(delta)||^2 = delta^2 * ||w_SLOO(1)||^2   (Eq. 10)
  ||Phi_eps(B)||^2  = N*sigma^2/(sigma+eps)^2 <= N
"""

import numpy as np
import pytest

from advantages import group_relative_advantage, sloo_weights, standardize

DELTAS = [1.0, 1e-1, 1e-3, 1e-9]


def _random_profile(rng, n):
    # distinct rewards (strict ordering assumed by Theorem 1)
    r = rng.normal(size=n)
    while len(np.unique(r)) != n:
        r = rng.normal(size=n)
    return r


def test_group_relative_scale_eq7_eq8():
    rng = np.random.default_rng(10)
    for _ in range(50):
        n = int(rng.integers(3, 10))
        r = _random_profile(rng, n)
        c = rng.uniform(-5, 5)
        for delta in DELTAS:
            g = c + delta * r
            A = group_relative_advantage(g)
            np.testing.assert_allclose(A, delta * (r - r.mean()), rtol=1e-7, atol=1e-12)
            sq = float((A**2).sum())
            assert sq == pytest.approx(n * delta**2 * r.var(ddof=0), rel=1e-6, abs=1e-18)


def test_sloo_scale_eq9_eq10():
    rng = np.random.default_rng(11)
    for _ in range(50):
        n = int(rng.integers(3, 9))
        k = int(rng.integers(2, n + 1))
        r = _random_profile(rng, n)
        c = rng.uniform(-5, 5)
        base = sloo_weights(r, k)
        for delta in DELTAS:
            g = c + delta * r
            np.testing.assert_allclose(sloo_weights(g, k), delta * base, rtol=1e-6, atol=1e-14)
            sq = float((sloo_weights(g, k) ** 2).sum())
            assert sq == pytest.approx(delta**2 * float((base**2).sum()), rel=1e-6, abs=1e-18)


def test_standardized_group_branch_closed_form():
    rng = np.random.default_rng(12)
    eps = 1e-8
    # `expected` is computed from r directly (high precision), while Z is computed
    # from g = c + delta*r, which loses the delta*r signal to catastrophic
    # cancellation once delta << |c|. So this reconstruction identity is only
    # numerically checkable at well-conditioned scales; the delta=1e-9 compression
    # regime is covered by the scaling-law tests (Eq. 7-10) via absolute tolerances.
    for _ in range(50):
        n = int(rng.integers(3, 10))
        r = _random_profile(rng, n)
        c = rng.uniform(-5, 5)
        for delta in (1.0, 1e-1, 1e-3):
            g = c + delta * r
            Z = standardize(group_relative_advantage(g), eps_num=eps)
            sigma_r = r.std(ddof=0)
            expected = delta * (r - r.mean()) / (delta * sigma_r + eps)
            np.testing.assert_allclose(Z, expected, rtol=1e-6, atol=1e-9)


def test_standardized_norm_bounded_by_n():
    rng = np.random.default_rng(13)
    eps = 1e-8
    for _ in range(100):
        n = int(rng.integers(2, 12))
        r = _random_profile(rng, n)
        for delta in DELTAS:
            for branch in (group_relative_advantage(delta * r), sloo_weights(delta * r + 1.0, min(3, n))):
                assert float((standardize(branch, eps_num=eps) ** 2).sum()) <= n + 1e-9
