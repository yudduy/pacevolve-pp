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

"""Spec for PKPO (Eq. 2) and SLOO_{k-1} (Eq. 3) weights in advantages.py.

The in-test brute-force oracles enumerate all size-k subsets directly from the
equations and are the ground truth; the module's implementation must match them.
"""

import itertools

import numpy as np
import pytest

from advantages import pkpo_weights, sloo_weights


# --- brute-force oracles straight from Eq. 2 / Eq. 3 ---------------------

def _pkpo_oracle(g, k):
    g = np.asarray(g, dtype=float)
    n = len(g)
    w = np.zeros(n)
    subsets = list(itertools.combinations(range(n), k))
    for subset in subsets:
        mx = max(g[j] for j in subset)
        for i in subset:
            w[i] += mx
    return w / len(subsets)


def _sloo_oracle(g, k):
    g = np.asarray(g, dtype=float)
    n = len(g)
    w = np.zeros(n)
    subsets = list(itertools.combinations(range(n), k))
    for subset in subsets:
        mx = max(g[j] for j in subset)
        for i in subset:
            second = max(g[j] for j in subset if j != i)  # k>=2 => non-empty
            w[i] += mx - second
    return w / len(subsets)


# --- closed form == oracle over a grid of N, k --------------------------

def test_pkpo_matches_oracle_grid():
    rng = np.random.default_rng(1)
    for n in range(2, 9):
        for k in range(1, n + 1):
            for _ in range(5):
                g = rng.normal(size=n) * rng.uniform(0.5, 4.0) + rng.uniform(-3, 3)
                np.testing.assert_allclose(pkpo_weights(g, k), _pkpo_oracle(g, k),
                                           rtol=1e-9, atol=1e-9)


def test_sloo_matches_oracle_grid():
    rng = np.random.default_rng(2)
    for n in range(2, 10):
        for k in range(2, n + 1):
            for _ in range(5):
                g = rng.normal(size=n) * rng.uniform(0.5, 4.0) + rng.uniform(-3, 3)
                np.testing.assert_allclose(sloo_weights(g, k), _sloo_oracle(g, k),
                                           rtol=1e-9, atol=1e-9)


# --- unbiasedness / summation identities --------------------------------

def test_pkpo_sum_equals_k_times_mean_subset_max():
    rng = np.random.default_rng(3)
    g = rng.normal(size=7)
    k = 4
    subset_max = [max(g[j] for j in S) for S in itertools.combinations(range(7), k)]
    assert pkpo_weights(g, k).sum() == pytest.approx(k * np.mean(subset_max))


def test_sloo_sum_equals_mean_margin():
    rng = np.random.default_rng(4)
    g = rng.normal(size=7)
    k = 4
    margins = []
    for S in itertools.combinations(range(7), k):
        vals = sorted((g[j] for j in S), reverse=True)
        margins.append(vals[0] - vals[1])
    assert sloo_weights(g, k).sum() == pytest.approx(np.mean(margins))


# --- structural properties of SLOO --------------------------------------

def test_sloo_bottom_k_minus_1_are_zero():
    rng = np.random.default_rng(5)
    n, k = 8, 4
    g = rng.permutation(np.arange(n).astype(float))  # distinct
    w = sloo_weights(g, k)
    order = np.argsort(g)  # ascending
    bottom = order[: k - 1]
    np.testing.assert_allclose(w[bottom], 0.0, atol=1e-12)
    # and the top element carries positive weight
    assert w[order[-1]] > 0.0


def test_sloo_k_equals_n_only_max_gets_margin():
    g = np.array([0.0, 1.0, 2.0, 3.0])
    w = sloo_weights(g, 4)  # single subset = whole set
    np.testing.assert_allclose(w, [0.0, 0.0, 0.0, 1.0])  # 3 - 2 for the max


def test_sloo_hand_example_k2():
    g = np.array([0.0, 1.0, 2.0])  # subsets {01}{02}{12}, C=3
    # i=2 (val 2) is max in {02}->2, {12}->1  => (2-0)+(2-1)=3 ; /3 = 1.0
    # i=1 (val 1) is max only in {01}->1-0=1 ; /3
    # i=0 never max -> 0
    np.testing.assert_allclose(sloo_weights(g, 2), [0.0, 1 / 3, 3 / 3])


# --- ties -----------------------------------------------------------------

def test_ties_match_oracle():
    for g in ([1, 1, 1, 1, 1, 1, 1, 0], [2, 2, 1, 1], [5, 5, 5, 5]):
        g = np.asarray(g, float)
        for k in range(2, len(g) + 1):
            np.testing.assert_allclose(sloo_weights(g, k), _sloo_oracle(g, k), atol=1e-12)


def test_all_equal_sloo_is_zero():
    g = np.array([3.0, 3.0, 3.0, 3.0])
    np.testing.assert_allclose(sloo_weights(g, 2), 0.0)


# --- translation equivariance / positive homogeneity (Appendix F) --------

def test_sloo_translation_and_scale():
    rng = np.random.default_rng(6)
    g = rng.normal(size=6)
    k = 3
    np.testing.assert_allclose(sloo_weights(g + 7.5, k), sloo_weights(g, k), atol=1e-12)
    np.testing.assert_allclose(sloo_weights(3.0 * g, k), 3.0 * sloo_weights(g, k), atol=1e-12)
