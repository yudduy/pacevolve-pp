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

"""Spec for workflows/rl_loss.py — masked clipped surrogate (Eq. 6).

L = -(1/|T_t|) * sum_{valid (i,tau)} min( r*A , clip(r, 1-eps_lo, 1+eps_hi)*A ).
Response-level advantage A is broadcast to every valid response token.
"""

import numpy as np
import pytest

from rl_loss import (
    ClipConfig,
    LossResult,
    broadcast_advantages,
    clipped_surrogate_loss,
    policy_loss_from_logprobs,
    token_ratios,
)


def test_token_ratios():
    new = np.log(np.array([[2.0, 4.0]]))
    old = np.log(np.array([[1.0, 2.0]]))
    np.testing.assert_allclose(token_ratios(new, old), [[2.0, 2.0]])


def test_broadcast_places_scalar_at_valid_positions_only():
    adv = np.array([2.0, -2.0])
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    np.testing.assert_allclose(broadcast_advantages(adv, mask),
                               [[2.0, 2.0, 0.0], [-2.0, 0.0, 0.0]])


def test_onpolicy_ratio_one_is_negative_mean_advantage():
    adv = np.array([1.0, 3.0, -2.0])
    mask = np.ones((3, 2))
    ratios = np.ones((3, 2))
    res = clipped_surrogate_loss(ratios, adv, mask)
    assert isinstance(res, LossResult)
    # per-token surrogate == broadcast advantage; loss == -mean over valid tokens
    assert res.loss == pytest.approx(-(1 + 1 + 3 + 3 - 2 - 2) / 6)
    assert res.num_valid_tokens == 6
    assert res.clip_fraction == pytest.approx(0.0)


def test_asymmetric_clip_hand_computed():
    # clip range = [1-0.2, 1+0.28] = [0.8, 1.28]
    adv = np.array([2.0, -2.0])
    mask = np.ones((2, 2))
    ratios = np.array([[1.25, 1.30], [0.75, 0.90]])
    res = clipped_surrogate_loss(ratios, adv, mask, ClipConfig(eps_lo=0.2, eps_hi=0.28))
    # row0 (A=+2): tau0 r=1.25 in-range -> 2.5 ; tau1 r=1.30 clipped 1.28 -> 2.56
    # row1 (A=-2): tau0 r=0.75 clipped 0.8 -> min(-1.5,-1.6)=-1.6 ; tau1 r=0.90 -> -1.8
    np.testing.assert_allclose(res.per_token, [[2.5, 2.56], [-1.6, -1.8]])
    assert res.loss == pytest.approx(-(2.5 + 2.56 - 1.6 - 1.8) / 4)
    assert res.clip_fraction == pytest.approx(0.5)  # 2 of 4 tokens clipped


def test_mask_excludes_tokens():
    adv = np.array([5.0, 5.0])
    mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    ratios = np.ones((2, 3))
    res = clipped_surrogate_loss(ratios, adv, mask)
    assert res.num_valid_tokens == 3
    assert res.loss == pytest.approx(-5.0)  # all valid tokens carry advantage 5
    # masked positions contribute exactly zero
    assert res.per_token[0, 2] == 0.0 and res.per_token[1, 1] == 0.0


def test_nonfinite_ratio_at_masked_position_cannot_poison_loss():
    new = np.array([[0.0, 1000.0]])
    old = np.zeros((1, 2))
    mask = np.array([[1.0, 0.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        poisoned = policy_loss_from_logprobs(new, old, [2.0], mask)
        baseline = policy_loss_from_logprobs(
            new[:, :1], old[:, :1], [2.0], mask[:, :1]
        )
    assert np.isfinite(poisoned.loss)
    assert poisoned.loss == pytest.approx(baseline.loss)
    assert poisoned.per_token[0, 1] == 0.0


def test_empty_mask_is_safe():
    res = clipped_surrogate_loss(np.ones((2, 2)), np.array([1.0, 2.0]), np.zeros((2, 2)))
    assert res.num_valid_tokens == 0
    assert res.loss == pytest.approx(0.0)
    assert res.clip_fraction == pytest.approx(0.0)


def test_policy_loss_from_logprobs_matches():
    rng = np.random.default_rng(30)
    new = rng.normal(size=(4, 5))
    old = rng.normal(size=(4, 5))
    adv = rng.normal(size=4)
    mask = (rng.uniform(size=(4, 5)) > 0.3).astype(float)
    a = policy_loss_from_logprobs(new, old, adv, mask)
    b = clipped_surrogate_loss(token_ratios(new, old), adv, mask)
    assert a.loss == pytest.approx(b.loss)
    np.testing.assert_allclose(a.per_token, b.per_token)


def test_accepts_token_level_advantages():
    # advantages already (n, T): still respect the mask
    adv_tok = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[1, 0], [1, 1]], dtype=float)
    res = clipped_surrogate_loss(np.ones((2, 2)), adv_tok, mask)
    assert res.num_valid_tokens == 3
    assert res.loss == pytest.approx(-(1.0 + 3.0 + 4.0) / 3)
