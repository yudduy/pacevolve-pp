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

"""Advantage estimators used by PACEvolve++."""

import dataclasses
import itertools
import logging
import math

import numpy as np


logger = logging.getLogger("controller")

DEFAULT_EPS_NUM = 1e-8
DEFAULT_EPS_SKIP = 1e-6


def group_relative_advantage(rewards) -> np.ndarray:
    """Return rewards centered by their population mean."""
    rewards = np.asarray(rewards, dtype=float)
    return rewards - rewards.mean()


def grpo_advantage(rewards, eps_num=DEFAULT_EPS_NUM) -> np.ndarray:
    """Return the population-standardized GRPO advantage."""
    rewards = np.asarray(rewards, dtype=float)
    sigma = rewards.std(ddof=0)
    if sigma == 0.0:
        return np.zeros_like(rewards)
    return (rewards - rewards.mean()) / (sigma + eps_num)


def standardize(values, eps_num=DEFAULT_EPS_NUM) -> np.ndarray:
    """Apply population standardization with an additive denominator epsilon."""
    values = np.asarray(values, dtype=float)
    sigma = values.std(ddof=0)
    if sigma == 0.0:
        return np.zeros_like(values)
    return (values - values.mean()) / (sigma + eps_num)


def branch_std_collapsed(values, eps_skip=DEFAULT_EPS_SKIP) -> bool:
    """Return whether a branch's population standard deviation is unusable."""
    sigma = np.asarray(values, dtype=float).std(ddof=0)
    return not np.isfinite(sigma) or sigma < eps_skip


def alpha_schedule(step, total_steps) -> float:
    """Linearly interpolate from zero to one across the scheduled steps."""
    if total_steps <= 1:
        return 0.0
    alpha = step / (total_steps - 1)
    return float(np.clip(alpha, 0.0, 1.0))


def _subset_count(n, k, minimum_k):
    if k < minimum_k or k > n:
        raise ValueError(f"k must satisfy {minimum_k} <= k <= {n}")

    count = math.comb(n, k)
    if count > 200_000:
        raise ValueError("direct subset enumeration exceeds 200,000 subsets")
    return count


def pkpo_weights(rewards, k) -> np.ndarray:
    """Compute PKPO weights by direct enumeration of size-k subsets."""
    rewards = np.asarray(rewards, dtype=float)
    n = len(rewards)
    count = _subset_count(n, k, minimum_k=1)
    weights = np.zeros(n, dtype=float)

    for subset in itertools.combinations(range(n), k):
        subset_max = max(rewards[j] for j in subset)
        for i in subset:
            weights[i] += subset_max

    return weights / count


def sloo_weights(rewards, k) -> np.ndarray:
    """Compute SLOO_(k-1) weights by direct size-k subset enumeration."""
    rewards = np.asarray(rewards, dtype=float)
    n = len(rewards)
    count = _subset_count(n, k, minimum_k=2)
    weights = np.zeros(n, dtype=float)

    for subset in itertools.combinations(range(n), k):
        subset_max = max(rewards[j] for j in subset)
        for i in subset:
            leave_one_out_max = max(rewards[j] for j in subset if j != i)
            weights[i] += subset_max - leave_one_out_max

    return weights / count


def _entropic_kl(rewards, beta):
    shifted = beta * (rewards - rewards.max())
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    positive = probabilities > 0.0
    return math.log(len(rewards)) + float(
        np.sum(probabilities[positive] * np.log(probabilities[positive]))
    )


def solve_entropic_beta(rewards, gamma, tol=1e-8, max_iter=200) -> float:
    """Solve for beta whose tilted distribution has the requested KL."""
    rewards = np.asarray(rewards, dtype=float)
    if rewards.max() == rewards.min():
        return 0.0

    max_gamma = (1.0 - 1e-9) * math.log(len(rewards))
    gamma = min(max(float(gamma), 0.0), max_gamma)
    if gamma <= 0.0:
        return 0.0

    lo = 0.0
    hi = 1.0
    while hi <= 1e6 and _entropic_kl(rewards, hi) < gamma:
        hi *= 2.0

    for _ in range(max_iter):
        beta = (lo + hi) / 2.0
        difference = _entropic_kl(rewards, beta) - gamma
        if abs(difference) <= tol:
            return beta
        if difference < 0.0:
            lo = beta
        else:
            hi = beta

    return (lo + hi) / 2.0


def entropic_advantage(
    rewards, gamma, eps_num=DEFAULT_EPS_NUM
) -> np.ndarray:
    """Return the TTT-Discover leave-one-out entropic advantage."""
    rewards = np.asarray(rewards, dtype=float)
    beta = solve_entropic_beta(rewards, gamma)
    if beta == 0.0:
        return np.zeros_like(rewards)

    shifted_weights = np.exp(beta * (rewards - rewards.max()))
    leave_one_out_normalizers = (
        shifted_weights.sum() - shifted_weights
    ) / (len(rewards) - 1)
    return shifted_weights / (leave_one_out_normalizers + eps_num) - 1.0


@dataclasses.dataclass
class MixResult:
    advantages: np.ndarray | None
    skip_update: bool
    skip_reason: str
    alpha: float
    a_group_std: np.ndarray | None
    a_topk_std: np.ndarray | None


def phase_adaptive_mix(
    rewards,
    k,
    alpha_t,
    eps_num=DEFAULT_EPS_NUM,
    eps_skip=DEFAULT_EPS_SKIP,
) -> MixResult:
    """Mix standardized group and SLOO branches with conditional skipping."""
    rewards = np.asarray(rewards, dtype=float)
    alpha = float(alpha_t)
    if not np.all(np.isfinite(rewards)):
        return MixResult(
            advantages=None,
            skip_update=True,
            skip_reason="rewards contain non-finite values",
            alpha=alpha,
            a_group_std=None,
            a_topk_std=None,
        )

    a_group = group_relative_advantage(rewards)
    # SLOO needs 2 <= k <= n; clamp an over-large k (best-of-k with k > n is
    # best-of-n) and treat k < 2 as a collapsed top-k branch, which only skips
    # the step when the top-k branch is actually active (alpha > 0).
    k_eff = min(k, len(rewards))
    if k_eff >= 2:
        a_topk = sloo_weights(rewards, k_eff)
    else:
        a_topk = np.zeros_like(rewards)
    collapsed_branches = []
    if 1.0 - alpha > 0.0 and branch_std_collapsed(a_group, eps_skip):
        collapsed_branches.append("group")
    if alpha > 0.0 and branch_std_collapsed(a_topk, eps_skip):
        collapsed_branches.append("top-k")

    if collapsed_branches:
        return MixResult(
            advantages=None,
            skip_update=True,
            skip_reason=(
                "active branch standard deviation collapsed: "
                + ", ".join(collapsed_branches)
            ),
            alpha=alpha,
            a_group_std=None,
            a_topk_std=None,
        )

    a_group_std = standardize(a_group, eps_num)
    a_topk_std = standardize(a_topk, eps_num)
    advantages = (1.0 - alpha) * a_group_std + alpha * a_topk_std
    return MixResult(
        advantages=advantages,
        skip_update=False,
        skip_reason="",
        alpha=alpha,
        a_group_std=a_group_std,
        a_topk_std=a_topk_std,
    )
