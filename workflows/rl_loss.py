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

"""Masked clipped surrogate policy loss for PACEvolve++."""

import dataclasses
import logging

import numpy as np

logger = logging.getLogger("controller")


@dataclasses.dataclass
class ClipConfig:
    """Asymmetric clipping bounds for policy ratios."""

    eps_lo: float = 0.2
    eps_hi: float = 0.28


@dataclasses.dataclass
class LossResult:
    """Aggregated loss and token-level diagnostics."""

    loss: float
    per_token: np.ndarray
    num_valid_tokens: int
    clip_fraction: float


def token_ratios(new_logprobs, old_logprobs) -> np.ndarray:
    """Return elementwise new-to-old policy probability ratios."""

    new_logprobs = np.asarray(new_logprobs, dtype=float)
    old_logprobs = np.asarray(old_logprobs, dtype=float)
    return np.exp(new_logprobs - old_logprobs)


def broadcast_advantages(advantages, mask) -> np.ndarray:
    """Broadcast response-level advantages over valid token positions."""

    advantages = np.asarray(advantages, dtype=float)
    mask = np.asarray(mask, dtype=float)
    return advantages[:, None] * mask


def clipped_surrogate_loss(
    ratios, advantages, mask, clip=ClipConfig()
) -> LossResult:
    """Compute the masked asymmetric clipped surrogate policy loss."""

    ratios = np.asarray(ratios, dtype=float)
    advantages = np.asarray(advantages, dtype=float)
    mask = np.asarray(mask, dtype=float)

    if advantages.ndim == 1:
        token_advantages = broadcast_advantages(advantages, mask)
    else:
        token_advantages = advantages * mask

    unclipped = ratios * token_advantages
    clipped = np.clip(
        ratios, 1.0 - clip.eps_lo, 1.0 + clip.eps_hi
    ) * token_advantages
    per_token = np.minimum(unclipped, clipped) * mask

    num_valid_tokens = int(mask.sum())
    if num_valid_tokens == 0:
        loss = 0.0
        clip_fraction = 0.0
    else:
        loss = -(per_token.sum() / num_valid_tokens)
        clip_active = (
            (ratios < 1.0 - clip.eps_lo)
            | (ratios > 1.0 + clip.eps_hi)
        ) * mask
        clip_fraction = float(clip_active.sum() / num_valid_tokens)

    return LossResult(
        loss=float(loss),
        per_token=per_token,
        num_valid_tokens=num_valid_tokens,
        clip_fraction=clip_fraction,
    )


def policy_loss_from_logprobs(
    new_logprobs,
    old_logprobs,
    advantages,
    mask,
    clip=ClipConfig(),
) -> LossResult:
    """Compute the clipped surrogate loss from policy log-probabilities."""

    ratios = token_ratios(new_logprobs, old_logprobs)
    return clipped_surrogate_loss(ratios, advantages, mask, clip)
