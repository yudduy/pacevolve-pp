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

"""Tinker-backed policy-gradient backend for the PACEvolve++ advisor.

Fills the documented ``TorchPolicyBackend`` seam in ``rl_trainer.py`` with a
real LoRA reinforcement-learning backend that trains the advisor through
Thinking Machines' **Tinker** managed API. Only the advisor is trained; the
implementer stays a frozen (black-box) client.

Sampling knobs come from the ``rl.advisor_*`` config surface (advisor_model,
advisor_temperature, advisor_max_tokens, advisor_enable_thinking) — NOT the
``advisor_llm`` block, which names the OpenRouter advisor for the mock/baseline
backends. Keep ``advisor_temperature`` at 1.0 unless you have confirmed Tinker
returns temperature-1 logprobs, otherwise the on-policy importance ratio is
biased.

Only importable where ``tinker``/``ttt_discover``/``transformers`` are installed
(FarmShare); the local suite never instantiates it. Correctness-critical
token/Datum assembly is REUSED from ``ttt_discover`` (copy-before-rewrite).

Threading: ``generate`` runs in ``generate_completion``'s ThreadPoolExecutor
worker; ``update``/``sync_weights`` run on the main thread. A single dedicated
thread owns the asyncio loop, the Tinker clients are BUILT on that loop thread
(loop affinity), and every Tinker call is dispatched onto it with
``run_coroutine_threadsafe`` under a wall-clock timeout that cancels a hung call.
"""

import asyncio
import logging
import os
import re
import threading
from concurrent.futures import TimeoutError as FuturesTimeoutError

import numpy as np

import rl_trainer


logger = logging.getLogger("controller")

_THINK_CLOSE = "</think>"


def _resolve_base_url(rl_config: dict) -> str | None:
    """Tinker server URL: TINKER_BASE_URL env > rl.tinker_base_url config > None (managed cloud)."""
    env = os.environ.get("TINKER_BASE_URL", "").strip()
    if env:
        return env
    value = rl_config.get("tinker_base_url")
    if value is None:
        return None
    return str(value).strip() or None


def _resolve_generation_timeout(config: dict, call_timeout: float) -> float:
    """Match generate_completion's role-specific wall-clock deadline."""
    llm = {**(config.get("llm") or {}), **(config.get("advisor_llm") or {})}
    hard_timeout = float(llm.get("request_timeout", 240)) + 30.0
    return min(call_timeout, hard_timeout)


def _resolve_fb_loss(fb_result) -> float:
    """Scalar training loss from a forward_backward result.

    Managed Tinker reports metrics["loss"]; skyrl-tx returns metrics={} and
    per-token losses in loss_fn_outputs[i]["elementwise_loss"] instead — take
    the token-mean so both servers produce a comparable curve.
    """
    metrics = getattr(fb_result, "metrics", None)
    if isinstance(metrics, dict) and "loss" in metrics:
        return float(metrics["loss"])
    total = 0.0
    count = 0
    for out in getattr(fb_result, "loss_fn_outputs", None) or []:
        if isinstance(out, dict):
            elem = out.get("elementwise_loss")
        else:
            elem = getattr(out, "elementwise_loss", None)
        if elem is None:
            continue
        if hasattr(elem, "tolist"):
            data = elem.tolist()
        elif isinstance(elem, dict):
            data = elem.get("data") or []
        else:
            data = getattr(elem, "data", None) or []
        for v in data:
            total += float(v)
            count += 1
    return total / count if count else float("nan")


class TinkerPolicyBackend(rl_trainer.PolicyBackend):
    """Real advisor policy backend: sample from Tinker, apply a LoRA update."""

    def __init__(self, config: dict):
        super().__init__(config)
        import tinker
        from transformers import AutoTokenizer

        rl = config.get("rl") or {}
        self.base_url = _resolve_base_url(rl)
        self.model_name = str(rl.get("advisor_model", "Qwen/Qwen3-8B"))
        self.lora_rank = int(rl.get("lora_rank", 32))
        self.learning_rate = float(rl.get("learning_rate", 1e-5))
        self.beta1 = float(rl.get("adam_beta1", 0.9))
        self.beta2 = float(rl.get("adam_beta2", 0.95))
        self.adam_eps = float(rl.get("adam_eps", 1e-8))
        # "importance_sampling" is what the ttt-discover run used (proven, safe).
        # NOTE: it is UNCLIPPED — the config clip_eps_* only affect the mock
        # backend's numpy loss. "ppo" enables Tinker's internal clipped surrogate
        # (bounds are Tinker's, not the exact DAPO 0.2/0.28). See warning below.
        self.loss_fn = str(rl.get("loss_fn", "importance_sampling"))
        self.temperature = float(rl.get("advisor_temperature", 1.0))
        self.max_tokens = int(rl.get("advisor_max_tokens", 2048))
        self.enable_thinking = bool(rl.get("advisor_enable_thinking", True))
        # Wall-clock backstop for any single Tinker call (bounds a hung worker).
        self._call_timeout = float(rl.get("tinker_call_timeout", 600.0))
        self._generation_timeout = _resolve_generation_timeout(
            config, self._call_timeout
        )

        self._tinker = tinker
        self._hf_tok = AutoTokenizer.from_pretrained(self.model_name)
        self._stop = self._compute_stop_ids()

        # One dedicated thread owns the loop; build the Tinker clients ON that
        # thread so any loop-bound primitives bind to self._loop (affinity).
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="tinker-loop", daemon=True
        )
        self._loop_thread.start()
        (
            self._service,
            self._training_client,
            self._sampling_client,
        ) = self._run(self._async_setup())

        # Populated by generate(); read by run_advisor_rl in the same rollout
        # worker thread to fill RolloutSample without cross-sample races.
        self._generation_state = threading.local()
        self.last_generation: rl_trainer.GenerationResult | None = None

        if self.loss_fn == "importance_sampling":
            logger.warning(
                "TinkerPolicyBackend loss_fn=importance_sampling: training is "
                "UNCLIPPED; DAPO clip (clip_eps_lo/hi) is NOT applied by Tinker. "
                "Set rl.loss_fn=ppo to enable Tinker's clipped surrogate."
            )
        logger.info(
            "TinkerPolicyBackend ready: model=%s lora_rank=%d lr=%g loss_fn=%s "
            "temperature=%.3f max_tokens=%d stop=%s base_url=%s",
            self.model_name, self.lora_rank, self.learning_rate, self.loss_fn,
            self.temperature, self.max_tokens, self._stop,
            self.base_url or "managed",
        )

    async def _async_setup(self):
        tinker = self._tinker
        service = tinker.ServiceClient(base_url=self.base_url)
        training_client = await service.create_lora_training_client_async(
            self.model_name, rank=self.lora_rank
        )
        sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async()
        )
        return service, training_client, sampling_client

    @property
    def last_generation(self) -> rl_trainer.GenerationResult | None:
        return getattr(self._generation_state, "last_generation", None)

    @last_generation.setter
    def last_generation(
        self, generation: rl_trainer.GenerationResult | None
    ) -> None:
        self._generation_state.last_generation = generation

    # -- async plumbing: dispatch onto the loop thread, bounded + cancellable --
    def _run(self, coro, timeout=None):
        call_timeout = self._call_timeout if timeout is None else timeout
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=call_timeout)
        except FuturesTimeoutError:
            future.cancel()
            logger.error(
                "Tinker call exceeded %.0fs wall-clock; cancelled.",
                call_timeout,
            )
            raise

    # -- tokenization helpers -----------------------------------------------
    def _compute_stop_ids(self) -> list[int]:
        """Stop on the real chat-turn terminator(s), not just eos.

        Qwen3 turns end with <|im_end|>; tokenizer.eos_token_id may be
        <|endoftext|>. Collect both valid ids so generation halts at end-of-turn
        instead of running to max_tokens.
        """
        ids: list[int] = []
        eos = self._hf_tok.eos_token_id
        if eos is not None:
            ids.append(int(eos))
        try:
            im_end = self._hf_tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            im_end = None
        unk = getattr(self._hf_tok, "unk_token_id", None)
        if im_end is not None and im_end >= 0 and im_end != unk and im_end not in ids:
            ids.append(int(im_end))
        return ids

    def _render_ids(self, prompt_text: str) -> list[int]:
        ids = self._hf_tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.enable_thinking,
        )
        # transformers version drift: apply_chat_template may return a list[int],
        # a BatchEncoding/dict ({"input_ids": [...]}), or a batched [[...]].
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return [int(t) for t in ids]

    def _model_input(self, ids):
        tinker = self._tinker
        return tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=[int(t) for t in ids])]
        )

    @staticmethod
    def _answer_text(full_text: str) -> str:
        """Strip Qwen3 <think>...</think> reasoning so the idea parsers (which the
        OpenRouter advisor only ever fed reasoning-stripped content) see the
        answer only. The FULL token sequence is still what gets trained."""
        if _THINK_CLOSE in full_text:
            return full_text.rsplit(_THINK_CLOSE, 1)[-1].strip()
        return full_text.strip()

    # -- PolicyBackend interface --------------------------------------------
    def generate(self, prompt: str, generation_config: dict) -> rl_trainer.GenerationResult:
        del generation_config  # sampling comes from rl.advisor_* (see docstring)
        tinker = self._tinker
        prompt_ids = self._render_ids(prompt)
        params = tinker.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self._stop,
        )
        result = self._run(
            self._sampling_client.sample_async(
                prompt=self._model_input(prompt_ids),
                num_samples=1,
                sampling_params=params,
            ),
            timeout=self._generation_timeout,
        )
        seq = result.sequences[0]
        tokens = list(seq.tokens)
        if seq.logprobs is None:
            raise RuntimeError(
                "Tinker sampling returned logprobs=None; cannot form the "
                "importance ratio. Enable logprobs on the sampling client."
            )
        logprobs = list(seq.logprobs)
        full_text = self._hf_tok.decode(tokens, skip_special_tokens=True)
        generation = rl_trainer.GenerationResult(
            # text = answer-only for the idea parsers; token_ids = FULL sequence
            # (reasoning + answer) which is the RL training target.
            text=self._answer_text(full_text),
            token_ids=np.asarray(tokens, dtype=np.int64),
            logprobs=np.asarray(logprobs, dtype=float),
            prompt_token_ids=np.asarray(prompt_ids, dtype=np.int64),
        )
        self.last_generation = generation
        return generation

    def update(self, group: "rl_trainer.RolloutGroup", advantages, clip) -> dict:
        del clip  # Tinker's loss_fn (a string) cannot receive clip bounds; the
        # config clip_eps_* apply only to the mock backend. See __init__ warning.
        from ttt_discover.rl import data_processing
        from ttt_discover.rl.types import Trajectory, Transition
        from ttt_discover.tinker_utils.completers import TokensWithLogprobs

        tinker = self._tinker
        advantages = np.asarray(advantages, dtype=float)
        datums = []
        trained_samples = 0
        trained_tokens = 0
        for sample, advantage in zip(group.samples, advantages):
            if sample.token_ids is None or sample.old_logprobs is None:
                continue
            if sample.prompt_token_ids is None:
                continue
            response = [int(t) for t in np.asarray(sample.token_ids).reshape(-1)]
            logprobs = [float(x) for x in np.asarray(sample.old_logprobs).reshape(-1)]
            mask = (
                [float(x) for x in np.asarray(sample.response_mask, dtype=float).reshape(-1)]
                if sample.response_mask is not None
                else [1.0] * len(response)
            )
            prompt_ids = [int(t) for t in np.asarray(sample.prompt_token_ids).reshape(-1)]
            # Per-sample validation: a single length mismatch would otherwise
            # trip the assert in trajectory_to_data and kill the WHOLE group's
            # update. Skip the bad sample instead.
            if not response or not prompt_ids or not (len(response) == len(logprobs) == len(mask)):
                logger.warning(
                    "Skipping misaligned rollout sample: prompt=%d tokens=%d "
                    "logprobs=%d mask=%d",
                    len(prompt_ids), len(response), len(logprobs), len(mask),
                )
                continue
            action = TokensWithLogprobs(
                tokens=response, maybe_logprobs=logprobs, maybe_mask=mask
            )
            trajectory = Trajectory(
                transitions=[
                    Transition(
                        ob=self._model_input(prompt_ids),
                        ac=action,
                        reward=0.0,
                        episode_done=True,
                    )
                ],
                final_ob=tinker.ModelInput.empty(),
            )
            datums.extend(
                data_processing.trajectory_to_data(trajectory, float(advantage))
            )
            trained_samples += 1
            trained_tokens += int(sum(mask))

        if not datums:
            return {
                "loss": float("nan"),
                "num_valid_tokens": 0,
                "clip_fraction": float("nan"),
                "trained_samples": 0,
            }

        # forward_backward does not accept the "mask" loss input (advantages
        # already zero the prompt tokens); strip it, mirroring ttt_discover.train.
        stripped = [
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    key: value
                    for key, value in datum.loss_fn_inputs.items()
                    if key != "mask"
                },
            )
            for datum in datums
        ]
        fb_future = self._run(
            self._training_client.forward_backward_async(
                stripped, loss_fn=self.loss_fn
            )
        )
        fb_result = self._run(fb_future.result_async())
        opt_future = self._run(
            self._training_client.optim_step_async(
                tinker.AdamParams(
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    eps=self.adam_eps,
                )
            )
        )
        self._run(opt_future.result_async())

        loss_value = _resolve_fb_loss(fb_result)
        # clip_fraction is unknown for Tinker's internal loss; report NaN rather
        # than a fake 0.0 so logs don't imply clipping that isn't measured here.
        return {
            "loss": loss_value,
            "num_valid_tokens": trained_tokens,
            "clip_fraction": float("nan"),
            "trained_samples": trained_samples,
        }

    def sync_weights(self) -> None:
        self._sampling_client = self._run(
            self._training_client.save_weights_and_get_sampling_client_async()
        )
