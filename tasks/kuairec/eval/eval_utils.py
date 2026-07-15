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

"""Evaluation-plugin skeleton for KuaiRec sequential recommendation."""

import ast
import dataclasses
import re


_CANDIDATE_RE = re.compile(r"Candidate:\s*(\{.+?\})")
_EVALUATOR_TODO = (
    "TODO: wire the external KuaiRec dataset and a GPU evaluator; every "
    "candidate requires 16 epochs of sampled-softmax training."
)


@dataclasses.dataclass
class EvalConfig:
    """Evaluation configuration for one KuaiRec dataset split."""

    dataset: str


def recompile_library(config: dict):
    """Stop until the dataset/GPU-backed candidate harness is available."""

    del config
    raise NotImplementedError(_EVALUATOR_TODO)


def evaluate_dataset(
    candidate_id: int,
    baseline_id: int,
    eval_config: EvalConfig,
    config: dict,
):
    """Stop until the dataset/GPU-backed 16-epoch evaluator is available."""

    del candidate_id, baseline_id, eval_config, config
    raise NotImplementedError(_EVALUATOR_TODO)


def parse_eval_results(
    eval_results: list[str] | str,
) -> float | None:
    """Extract one score or the mean across dataset outputs."""

    if isinstance(eval_results, str):
        matches = list(_CANDIDATE_RE.finditer(eval_results))
        if not matches:
            return None
        try:
            data = ast.literal_eval(matches[-1].group(1))
            return float(data["score"])
        except (ValueError, SyntaxError, KeyError, TypeError):
            return None

    if isinstance(eval_results, list):
        parsed = [parse_eval_results(result) for result in eval_results]
        scores = [score for score in parsed if score is not None]
        if len(scores) == 1:
            return scores[0]
        if not scores:
            return None
        return float(sum(scores) / len(scores))

    raise ValueError("Input must be a string or a list of strings.")
