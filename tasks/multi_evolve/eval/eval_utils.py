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

"""Scoring math and evaluation-plugin skeleton for Multi-Evolve."""

import ast
import dataclasses
import logging
import os
import re
import shlex

import numpy as np

from task_utils import CompletedProcess, _call_shell_command


logger = logging.getLogger("controller")
_CANDIDATE_RE = re.compile(r"Candidate:\s*(\{.+?\})")


def _paired_vectors(y_true, y_pred):
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.size != pred.size:
        raise ValueError("y_true and y_pred must contain the same number of items")
    return true, pred


def pearson_r(y_true, y_pred) -> float:
    """Return the NumPy Pearson correlation, or zero for a constant input."""

    true, pred = _paired_vectors(y_true, y_pred)
    if true.size == 0 or np.std(true) == 0.0 or np.std(pred) == 0.0:
        return 0.0
    return float(np.corrcoef(true, pred)[0, 1])


def precision_at_5(y_true, y_pred) -> float:
    """Return overlap between the true and predicted top five item sets."""

    true, pred = _paired_vectors(y_true, y_pred)
    k = min(5, true.size)
    if k == 0:
        return 0.0

    # Stable ascending argsort makes index the deterministic secondary key.
    true_top = np.argsort(true, kind="stable")[-k:]
    pred_top = np.argsort(pred, kind="stable")[-k:]
    return float(np.intersect1d(true_top, pred_top).size / k)


def combined_score(y_true, y_pred) -> float:
    """Return the Multi-Evolve 70/30 correlation and top-five score."""

    return 0.7 * pearson_r(y_true, y_pred) + 0.3 * precision_at_5(
        y_true, y_pred
    )


@dataclasses.dataclass
class EvalConfig:
    """Evaluation configuration for one Multi-Evolve dataset."""

    dataset: str


def recompile_library(config: dict) -> CompletedProcess:
    """Compile-check the candidate with the configured Python interpreter."""

    compilation = config["compilation"]
    python_bin = compilation.get("python_bin", "python3")
    candidate_script = os.path.join(
        os.path.expanduser(config["paths"]["src_path"]),
        config["paths"]["target_file_path"],
    )
    command = (
        f"{shlex.quote(python_bin)} -m py_compile "
        f"{shlex.quote(candidate_script)}"
    )
    logger.info("recompile_library: Running command: %s", command)
    process_result = _call_shell_command(
        command,
        timeout=compilation["recompile_timeout"],
        max_retries=compilation["recompile_max_retries"],
    )
    if not process_result:
        return CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr="Compilation command failed to complete.",
        )
    return CompletedProcess(
        args=command,
        returncode=process_result.returncode,
        stdout=process_result.stdout.strip(),
        stderr=process_result.stderr.strip(),
    )


def evaluate_dataset(
    candidate_id: int,
    baseline_id: int,
    eval_config: EvalConfig,
    config: dict,
) -> CompletedProcess:
    """Explain why the external Multi-Evolve evaluator cannot run here."""

    del candidate_id, baseline_id, config
    raise FileNotFoundError(
        "Multi-Evolve dataset "
        f"{eval_config.dataset!r} is unavailable: the Multi-Evolve datasets "
        "from Tran et al. (2026) are external and are not shipped with this "
        "repository. Install those datasets and wire the GPU evaluator before "
        "running candidate evaluation."
    )


def parse_eval_results(
    eval_results: list[str] | str,
) -> list[float] | float | None:
    """Extract score values from evaluator `Candidate: {...}` output."""

    if isinstance(eval_results, str):
        match = _CANDIDATE_RE.search(eval_results)
        if not match:
            return None
        try:
            data = ast.literal_eval(match.group(1))
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
        return scores

    raise ValueError("Input must be a string or a list of strings.")
