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

"""Test-only fake task evaluator. Conda-free and deterministic: it compiles the
candidate in-process and reads its score from a ``# SCORE: <float>`` marker
(default 1.0). This lets wiring tests inject a known reward without a real
evaluator, network, or GPU. Satisfies the task eval contract:
``EvalConfig``, ``recompile_library``, ``evaluate_dataset``, ``parse_eval_results``.
"""

import ast
import dataclasses
import os
import re

from task_utils import CompletedProcess

_SCORE_RE = re.compile(r"#\s*SCORE:\s*(-?\d+\.?\d*)")
_CANDIDATE_RE = re.compile(r"Candidate:\s*(\{.+?\})")


@dataclasses.dataclass
class EvalConfig:
    """One fake evaluation scenario."""
    dataset: str


def _target_path(config: dict) -> str:
    src_path = os.path.expanduser(config["paths"]["src_path"])
    return os.path.join(src_path, config["paths"]["target_file_path"])


def _read_target(config: dict) -> str | None:
    try:
        with open(_target_path(config), "r") as f:
            return f.read()
    except OSError:
        return None


def recompile_library(config: dict) -> CompletedProcess:
    src = _read_target(config)
    if src is None:
        return CompletedProcess(args="compile", returncode=-1, stdout="",
                                stderr=f"target not found: {_target_path(config)}")
    try:
        compile(src, _target_path(config), "exec")
    except SyntaxError as e:
        return CompletedProcess(args="compile", returncode=1, stdout="",
                                stderr=f"error: {e}")
    return CompletedProcess(args="compile", returncode=0, stdout="compiled", stderr="")


def evaluate_dataset(candidate_id, baseline_id, eval_config, config) -> CompletedProcess:
    src = _read_target(config)
    if src is None:
        return CompletedProcess(args="eval", returncode=-1, stdout="",
                                stderr=f"target not found: {_target_path(config)}")
    m = _SCORE_RE.search(src)
    score = float(m.group(1)) if m else 1.0
    stdout = f"Candidate: {{'score': {score}}}"
    return CompletedProcess(args="eval", returncode=0, stdout=stdout, stderr="")


def parse_eval_results(eval_results):
    if isinstance(eval_results, str):
        m = _CANDIDATE_RE.search(eval_results)
        if not m:
            return None
        try:
            return float(ast.literal_eval(m.group(1))["score"])
        except (ValueError, SyntaxError, KeyError, TypeError):
            return None
    if isinstance(eval_results, list):
        parsed = [parse_eval_results(r) for r in eval_results]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            return None
        return parsed[0] if len(parsed) == 1 else parsed
    raise ValueError("Input must be a string or a list of strings.")
