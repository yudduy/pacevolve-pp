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

"""PACEvolve evaluation-plugin contract for the EPLB task."""

import ast
from dataclasses import dataclass
import logging
import os
import re

from task_utils import CompletedProcess, _call_shell_command


logger = logging.getLogger("controller")
_CANDIDATE_RE = re.compile(r"Candidate:\s*(\{.+?\})")


@dataclass
class EvalConfig:
    """Evaluation configuration for one synthetic load-profile dataset."""

    dataset: str


def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config["compilation"]
    python_bin = comp_config.get("python_bin", "python3")
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    src_path = os.path.expanduser(config["paths"]["src_path"])
    eval_script = os.path.join(
        eval_path, config["evaluation"]["eval_script_name"]
    )
    candidate_script = os.path.join(
        src_path, config["paths"]["target_file_path"]
    )
    command = (
        f"{python_bin} {eval_script} --candidate_path {candidate_script} "
        "--compile_only"
    )
    logger.info("recompile_library: Running command: %s", command)
    process_result = _call_shell_command(
        command,
        timeout=comp_config["recompile_timeout"],
        max_retries=comp_config["recompile_max_retries"],
    )

    if not process_result:
        return CompletedProcess(
            args=command,
            returncode=-1,
            stdout="",
            stderr="Compilation command failed to complete.",
        )

    logger.info(
        "recompile_library: Success: %s.", process_result.returncode == 0
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
    python_bin = config["compilation"].get("python_bin", "python3")
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    src_path = os.path.expanduser(config["paths"]["src_path"])
    eval_script = os.path.join(
        eval_path, config["evaluation"]["eval_script_name"]
    )
    candidate_script = os.path.join(
        src_path, config["paths"]["target_file_path"]
    )
    evaluation = config["evaluation"]
    eval_command = (
        f"{python_bin} {eval_script} --candidate_path {candidate_script} "
        f"--num_experts {evaluation['num_experts']} "
        f"--num_devices {evaluation['num_devices']} "
        f"--num_profiles {evaluation['num_profiles']} "
        f"--seed {evaluation['seed']} "
        f"--ref_time {evaluation['ref_time']}"
    )

    logger.info("evaluate_dataset: Running %s", eval_command)
    process_result = _call_shell_command(
        eval_command,
        timeout=evaluation["eval_timeout"],
        max_retries=evaluation["eval_max_retries"],
    )
    if not process_result:
        logger.error(
            "evaluate_dataset: Evaluation for %s failed.", eval_config.dataset
        )
        return CompletedProcess(
            args=eval_command,
            returncode=-1,
            stdout="",
            stderr=(
                f"evaluate_dataset for {eval_config.dataset} failed to complete."
            ),
        )
    return process_result


def parse_eval_results(eval_results: list[str] | str) -> float | None:
    if isinstance(eval_results, str):
        match = _CANDIDATE_RE.search(eval_results)
        if not match:
            return None
        try:
            result = ast.literal_eval(match.group(1))
            return float(result["score"])
        except (ValueError, SyntaxError, KeyError, TypeError):
            return None

    if isinstance(eval_results, list):
        for result in eval_results:
            parsed = parse_eval_results(result)
            if parsed is not None:
                return parsed
        return None

    raise ValueError("Input must be a string or a list of strings.")
