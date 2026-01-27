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

import dataclasses
import time
import os
from task_utils import CompletedProcess, _call_shell_command
import logging


logger = logging.getLogger("controller")
@dataclasses.dataclass
class EvalConfig:
  """Evaluation configuration for a single consistent hashing scenario."""
  dataset: str
  epsilons: list[float]
  servers: int
  objects_list: list[int]
  duplicates_list: list[int]
  repetitions: int

def recompile_library(config: dict) -> CompletedProcess:
    comp_config = config['compilation']
    CONDA_PREFIX = f"conda run -n {comp_config['conda_env']} "
    EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
    EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
    )
    CONDA_PREFIX = f"conda run -n {config['compilation']['conda_env']} "
    command = (
      f"{CONDA_PREFIX} python {EVAL_SCRIPT} "
    )
    logger.info(f"recompile_library: Running command: {command}")
    process_result = _call_shell_command(
      command, timeout=comp_config['recompile_timeout'], max_retries=comp_config['recompile_max_retries']
    )

    if not process_result:
      return CompletedProcess(
        args=command,
        returncode=-1,
        stdout="",
        stderr="Compilation command failed to complete.",
      )

    success = (process_result.returncode == 0)
    logger.info(f"recompile_library: Success: {success}.")
    for line in process_result.stdout.splitlines():
      logger.debug(f"recompile_library: STDOUT: {line}")
    for line in process_result.stderr.splitlines():
      logger.debug(f"recompile_library: STDERR: {line}")
    return CompletedProcess(
      args=command,
      returncode=process_result.returncode,
      stdout=process_result.stdout.strip(),
      stderr=process_result.stderr.strip()
    )


def evaluate_dataset(
  candidate_id: int,
  baseline_id: int,
  eval_config: EvalConfig,
  config: dict,
) -> CompletedProcess:
  EVAL_PATH = os.path.expanduser(config['paths']['eval_path'])
  RESULTS_PATH = os.path.expanduser(config['paths']['results_path'])
  EVAL_SCRIPT = os.path.join(
      EVAL_PATH, config['evaluation']['eval_script_name']
  )
  # BASELINE_DIFF_SCRIPT = os.path.join(EVAL_PATH, config['evaluation']['baseline_diff_script_name'])
  CONDA_PREFIX = f"conda run -n {config['compilation']['conda_env']} "
  results_dir = os.path.join(RESULTS_PATH, eval_config.dataset)
  output_file = os.path.join(results_dir, f"candidate_{candidate_id}.pickle")

  eval_command = (
    f"{CONDA_PREFIX} python {EVAL_SCRIPT} --config_path {config['config_path']} --dataset {eval_config.dataset} --output {output_file}"
  )

  try:
    os.makedirs(results_dir, exist_ok=True)
  except OSError as e:
    logger.error(
      f"evaluate_dataset: Could not create results directory {results_dir}: {e}"
    )
    return CompletedProcess(
      args=eval_command,
      returncode=-1,
      stdout="",
      stderr=f"Could not create results directory {results_dir}: {e}"
    )

  logger.info(f"evaluate_dataset: Running {eval_command}")
  process_result_eval = _call_shell_command(
    eval_command, timeout=config['evaluation']['eval_timeout'], max_retries=config['evaluation']['eval_max_retries']
  )
  if not process_result_eval:
    logger.error(
      f"evaluate_dataset: evaluate_dataset for {eval_config.dataset} failed."
    )
    return CompletedProcess(
      args=eval_command,
      returncode=-1,
      stdout="",
      stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete."
    )
  return process_result_eval
