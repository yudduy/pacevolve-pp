"""PACEvolve evaluation-plugin contract for the rectangle-free-grid task.

Same interface the runners import dynamically as tasks.<task_id>.eval.eval_utils:
  - EvalConfig
  - recompile_library(config) -> CompletedProcess   (C++ compile check)
  - evaluate_dataset(candidate_id, baseline_id, eval_config, config) -> CompletedProcess
  - parse_eval_results(eval_results) -> float | None

Unlike the EPLB task (which evals a Python module), the candidate here is C++, so the
"compile" step is a g++/clang++ build and the "eval" step runs the compiled binary over
a fixed (n,m) set and scores it. Both are shell commands, matching the language-agnostic
_call_shell_command contract.
"""

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
    """One evaluation group. RFG uses a single fixed (n,m) set, so one config."""

    dataset: str


def _resolve(config: dict):
    comp = config["compilation"]
    python_bin = comp.get("python_bin", "python3")
    eval_path = os.path.expanduser(config["paths"]["eval_path"])
    src_path = os.path.expanduser(config["paths"]["src_path"])
    eval_script = os.path.join(eval_path, config["evaluation"]["eval_script_name"])
    solution = os.path.join(src_path, config["paths"]["target_file_path"])
    build_dir = os.path.expanduser(
        config["paths"].get("build_dir", config["paths"].get("results_path", "."))
    )
    cxx = comp.get("cxx")  # None -> eval_rfg auto-detects g++/clang++
    return python_bin, eval_script, solution, build_dir, cxx


def _cmd(python_bin, eval_script, solution, build_dir, cxx, extra=""):
    cmd = (
        f'{python_bin} "{eval_script}" --solution "{solution}" '
        f'--build_dir "{build_dir}"{extra}'
    )
    if cxx:
        cmd += f' --cxx {cxx}'
    return cmd


def recompile_library(config: dict) -> CompletedProcess:
    comp = config["compilation"]
    python_bin, eval_script, solution, build_dir, cxx = _resolve(config)
    os.makedirs(build_dir, exist_ok=True)
    command = _cmd(python_bin, eval_script, solution, build_dir, cxx, extra=" --compile_only")
    logger.info("recompile_library: Running command: %s", command)
    process_result = _call_shell_command(
        command,
        timeout=comp["recompile_timeout"],
        max_retries=comp["recompile_max_retries"],
    )
    if not process_result:
        return CompletedProcess(
            args=command, returncode=-1, stdout="",
            stderr="Compilation command failed to complete.",
        )
    logger.info("recompile_library: Success: %s.", process_result.returncode == 0)
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
    python_bin, eval_script, solution, build_dir, cxx = _resolve(config)
    evaluation = config["evaluation"]
    tl = evaluation.get("time_limit", 1.0)
    os.makedirs(build_dir, exist_ok=True)
    eval_command = _cmd(
        python_bin, eval_script, solution, build_dir, cxx, extra=f" --tl {tl}"
    )
    logger.info("evaluate_dataset: Running %s", eval_command)
    process_result = _call_shell_command(
        eval_command,
        timeout=evaluation["eval_timeout"],
        max_retries=evaluation["eval_max_retries"],
    )
    if not process_result:
        logger.error("evaluate_dataset: Evaluation for %s failed.", eval_config.dataset)
        return CompletedProcess(
            args=eval_command, returncode=-1, stdout="",
            stderr=f"evaluate_dataset for {eval_config.dataset} failed to complete.",
        )
    return process_result


def parse_eval_results(eval_results) -> float | None:
    if isinstance(eval_results, str):
        matches = list(_CANDIDATE_RE.finditer(eval_results))
        if not matches:
            return None
        try:
            result = ast.literal_eval(matches[-1].group(1))
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
