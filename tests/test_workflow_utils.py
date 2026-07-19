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

"""Regression tests for implementer-response extraction and edit application."""

import importlib

import pytest

import llm_utils
import task_utils
import workflow_utils


START_TAG = "RegexTagEvolveStart"
END_TAG = "RegexTagEvolveEnd"
SOURCE = f"""int fixed_before = 1;
// {START_TAG}
int old_candidate = 0;
// {END_TAG}
int fixed_after = 2;
"""
CANDIDATE = """static int evolve_solve() {
  return 7;
}"""


@pytest.fixture
def compile_harness(tmp_path, monkeypatch):
  target = tmp_path / "solution.cpp"
  target.write_text(SOURCE)
  compiler_calls = []

  eval_utils = importlib.import_module("tasks.fake_task.eval.eval_utils")

  def successful_compile(config):
    compiler_calls.append(config)
    return task_utils.CompletedProcess(
        args="compile", returncode=0, stdout="compiled", stderr=""
    )

  monkeypatch.setattr(eval_utils, "recompile_library", successful_compile)
  config = {
      "experiment": {"task_id": "fake_task"},
      "compilation": {
          "edit_start_tag": START_TAG,
          "edit_end_tag": END_TAG,
      },
      "summarization": {
          "context_lines_after_error": 1,
          "max_summary_lines": 10,
      },
  }
  compile_config = task_utils.CompilationConfig(target_file_path=str(target))

  def run_response(response, trial=None):
    transcript = llm_utils.Transcript()
    transcript.append(llm_utils.ContentChunk(response, "model"))
    trial = workflow_utils.edit_until_compile(
        "unused-model",
        trial or workflow_utils.AlgorithmTrial(),
        transcript,
        compile_config,
        config,
        loop_config={
            "max_attempts": 1,
            "loop_tag": "compile_loop",
            "summary_tag": "compile_summary",
        },
    )
    return trial, transcript

  return target, compiler_calls, config, compile_config, run_response


def expected_source(candidate=CANDIDATE):
  return f"""int fixed_before = 1;
// {START_TAG}
{candidate}
// {END_TAG}
int fixed_after = 2;
"""


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            f"// {START_TAG}\n{CANDIDATE}\n// {END_TAG}",
            id="clean-tagged-edit",
        ),
        pytest.param(
            f"// {START_TAG}\n```cpp\n{CANDIDATE}\n```\n// {END_TAG}",
            id="fence-wrapped-tagged-edit",
        ),
        pytest.param(
            f"// {START_TAG}\n````cpp\n{CANDIDATE}\n````\n// {END_TAG}",
            id="long-fence-wrapped-tagged-edit",
        ),
        pytest.param(
            f"```cpp\n// {START_TAG}\n{CANDIDATE}\n```\n// {END_TAG}",
            id="fence-intersects-tagged-edit",
        ),
        pytest.param(
            "Wait, here is the complete edit.\n"
            f"// {START_TAG}\n{CANDIDATE}\n// {END_TAG}\n"
            "This replaces only the evolvable region.",
            id="prose-around-tagged-edit",
        ),
        pytest.param(
            f"```cpp\n// {START_TAG}\n{CANDIDATE}\n// {END_TAG}\n```",
            id="fence-containing-tags",
        ),
    ],
)
def test_tagged_implementer_formatting_is_normalized(compile_harness, response):
  target, compiler_calls, _, _, run_response = compile_harness

  trial, _ = run_response(response)

  assert trial.compile_success
  assert trial.algorithm_implementation == CANDIDATE
  assert target.read_text() == expected_source()
  assert len(compiler_calls) == 1


def test_legacy_single_fenced_block_remains_byte_identical(compile_harness):
  target, compiler_calls, _, _, run_response = compile_harness
  candidate = "  static int evolve_solve() {\n    return 9;  \n  }"

  trial, _ = run_response(f"Reasoning first.\n```cpp\n{candidate}\n```\nDone.")

  assert trial.compile_success
  assert trial.algorithm_implementation == candidate
  assert target.read_text() == expected_source(candidate)
  assert len(compiler_calls) == 1


def test_mangled_response_uses_failed_edit_path(compile_harness):
  target, compiler_calls, _, _, run_response = compile_harness
  response = (
      "Wait, this is not ready.\n"
      f"// {START_TAG}\n"
      f"```cpp\n{CANDIDATE}\n```"
  )

  trial, transcript = run_response(
      response, workflow_utils.AlgorithmTrial(compile_success=True)
  )

  assert not trial.compile_success
  assert target.read_text() == SOURCE
  assert compiler_calls == []
  assert transcript[-1].role == "system"
  assert "Compilation failed" in transcript[-1].content


def test_multiple_untagged_code_blocks_are_ambiguous(compile_harness):
  target, compiler_calls, _, _, run_response = compile_harness
  response = "```cpp\nint first = 1;\n```\n```cpp\nint second = 2;\n```"

  trial, _ = run_response(response)

  assert not trial.compile_success
  assert target.read_text() == SOURCE
  assert compiler_calls == []


def test_nested_language_fence_is_rejected_before_compile(compile_harness):
  target, compiler_calls, _, _, run_response = compile_harness
  response = (
      "```cpp\n"
      "Wait, I need to revise this.\n"
      "```cpp\n"
      f"{CANDIDATE}\n"
      "```"
  )

  trial, _ = run_response(
      response, workflow_utils.AlgorithmTrial(compile_success=True)
  )

  assert not trial.compile_success
  assert target.read_text() == SOURCE
  assert compiler_calls == []


def test_nested_bare_fences_are_rejected_before_compile(compile_harness):
  target, compiler_calls, _, _, run_response = compile_harness
  response = (
      "```\n"
      "Wait, I need to revise this.\n"
      "```\n"
      f"{CANDIDATE}\n"
      "```\n"
      "```"
  )

  trial, _ = run_response(response)

  assert not trial.compile_success
  assert target.read_text() == SOURCE
  assert compiler_calls == []


@pytest.mark.parametrize(
    ("candidate", "error_fragment"),
    [
        pytest.param(
            f"{CANDIDATE}\n```cpp\nint stray = 1;",
            "Markdown fence",
            id="stray-fence-line",
        ),
        pytest.param(
            f"{CANDIDATE}\n// {START_TAG}",
            "edit tag",
            id="leftover-edit-tag",
        ),
    ],
)
def test_precompile_sanity_gate_skips_compiler(
    compile_harness, candidate, error_fragment
):
  target, compiler_calls, config, compile_config, _ = compile_harness
  trial = workflow_utils.AlgorithmTrial(
      algorithm_implementation=candidate,
      compile_success=True,
  )

  result, message = workflow_utils.attempt_compile(
      trial, compile_config, config
  )

  assert not result.compile_success
  assert target.read_text() == SOURCE
  assert compiler_calls == []
  assert "rejected before compilation" in message
  assert error_fragment in message
