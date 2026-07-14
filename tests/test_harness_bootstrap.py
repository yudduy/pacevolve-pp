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

"""Verifies the pytest harness: flat workflows/ modules import, and the fake
task merges into the real `tasks` namespace so importlib resolves it."""

import importlib


def test_workflows_modules_import():
    for mod in ("llm_utils", "workflow_utils", "task_utils",
                "idea_select_utils", "program_database", "crossover_utils"):
        importlib.import_module(mod)


def test_real_tasks_namespace_resolves():
    importlib.import_module("tasks.llmsr.eval.eval_utils")


def test_fake_task_merges_into_tasks_namespace():
    prompts = importlib.import_module("tasks.fake_task.config.prompts")
    eval_utils = importlib.import_module("tasks.fake_task.eval.eval_utils")
    for sym in ("FAKE_SOTA", "construct_idea_gen_prompt",
                "construct_idea_select_no_code_prompt", "construct_code_impl_prompt",
                "SUMMARIZE_EVAL_PROMPT", "EVAL_DESCRIPTION_PROMPT"):
        assert hasattr(prompts, sym), sym
    for sym in ("EvalConfig", "recompile_library", "evaluate_dataset", "parse_eval_results"):
        assert hasattr(eval_utils, sym), sym


def test_fake_evaluator_roundtrips_score():
    eval_utils = importlib.import_module("tasks.fake_task.eval.eval_utils")
    out = "Candidate: {'score': 2.5}"
    assert eval_utils.parse_eval_results(out) == 2.5
    assert eval_utils.parse_eval_results([out]) == 2.5
    assert eval_utils.parse_eval_results("no score here") is None
