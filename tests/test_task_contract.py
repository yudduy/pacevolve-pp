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

"""Task-plugin contract: every advisor/implementer-split task exports the full
prompt + eval symbol set and a complete config; plus the Multi-Evolve scorer math."""

import importlib
import os

import numpy as np
import pytest
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shipped tasks with advisor/implementer split prompts; all import with no
# file/network side effects.
SPLIT_TASKS = ["eplb", "kuairec", "multi_evolve"]
NEW_TASKS = ["eplb", "kuairec", "multi_evolve"]

CORE_PROMPTS = ["construct_idea_gen_prompt", "construct_idea_select_prompt",
                "construct_mutation_prompt", "SUMMARIZE_EVAL_PROMPT", "EVAL_DESCRIPTION_PROMPT"]
SPLIT_PROMPTS = ["construct_idea_select_no_code_prompt", "construct_code_impl_prompt"]
EVAL_SYMBOLS = ["EvalConfig", "recompile_library", "evaluate_dataset", "parse_eval_results"]
CONFIG_SECTIONS = ["llm", "experiment", "paths", "compilation", "evaluation",
                   "database", "workflow_loops"]


@pytest.mark.parametrize("task", SPLIT_TASKS)
def test_prompts_contract(task):
    prompts = importlib.import_module(f"tasks.{task}.config.prompts")
    for sym in CORE_PROMPTS + SPLIT_PROMPTS:
        assert hasattr(prompts, sym), f"{task} prompts missing {sym}"


@pytest.mark.parametrize("task", SPLIT_TASKS)
def test_eval_contract(task):
    eval_utils = importlib.import_module(f"tasks.{task}.eval.eval_utils")
    for sym in EVAL_SYMBOLS:
        assert hasattr(eval_utils, sym), f"{task} eval_utils missing {sym}"


@pytest.mark.parametrize("task", NEW_TASKS)
def test_config_sections_and_seed(task):
    with open(os.path.join(_REPO, "tasks", task, "config", "config_1.yaml")) as f:
        config = yaml.safe_load(f)
    for section in CONFIG_SECTIONS:
        assert section in config, f"{task} config missing section {section}"
    prompts = importlib.import_module(f"tasks.{task}.config.prompts")
    assert hasattr(prompts, config["experiment"]["sota_algo_name"])


# --- Multi-Evolve scorer math (real, unit-tested) ------------------------

def _multi_evolve_eval():
    return importlib.import_module("tasks.multi_evolve.eval.eval_utils")


def test_multi_evolve_combined_score_formula():
    ev = _multi_evolve_eval()
    y_true = np.arange(6.0)
    y_pred = y_true[::-1]
    assert ev.combined_score(y_true, y_pred) == pytest.approx(
        0.7 * -1.0 + 0.3 * (4 / 5)
    )


def test_multi_evolve_precision_at_5_perfect_ranking():
    ev = _multi_evolve_eval()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert ev.precision_at_5(y, y) == pytest.approx(1.0)


def test_multi_evolve_pearson_constant_is_zero():
    ev = _multi_evolve_eval()
    assert ev.pearson_r(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])) == 0.0


def test_multi_evolve_pearson_anticorrelated_is_negative_one():
    ev = _multi_evolve_eval()
    assert ev.pearson_r([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)


def test_multi_evolve_precision_at_5_known_overlap():
    ev = _multi_evolve_eval()
    y_true = np.arange(10.0)
    y_pred = np.array([5.0, 6.0, 7.0, 2.0, 1.0, 0.0, 3.0, 4.0, 8.0, 9.0])
    assert ev.precision_at_5(y_true, y_pred) == pytest.approx(2 / 5)


@pytest.mark.parametrize("task", NEW_TASKS)
def test_parse_eval_results_uses_last_candidate_line(task):
    eval_utils = importlib.import_module(f"tasks.{task}.eval.eval_utils")
    output = (
        "Candidate: {'score': 1.0, 'valid': True}\n"
        "Candidate: {'score': 0.25, 'valid': True}"
    )
    assert eval_utils.parse_eval_results(output) == pytest.approx(0.25)


@pytest.mark.parametrize("task", ["kuairec", "multi_evolve"])
def test_parse_eval_results_means_multiple_datasets(task):
    eval_utils = importlib.import_module(f"tasks.{task}.eval.eval_utils")
    outputs = [
        "Candidate: {'score': 0.5}",
        "Candidate: {'score': 0.7}",
    ]
    assert eval_utils.parse_eval_results(outputs) == pytest.approx(0.6)


# --- KuaiRec is a contract-only skeleton (not runnable) ------------------

def test_kuairec_evaluate_is_not_implemented():
    ev = importlib.import_module("tasks.kuairec.eval.eval_utils")
    with pytest.raises(NotImplementedError):
        ev.evaluate_dataset(1, -1, ev.EvalConfig(dataset="kuairec"), {})
