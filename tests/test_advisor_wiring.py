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

"""Spec for workflows/advisor_utils.py — the advisor(select)/implementer(code)
split, driven end-to-end against the fake task with a scripted LLM client."""

import importlib
import os
import shutil
import types

import yaml
import pytest

import advisor_utils
import llm_utils
import run_advisor_rl
import task_utils
from idea_select_utils import Idea, IdeaRepo

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIX = os.path.join(_REPO, "tests", "fixtures", "tasks", "fake_task")


class FakeLLMClient(llm_utils.LLMClient):
    """Returns scripted responses in order (last one repeats)."""

    def __init__(self, responses):
        super().__init__({})
        self.responses = list(responses)
        self.calls = 0

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    def generate(self, prompt, generation_config):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _fake_config(tmp_path):
    with open(os.path.join(_FIX, "config", "config_1.yaml")) as f:
        config = yaml.safe_load(f)
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(os.path.join(_FIX, "src", "fake_1.py"), src / "fake_1.py")
    config["paths"]["src_path"] = str(src)
    config["paths"]["target_file_path"] = "fake_1.py"
    return config


def _prompts():
    return importlib.import_module("tasks.fake_task.config.prompts")


# --- make_role_config -----------------------------------------------------

def test_make_role_config_overlays_role_section():
    config = {"llm": {"name": "base", "temperature": 1.0},
              "advisor_llm": {"name": "adv"},
              "implementer_llm": {"name": "impl", "max_output_tokens": 2048}}
    adv = advisor_utils.make_role_config(config, "advisor")
    imp = advisor_utils.make_role_config(config, "implementer")
    assert adv["llm"]["name"] == "adv" and adv["llm"]["temperature"] == 1.0
    assert imp["llm"]["name"] == "impl" and imp["llm"]["max_output_tokens"] == 2048
    assert config["llm"]["name"] == "base"  # source untouched


def test_make_role_config_missing_section_is_deep_copy():
    config = {"llm": {"name": "base"}}
    out = advisor_utils.make_role_config(config, "advisor")
    out["llm"]["name"] = "mutated"
    assert config["llm"]["name"] == "base"


def test_role_configs_yield_distinct_model_names():
    config = {"llm": {"name": "base"},
              "advisor_llm": {"name": "qwen-adv"},
              "implementer_llm": {"name": "gemini-impl"}}
    assert advisor_utils.make_role_config(config, "advisor")["llm"]["name"] == "qwen-adv"
    assert advisor_utils.make_role_config(config, "implementer")["llm"]["name"] == "gemini-impl"


# --- select_idea_no_code --------------------------------------------------

def test_select_idea_no_code_parses_valid_response():
    prompts = _prompts()
    repo = IdeaRepo(ideas=[Idea(id=2, description="scaling")])
    advisor = FakeLLMClient(["Idea ID: 2\nExperiment description: try scaling x2"])
    transcript = llm_utils.Transcript()
    idea_id, desc, raw, prompt = advisor_utils.select_idea_no_code(
        advisor, transcript, prompts, prompts.FAKE_SOTA, repo, {"llm": {}})
    assert idea_id == 2
    assert "scaling" in desc
    assert raw is not None
    assert prompt == prompts.construct_idea_select_no_code_prompt(
        prompts.FAKE_SOTA, repo
    )


def test_select_idea_no_code_retries_on_malformed():
    prompts = _prompts()
    repo = IdeaRepo(ideas=[Idea(id=1, description="x")])
    advisor = FakeLLMClient(["garbage without an id",
                             "Idea ID: 1\nExperiment description: do it"])
    transcript = llm_utils.Transcript()
    idea_id, desc, raw, prompt = advisor_utils.select_idea_no_code(
        advisor, transcript, prompts, prompts.FAKE_SOTA, repo, {"llm": {}}, max_attempts=3)
    assert idea_id == 1
    assert advisor.calls == 2  # one failed attempt, then success
    assert prompt is not None


def test_select_idea_no_code_gives_up_after_max_attempts():
    prompts = _prompts()
    repo = IdeaRepo(ideas=[Idea(id=1, description="x")])
    advisor = FakeLLMClient(["still no id"])
    idea_id, desc, raw, prompt = advisor_utils.select_idea_no_code(
        advisor, llm_utils.Transcript(), prompts, prompts.FAKE_SOTA, repo, {"llm": {}}, max_attempts=2)
    assert idea_id is None and desc is None and raw is None
    assert prompt == prompts.construct_idea_select_no_code_prompt(
        prompts.FAKE_SOTA, repo
    )


def test_build_rollout_sample_records_advisor_and_program_text(monkeypatch):
    prompts = _prompts()
    repo = IdeaRepo(ideas=[Idea(id=2, description="scaling")])
    idea_repo_db = types.SimpleNamespace(idea_repos=[[repo]])
    db = types.SimpleNamespace(
        get_candidate=lambda: (prompts.FAKE_SOTA, 0)
    )
    raw_response = "Idea ID: 2\nExperiment description: try scaling x2"
    monkeypatch.setattr(
        llm_utils,
        "generate_completion",
        lambda advisor, transcript, config: raw_response,
    )
    trial = types.SimpleNamespace(
        compile_success=True,
        eval_success=[True],
        eval_results=["Candidate: {'score': 3.0}"],
        algorithm_implementation="def solve(x):\n    return x * 2\n",
    )
    monkeypatch.setattr(
        advisor_utils, "implement_idea", lambda *args, **kwargs: trial
    )
    config = {
        "llm": {"name": "fake"},
        "experiment": {
            "task_id": "fake_task",
            "initial_baseline_id": -1,
        },
    }
    args = types.SimpleNamespace(
        _transcript_file=None, use_idea_repo=False, n_samples=1
    )

    sample = run_advisor_rl.build_rollout_sample(
        0,
        0,
        config,
        args,
        db,
        idea_repo_db,
        prompts,
        "advisor",
        "implementer",
        None,
        [],
        None,
    )

    assert sample.prompt_text.startswith("Select an idea to test")
    assert sample.response_text == raw_response
    assert sample.program_text == trial.algorithm_implementation
    assert sample.raw_score == pytest.approx(3.0)
    assert sample.eval_success


def test_build_rollout_sample_keeps_prompt_on_advisor_failure(monkeypatch):
    prompts = _prompts()
    idea_repo_db = types.SimpleNamespace(
        idea_repos=[[IdeaRepo(ideas=[Idea(id=1, description="scaling")])]]
    )
    db = types.SimpleNamespace(
        get_candidate=lambda: (prompts.FAKE_SOTA, 0)
    )
    monkeypatch.setattr(
        llm_utils,
        "generate_completion",
        lambda advisor, transcript, config: "malformed",
    )
    config = {
        "llm": {"name": "fake"},
        "experiment": {
            "task_id": "fake_task",
            "initial_baseline_id": -1,
        },
    }
    args = types.SimpleNamespace(
        _transcript_file=None, use_idea_repo=False, n_samples=1
    )

    sample = run_advisor_rl.build_rollout_sample(
        0,
        0,
        config,
        args,
        db,
        idea_repo_db,
        prompts,
        "advisor",
        "implementer",
        None,
        [],
        None,
    )

    assert sample.prompt_text.startswith("Select an idea to test")
    assert sample.response_text == ""
    assert sample.program_text == ""
    assert not sample.eval_success


# --- implement_idea -------------------------------------------------------

def test_implement_idea_splices_compiles_and_evaluates(tmp_path):
    prompts = _prompts()
    eval_utils = importlib.import_module("tasks.fake_task.eval.eval_utils")
    config = _fake_config(tmp_path)
    compile_config = task_utils.CompilationConfig(
        target_file_path=os.path.join(config["paths"]["src_path"],
                                      config["paths"]["target_file_path"]))
    eval_configs = [eval_utils.EvalConfig(dataset="synthetic")]
    implementer = FakeLLMClient(["```python\n# SCORE: 3.0\ndef solve(x):\n    return x * 2\n```"])
    trial = advisor_utils.implement_idea(
        implementer, llm_utils.Transcript(), prompts, prompts.FAKE_SOTA,
        idea_id=2, exp_description="scale by two",
        compile_config=compile_config, eval_configs=eval_configs, config=config,
        candidate_id=1, baseline_id=-1)
    assert trial.compile_success
    assert all(trial.eval_success)
    assert trial.idea_id == 2  # set from the advisor's turn, not extracted from code
    assert eval_utils.parse_eval_results(trial.eval_results) == 3.0
    # the new score marker was actually spliced into the target file
    with open(compile_config.target_file_path) as f:
        assert "# SCORE: 3.0" in f.read()
