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

"""Smoke test for workflows/run_advisor_rl.py — drives the barrier loop end to
end against the fake task with a scripted LLM (no network) and a mock policy
backend, asserting the paper's population-update-before-policy-update order."""

import importlib
import os
import shutil
import types

import yaml

import idea_select_utils
import llm_utils
import rl_rewards
import run_advisor_rl
import task_utils
from program_database import ProgramsDatabase, ProgramsDatabaseConfig
from rl_trainer import AdvisorTrainer, AdvisorTrainerConfig, MockPolicyBackend

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIX = os.path.join(_REPO, "tests", "fixtures", "tasks", "fake_task")


def _config(tmp_path):
    with open(os.path.join(_FIX, "config", "config_1.yaml")) as f:
        config = yaml.safe_load(f)
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(os.path.join(_FIX, "src", "fake_1.py"), src / "fake_1.py")
    config["paths"]["src_path"] = str(src)
    config["paths"]["target_file_path"] = "fake_1.py"
    config["database"]["num_islands"] = 1  # deterministic single-island routing
    return config


def _scripted_generate():
    """A generate_completion replacement that dispatches on the last prompt and
    returns varying candidate scores so the RL branch does not collapse."""
    state = {"n": 0}

    def fake(name, transcript, config):
        prompt = transcript[-1].content if len(transcript) else ""
        if "Generate ideas" in prompt:
            return "Idea 1\nHypothesis: scale the input\nReasoning: larger outputs help"
        if "Idea Exists" in prompt:  # classification prompt from idea_select_utils
            return "Idea Exists: False\nIdea description: scale the input by a factor"
        if "Implement the selected" in prompt:  # code implementation
            score = 2.0 + 0.3 * state["n"]
            state["n"] += 1
            return f"```python\n# SCORE: {score}\ndef solve(x):\n    return x\n```"
        if "Select an idea" in prompt:  # idea selection (no code)
            return "Idea ID: 1\nExperiment description: scale by a factor"
        return "ok"

    return fake


class OrderedMockBackend(MockPolicyBackend):
    def __init__(self, config, order_log):
        super().__init__(config)
        self._order = order_log

    def update(self, group, advantages, clip):
        self._order.append("update")
        return super().update(group, advantages, clip)


def _harness(config, objective, monkeypatch, order_log):
    monkeypatch.setattr(llm_utils, "generate_completion", _scripted_generate())
    prompts = importlib.import_module("tasks.fake_task.config.prompts")
    eval_utils = importlib.import_module("tasks.fake_task.eval.eval_utils")
    n_islands = config["database"]["num_islands"]

    db = ProgramsDatabase(
        ProgramsDatabaseConfig(num_islands=n_islands, tournament_size=2, top_k=2, max_queue_size=50),
        template=prompts.FAKE_SOTA, function_to_evolve="fake_task", metric_direction="max")
    idea_repo_db = idea_select_utils.IdeaRepoDatabase(
        num_islands=n_islands, target_score=config["evaluation"]["target_score"], metric_direction="max")
    for i in range(n_islands):
        db.register_program(program=prompts.FAKE_SOTA, island_id=i, score=config["evaluation"]["init_score"])
        idea_repo_db.best_scores_history[i].append(config["evaluation"]["init_score"])
        idea_repo_db.scheduler.update_score(i, config["evaluation"]["init_score"])
        repo = idea_select_utils.IdeaRepo()
        repo.sota = prompts.FAKE_SOTA
        idea_repo_db.idea_repos[i].append(repo)

    original_register = db.register_program

    def logged_register(*args, **kwargs):
        order_log.append("register")
        return original_register(*args, **kwargs)

    db.register_program = logged_register

    backend = OrderedMockBackend({}, order_log)
    trainer = AdvisorTrainer(
        AdvisorTrainerConfig(objective=objective, n_samples=4, top_k=2, total_steps=2), backend)
    compile_config = task_utils.CompilationConfig(
        target_file_path=os.path.join(config["paths"]["src_path"], config["paths"]["target_file_path"]))
    eval_configs = [eval_utils.EvalConfig(dataset="synthetic")]
    reward_cfg = rl_rewards.RewardShapingConfig.from_config(config)
    args = types.SimpleNamespace(n_samples=4, max_steps=2, use_idea_repo=True, _transcript_file=None)
    return db, idea_repo_db, prompts, trainer, backend, compile_config, eval_configs, reward_cfg, args


def test_smoke_registers_population_before_policy_update(tmp_path, monkeypatch):
    config = _config(tmp_path)
    order = []
    db, idea_repo_db, prompts, trainer, backend, cc, ec, rc, args = _harness(
        config, "pacevolve++", monkeypatch, order)
    run_advisor_rl.run_evolution(config, args, db, idea_repo_db, prompts, trainer,
                                 "advisor", "implementer", cc, ec, rc)
    assert "update" in order  # at least one policy update fired
    assert order.count("register") >= 4  # step-0 candidates registered
    # the first policy update is preceded by candidate registrations (3.1 order)
    assert "register" in order[: order.index("update")]
    # the idea pool accumulates across steps rather than resetting each step
    assert len(idea_repo_db.idea_repos[0]) > 1


def test_smoke_none_objective_performs_no_updates(tmp_path, monkeypatch):
    config = _config(tmp_path)
    order = []
    db, idea_repo_db, prompts, trainer, backend, cc, ec, rc, args = _harness(
        config, "none", monkeypatch, order)
    run_advisor_rl.run_evolution(config, args, db, idea_repo_db, prompts, trainer,
                                 "advisor", "implementer", cc, ec, rc)
    assert order.count("update") == 0
    assert len(backend.update_calls) == 0
    assert order.count("register") >= 4  # candidates still enter the population
