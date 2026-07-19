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

from copy import deepcopy
import importlib
import os
import shutil
import sys
import threading
import types

import pytest
import yaml

import advisor_utils
import idea_select_utils
import llm_utils
import rl_rewards
import run_advisor_rl
import task_utils
from program_database import ProgramsDatabase, ProgramsDatabaseConfig
from rl_trainer import (
    AdvisorTrainer,
    AdvisorTrainerConfig,
    MockPolicyBackend,
    RolloutGroup,
    RolloutSample,
)

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


def test_parallel_group_isolates_workers_and_preserves_step_seed(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    config["rl"].update(
        {
            "advisor_flow": "propose",
            "parallel_rollouts": True,
        }
    )
    config["paths"]["results_path"] = str(tmp_path / "results")
    config["paths"]["build_dir"] = str(tmp_path / "build")
    order = []
    real_generate_completion = llm_utils.generate_completion
    db, idea_repo_db, prompts, _, backend, cc, ec, rc, args = _harness(
        config, "none", monkeypatch, order
    )
    monkeypatch.setattr(
        llm_utils, "generate_completion", real_generate_completion
    )

    class StaticImplementer(llm_utils.LLMClient):
        def count_tokens(self, text):
            return max(1, len(text) // 4)

        def generate(self, prompt, generation_config):
            del prompt, generation_config
            return (
                "```python\n# SCORE: 3.0\n"
                "def solve(x):\n    return x\n```"
            )

    advisor = run_advisor_rl.rl_trainer.BackendLLMClient(
        backend, advisor_utils.make_role_config(config, "advisor")
    )
    barrier = threading.Barrier(4)
    seen = []
    seen_lock = threading.Lock()
    original_implement_idea = advisor_utils.implement_idea

    def record_worker(*call_args, **call_kwargs):
        worker_compile_config = call_args[6]
        worker_eval_configs = call_args[7]
        worker_config = call_args[8]
        with open(worker_compile_config.target_file_path) as worker_file:
            worker_source = worker_file.read()
        has_current_snapshot = "# CURRENT SNAPSHOT" in worker_source
        has_slot_mutation = "# SLOT MUTATION" in worker_source
        with seen_lock:
            seen.append(
                (
                    worker_config["paths"]["src_path"],
                    worker_config["paths"]["build_dir"],
                    worker_config["paths"]["results_path"],
                    worker_compile_config.target_file_path,
                    id(worker_eval_configs),
                    os.environ.get("PACE_EVAL_CASE_SEED"),
                    has_current_snapshot,
                    has_slot_mutation,
                )
            )
        barrier.wait(timeout=5)
        trial = original_implement_idea(*call_args, **call_kwargs)
        with open(worker_compile_config.target_file_path, "a") as worker_file:
            worker_file.write("\n# SLOT MUTATION\n")
        return trial

    monkeypatch.setattr(advisor_utils, "implement_idea", record_worker)
    monkeypatch.setenv("PACE_EVAL_CASE_SEED", "step-seed")
    canonical_source = (tmp_path / "src" / "fake_1.py").read_text()

    group = run_advisor_rl.build_rollout_group(
        0,
        config,
        args,
        db,
        idea_repo_db,
        prompts,
        advisor,
        StaticImplementer({}),
        cc,
        ec,
        rc,
    )

    assert len(group.samples) == 4
    assert [sample.raw_score for sample in group.samples] == [3.0] * 4
    assert all(sample.eval_success for sample in group.samples)
    assert len({record[0] for record in seen}) == 4
    assert len({record[1] for record in seen}) == 4
    assert len({record[2] for record in seen}) == 4
    assert len({record[3] for record in seen}) == 4
    assert len({record[4] for record in seen}) == 4
    assert {record[5] for record in seen} == {"step-seed"}
    assert (tmp_path / "src" / "fake_1.py").read_text() == canonical_source

    canonical_path = tmp_path / "src" / "fake_1.py"
    current_source = "# CURRENT SNAPSHOT\n" + canonical_path.read_text()
    canonical_path.write_text(current_source)
    seen.clear()
    config["rl"]["rollout_workers"] = 2
    barrier = threading.Barrier(2)
    next_group = run_advisor_rl.build_rollout_group(
        1,
        config,
        args,
        db,
        idea_repo_db,
        prompts,
        advisor,
        StaticImplementer({}),
        cc,
        ec,
        rc,
    )

    assert len(next_group.samples) == 4
    assert len({record[0] for record in seen}) == 2
    assert all(record[6] for record in seen)
    assert not any(record[7] for record in seen)
    assert canonical_path.read_text() == current_source


def test_barrier_merges_all_successful_idea_forks(tmp_path, monkeypatch):
    config = _config(tmp_path)
    order = []
    db, idea_repo_db, prompts, trainer, backend, cc, ec, rc, args = _harness(
        config, "none", monkeypatch, order
    )
    args.n_samples = 3
    args.max_steps = 2
    next_step_descriptions = []

    def fake_build_rollout_group(step, *unused_args, **unused_kwargs):
        if step == 1:
            next_step_descriptions.extend(
                idea.description
                for idea in idea_repo_db.idea_repos[0][-1].ideas
            )
            return RolloutGroup(step=step, samples=[])

        samples = []
        for index in range(3):
            fork = deepcopy(idea_repo_db.idea_repos[0][-1])
            idea = idea_select_utils.Idea(
                id=fork.get_next_id(), description=f"distinct idea {index}"
            )
            fork.ideas.append(idea)
            samples.append(
                RolloutSample(
                    island_id=0,
                    response_text=f"advisor response {index}",
                    program_text=f"implementer program {index}",
                    idea_id=idea.id,
                    exp_description=f"experiment {index}",
                    raw_score=2.0 + index,
                    reward=float(index),
                    eval_success=True,
                    updated_idea_repo=fork,
                )
            )
        return RolloutGroup(step=step, samples=samples)

    monkeypatch.setattr(
        run_advisor_rl, "build_rollout_group", fake_build_rollout_group
    )
    run_advisor_rl.run_evolution(
        config,
        args,
        db,
        idea_repo_db,
        prompts,
        trainer,
        "advisor",
        "implementer",
        cc,
        ec,
        rc,
    )

    assert next_step_descriptions == [
        "distinct idea 0",
        "distinct idea 1",
        "distinct idea 2",
    ]
    merged = idea_repo_db.idea_repos[0][-1]
    assert len(idea_repo_db.idea_repos[0]) == 2
    assert [idea.exp_count for idea in merged.ideas] == [1, 1, 1]
    assert [idea.exp_history for idea in merged.ideas] == [
        ["score=2.0000 — experiment 0"],
        ["score=3.0000 — experiment 1"],
        ["score=4.0000 — experiment 2"],
    ]
    registered_programs = {
        program for _, program in db._islands[0]._candidates
    }
    assert {
        "implementer program 0",
        "implementer program 1",
        "implementer program 2",
    } <= registered_programs


def test_main_rejects_invalid_objective(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_advisor_rl.py", "--task_id", "fake_task", "--objective", "bad"],
    )
    with pytest.raises(SystemExit):
        run_advisor_rl.main()
    assert "invalid choice" in capsys.readouterr().err


def test_resolve_config_paths_anchors_relative_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(run_advisor_rl, "project_root", str(tmp_path))
    relative_paths = {
        "data_path": "tasks/fake/data",
        "src_path": "tasks/fake/src",
        "eval_path": "tasks/fake/eval",
        "results_path": "tasks/fake/results",
        "build_dir": "tasks/fake/results/build",
        "log_dir": "tasks/fake/logs",
        "transcript_dir": "tasks/fake/transcripts",
        "frontier_solution": "external/solution.cpp",
    }
    config = {
        "paths": {**relative_paths, "target_file_path": "solution.cpp"}
    }
    compile_config = task_utils.CompilationConfig(
        target_file_path="tasks/fake/src/solution.cpp"
    )

    run_advisor_rl._resolve_config_paths(config, compile_config)

    for key, value in relative_paths.items():
        assert config["paths"][key] == os.path.join(tmp_path, value)
    assert config["paths"]["target_file_path"] == "solution.cpp"
    assert compile_config.target_file_path == os.path.join(
        tmp_path, "tasks/fake/src/solution.cpp"
    )
    for key in ("results_path", "build_dir", "log_dir", "transcript_dir"):
        assert os.path.isdir(config["paths"][key])


def test_main_rejects_unimplemented_torch_backend(
    tmp_path, monkeypatch, capsys
):
    config = _config(tmp_path)
    monkeypatch.setattr(
        run_advisor_rl.run_experiment,
        "load_configs",
        lambda path: (config, None, [], None),
    )
    monkeypatch.setattr(run_advisor_rl.rl_trainer, "TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_advisor_rl.py", "--task_id", "fake_task", "--backend", "torch"],
    )
    with pytest.raises(SystemExit):
        run_advisor_rl.main()
    assert "use --backend mock" in capsys.readouterr().err


def test_main_accepts_none_rl_section(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["rl"] = None
    config["paths"]["log_dir"] = str(tmp_path / "logs")
    config["paths"]["transcript_dir"] = str(tmp_path / "transcripts")
    loaded_paths = []

    def fake_load_configs(path):
        loaded_paths.append(path)
        return config, None, [], None

    monkeypatch.setattr(
        run_advisor_rl.run_experiment, "load_configs", fake_load_configs
    )
    called = []
    monkeypatch.setattr(
        run_advisor_rl,
        "run_evolution",
        lambda *args, **kwargs: called.append(True),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_advisor_rl.py", "--task_id", "fake_task", "--max_steps", "0"],
    )
    monkeypatch.chdir(tmp_path)

    run_advisor_rl.main()

    assert called == [True]
    assert os.path.normpath(loaded_paths[0]) == os.path.join(
        _REPO, "tasks", "fake_task", "config", "config_1.yaml"
    )
