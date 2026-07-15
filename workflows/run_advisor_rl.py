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

"""PACEvolve++ advisor reinforcement-learning barrier loop."""

import argparse
from copy import deepcopy
from importlib import import_module
import logging
import os
import sys
import time


current_script_path = os.path.abspath(__file__)
workflows_dir = os.path.dirname(current_script_path)
project_root = os.path.dirname(workflows_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import advisor_utils
import idea_select_utils
import llm_utils
import program_database
import rl_rewards
import rl_trainer
import run_experiment


logger = logging.getLogger("controller")


def build_rollout_sample(
    step,
    sample_index,
    config,
    args,
    db,
    idea_repo_db,
    prompts,
    advisor,
    implementer,
    compile_config,
    eval_configs,
    reward_cfg,
) -> rl_trainer.RolloutSample:
    """Build and evaluate one advisor rollout sample."""
    parent, island_id = db.get_candidate()
    if idea_repo_db.idea_repos[island_id]:
        idea_repo = deepcopy(idea_repo_db.idea_repos[island_id][-1])
    else:
        idea_repo = idea_select_utils.IdeaRepo()
    idea_repo.sota = parent

    transcript = llm_utils.Transcript(log_filename=args._transcript_file)
    if args.use_idea_repo:
        idea_select_utils.scratch_pad(
            idea_repo,
            advisor,
            transcript,
            advisor_utils.make_role_config(config, "advisor"),
            prompts.construct_idea_gen_prompt(parent, idea_repo),
        )

    idea_id, exp_desc, raw = advisor_utils.select_idea_no_code(
        advisor,
        transcript,
        prompts,
        parent,
        idea_repo,
        advisor_utils.make_role_config(config, "advisor"),
    )
    if (idea_id, exp_desc, raw) == (None, None, None):
        return rl_trainer.RolloutSample(
            island_id=island_id,
            prompt_text=raw or "",
            response_text="",
            idea_id=idea_id or -1,
            exp_description=exp_desc or "",
            raw_score=None,
            eval_success=False,
            response_mask=None,
        )

    trial = advisor_utils.implement_idea(
        implementer,
        transcript,
        prompts,
        parent,
        idea_id,
        exp_desc,
        compile_config,
        eval_configs,
        advisor_utils.make_role_config(config, "implementer"),
        candidate_id=step * args.n_samples + sample_index,
        baseline_id=config["experiment"]["initial_baseline_id"],
    )

    raw_score = None
    evaluation_completed = trial.compile_success and all(trial.eval_success)
    if evaluation_completed:
        task_id = config["experiment"]["task_id"]
        eval_utils = import_module(f"tasks.{task_id}.eval.eval_utils")
        raw_score = eval_utils.parse_eval_results(trial.eval_results)

    return rl_trainer.RolloutSample(
        island_id=island_id,
        prompt_text=raw or "",
        response_text=trial.algorithm_implementation,
        idea_id=idea_id or -1,
        exp_description=exp_desc or "",
        raw_score=raw_score,
        eval_success=evaluation_completed and raw_score is not None,
        response_mask=None,
        updated_idea_repo=idea_repo,
    )


def build_rollout_group(
    step,
    config,
    args,
    db,
    idea_repo_db,
    prompts,
    advisor,
    implementer,
    compile_config,
    eval_configs,
    reward_cfg,
) -> rl_trainer.RolloutGroup:
    """Build a sequential rollout group and shape every sample's reward."""
    # Parallel rollout construction is a future extension. Candidates share a
    # target file, so naive thread-based evaluation is not safe.
    samples = [
        build_rollout_sample(
            step,
            sample_index,
            config,
            args,
            db,
            idea_repo_db,
            prompts,
            advisor,
            implementer,
            compile_config,
            eval_configs,
            reward_cfg,
        )
        for sample_index in range(args.n_samples)
    ]
    for sample in samples:
        sample.reward = rl_rewards.shape_reward(sample.raw_score, reward_cfg)
    return rl_trainer.RolloutGroup(step=step, samples=samples)


def run_evolution(
    config,
    args,
    db,
    idea_repo_db,
    prompts,
    trainer,
    advisor,
    implementer,
    compile_config,
    eval_configs,
    reward_cfg,
) -> None:
    """Run rollout barriers with population updates before policy updates."""
    for step in range(args.max_steps):
        group = build_rollout_group(
            step,
            config,
            args,
            db,
            idea_repo_db,
            prompts,
            advisor,
            implementer,
            compile_config,
            eval_configs,
            reward_cfg,
        )

        for sample in group.samples:
            if sample.eval_success and sample.raw_score is not None:
                db.register_program(
                    program=sample.response_text,
                    island_id=sample.island_id,
                    score=sample.raw_score,
                )
                idea_repo_db.best_scores_history[sample.island_id].append(
                    sample.raw_score
                )
                idea_repo_db.scheduler.update_score(
                    sample.island_id, sample.raw_score
                )
                # Persist the accumulated idea repo so the pool + experiment
                # history grow across steps (matching run_experiment.py).
                if sample.updated_idea_repo is not None:
                    idea_repo_db.idea_repos[sample.island_id].append(
                        sample.updated_idea_repo
                    )

        result = trainer.train_step(group)
        rewards = group.rewards()
        mean_reward = float(rewards.mean()) if len(rewards) else float("nan")
        max_reward = float(rewards.max()) if len(rewards) else float("nan")
        logger.info(
            "Step %d: mean_reward=%.6f max_reward=%.6f skipped=%s "
            "alpha=%.6f metrics=%s",
            step,
            mean_reward,
            max_reward,
            result.skipped,
            result.alpha,
            result.metrics,
        )


def main() -> None:
    """Run PACEvolve++ from a task configuration."""
    parser = argparse.ArgumentParser(
        description="Run PACEvolve++ advisor reinforcement learning"
    )
    parser.add_argument("--task_id", "-t", required=True)
    parser.add_argument("--dataset_id", "-d", default=".")
    parser.add_argument("--run_id", "-r", type=int, default=1)
    parser.add_argument("--objective", default=None)
    parser.add_argument(
        "--backend", choices=("mock", "torch"), default=None
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--use_idea_repo",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    CONFIG_PATH = os.path.abspath(
        f"../tasks/{args.task_id}/config/{args.dataset_id}/"
        f"config_{args.run_id}.yaml"
    )
    config, compile_config, eval_configs, _ = run_experiment.load_configs(
        CONFIG_PATH
    )

    trainer_config = rl_trainer.AdvisorTrainerConfig.from_config(config)
    rl_config = config.get("rl", {})
    if args.objective is None:
        args.objective = trainer_config.objective
    if args.backend is None:
        args.backend = rl_config.get("backend", "mock")
    if args.n_samples is None:
        args.n_samples = trainer_config.n_samples
    if args.max_steps is None:
        args.max_steps = trainer_config.total_steps

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    logfile_dir = os.path.expanduser(config["paths"]["log_dir"])
    os.makedirs(logfile_dir, exist_ok=True)
    logfile_path = os.path.join(
        logfile_dir, f"controller_verbose_{timestamp}.log"
    )
    log_file_handler = logging.FileHandler(logfile_path)
    log_file_handler.setLevel(logging.DEBUG)
    log_file_handler.setFormatter(formatter)
    logger.addHandler(log_file_handler)

    transcript_dir = os.path.expanduser(config["paths"]["transcript_dir"])
    os.makedirs(transcript_dir, exist_ok=True)
    args._transcript_file = os.path.join(
        transcript_dir, f"transcript_{timestamp}.txt"
    )
    print("Transcript will be written to: ", args._transcript_file)

    num_islands = config["database"]["num_islands"]
    metric_direction = config["evaluation"]["metric_direction"]
    idea_repo_db = idea_select_utils.IdeaRepoDatabase(
        num_islands=num_islands,
        target_score=config["evaluation"]["target_score"],
        metric_direction=metric_direction,
    )

    task_id = config["experiment"]["task_id"]
    prompt_filename = config["experiment"].get("prompts_file", "prompts")
    if args.dataset_id == ".":
        prompts = import_module(f"tasks.{task_id}.config.{prompt_filename}")
    else:
        prompts = import_module(
            f"tasks.{task_id}.config.{args.dataset_id}.{prompt_filename}"
        )
    sota_algo = getattr(
        prompts, config["experiment"]["sota_algo_name"]
    )

    programs_db_config = program_database.ProgramsDatabaseConfig(
        num_islands=num_islands,
        tournament_size=config["database"]["tournament_size"],
        top_k=config["database"]["top_k"],
        max_queue_size=config["database"]["max_queue_size"],
    )
    db = program_database.ProgramsDatabase(
        config=programs_db_config,
        template=sota_algo,
        function_to_evolve=task_id,
        metric_direction=metric_direction,
    )

    logger.info("Registering initial scores and solutions.")
    init_score = config["evaluation"]["init_score"]
    for island_id in range(num_islands):
        db.register_program(
            program=sota_algo,
            island_id=island_id,
            score=init_score,
        )
        idea_repo_db.best_scores_history[island_id].append(init_score)
        idea_repo_db.scheduler.update_score(island_id, init_score)
        initial_repo = idea_select_utils.IdeaRepo()
        initial_repo.sota = sota_algo
        idea_repo_db.idea_repos[island_id].append(initial_repo)

    trainer_config.objective = args.objective
    trainer_config.total_steps = args.max_steps
    trainer_config.n_samples = args.n_samples
    if args.backend == "torch" and rl_trainer.TORCH_AVAILABLE:
        backend = rl_trainer.TorchPolicyBackend(config)
    else:
        if args.backend == "torch":
            logger.warning(
                "Torch is unavailable; falling back to the mock backend."
            )
        backend = rl_trainer.MockPolicyBackend(config)
    trainer = rl_trainer.AdvisorTrainer(trainer_config, backend)

    advisor = advisor_utils.make_role_config(config, "advisor")["llm"][
        "name"
    ]
    implementer = advisor_utils.make_role_config(config, "implementer")[
        "llm"
    ]["name"]
    reward_cfg = rl_rewards.RewardShapingConfig.from_config(config)

    run_evolution(
        config,
        args,
        db,
        idea_repo_db,
        prompts,
        trainer,
        advisor,
        implementer,
        compile_config,
        eval_configs,
        reward_cfg,
    )


if __name__ == "__main__":
    main()
