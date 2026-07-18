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

import numpy as np


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

_PATH_SKIP = {"target_file_path"}
_RUNTIME_DIRS = ("results_path", "build_dir", "log_dir", "transcript_dir")


def _resolve_config_paths(config, compile_config):
    """Resolve filesystem paths independently of the caller's working directory."""
    paths = config["paths"]
    for key, value in paths.items():
        if key in _PATH_SKIP or not isinstance(value, str) or not value:
            continue
        value = os.path.expanduser(value)
        if not os.path.isabs(value):
            value = os.path.join(project_root, value)
        paths[key] = value

    if compile_config is not None:
        target_file_path = paths["target_file_path"]
        if not os.path.isabs(target_file_path):
            target_file_path = os.path.join(paths["src_path"], target_file_path)
        compile_config.target_file_path = target_file_path

    for key in _RUNTIME_DIRS:
        if paths.get(key):
            os.makedirs(paths[key], exist_ok=True)


def _clean_text(text):
    """Match generate_completion's post-processing so a captured generation's
    text can be compared against the returned response."""
    if text is None:
        return None
    return text.replace("<end_of_turn>", "").replace("<start_of_turn>", "").strip()


def _capture_advisor_tokens(advisor, expected_text):
    """Return (token_ids, old_logprobs, prompt_token_ids) for the advisor's last
    generation, or (None, None, None).

    With the Tinker backend the sampled tokens arrive via backend.last_generation
    (a side channel, since the LLMClient contract returns only text). A
    name-string advisor has no `.backend`, so this is a no-op and the
    mock/baseline paths are unchanged. When `expected_text` is the returned
    response, only accept the capture if it matches — this rejects a stale or
    leaked (timed-out, still-running) generation that would otherwise mis-pair
    tokens with the wrong sample's reward.
    """
    backend = getattr(advisor, "backend", None)
    gen = getattr(backend, "last_generation", None) if backend is not None else None
    if gen is None or gen.token_ids is None:
        return None, None, None
    if expected_text is not None and _clean_text(gen.text) != _clean_text(expected_text):
        return None, None, None
    return gen.token_ids, gen.logprobs, gen.prompt_token_ids


def _reset_advisor_capture(advisor):
    """Clear the capture slot so a prior sample's tokens can't be read for this
    one if this sample's advisor call produces nothing usable."""
    backend = getattr(advisor, "backend", None)
    if backend is not None:
        backend.last_generation = None


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
    _reset_advisor_capture(advisor)
    parent, island_id = db.get_candidate()
    if idea_repo_db.idea_repos[island_id]:
        idea_repo = deepcopy(idea_repo_db.idea_repos[island_id][-1])
    else:
        idea_repo = idea_select_utils.IdeaRepo()
    idea_repo.sota = parent

    transcript = llm_utils.Transcript(log_filename=args._transcript_file)
    advisor_flow = (config.get("rl") or {}).get("advisor_flow", "select")

    if advisor_flow == "propose":
        # Direct propose flow: a single advisor generation (build_advisor_prompt)
        # IS the trained action. Matches tasks whose prompts propose an idea
        # rather than select one by id, and gives RL a clean single-turn action.
        prompt = prompts.construct_idea_gen_prompt(parent, idea_repo)
        transcript.append(
            llm_utils.ContentChunk(prompt, "user", tags=["idea_propose_prompt"])
        )
        raw = llm_utils.generate_completion(
            advisor, transcript, advisor_utils.make_role_config(config, "advisor")
        )
        if not raw:
            return rl_trainer.RolloutSample(
                island_id=island_id,
                prompt_text=prompt or "",
                response_text="",
                idea_id=-1,
                exp_description="",
                raw_score=None,
                eval_success=False,
                response_mask=None,
            )
        transcript.append(
            llm_utils.ContentChunk(raw, "model", tags=["idea_propose_response"])
        )
        # The proposal text IS the experiment description handed to the
        # implementer; idea_id is unused by the propose-style prompts.
        idea_id, exp_desc = 0, raw
        token_ids, old_logprobs, prompt_token_ids = _capture_advisor_tokens(advisor, raw)
    else:
        if args.use_idea_repo:
            idea_select_utils.scratch_pad(
                idea_repo,
                advisor,
                transcript,
                advisor_utils.make_role_config(config, "advisor"),
                prompts.construct_idea_gen_prompt(parent, idea_repo),
            )

        idea_id, exp_desc, raw, prompt = advisor_utils.select_idea_no_code(
            advisor,
            transcript,
            prompts,
            parent,
            idea_repo,
            advisor_utils.make_role_config(config, "advisor"),
        )
        if (idea_id, exp_desc, raw) == (None, None, None):
            # Parse failure: leave tokens uncaptured. run_evolution computes
            # advantages over only the captured (trainable) samples, so this
            # sample does not skew the group baseline; it still counts for logging.
            return rl_trainer.RolloutSample(
                island_id=island_id,
                prompt_text=prompt or "",
                response_text="",
                idea_id=idea_id or -1,
                exp_description=exp_desc or "",
                raw_score=None,
                eval_success=False,
                response_mask=None,
            )

        # Snapshot the winning generation NOW, before the long implement_idea
        # phase, correlated against `raw` so we never train tokens that didn't
        # produce the scored idea.
        token_ids, old_logprobs, prompt_token_ids = _capture_advisor_tokens(advisor, raw)

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
        prompt_text=prompt or "",
        response_text=raw or "",
        token_ids=token_ids,
        old_logprobs=old_logprobs,
        prompt_token_ids=prompt_token_ids,
        response_mask=(
            np.ones(len(token_ids), dtype=float)
            if token_ids is not None
            else None
        ),
        idea_id=idea_id or -1,
        exp_description=exp_desc or "",
        raw_score=raw_score,
        eval_success=evaluation_completed and raw_score is not None,
        updated_idea_repo=idea_repo,
        program_text=trial.algorithm_implementation,
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
        rl_cfg = config.get("rl") or {}
        if bool(rl_cfg.get("eval_case_resampling", False)):
            base_seed = int((config.get("run") or {}).get("seed", 0))
            os.environ["PACE_EVAL_CASE_SEED"] = str(base_seed * 1_000_003 + step)
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
                    program=sample.program_text,
                    island_id=sample.island_id,
                    score=sample.raw_score,
                )
                idea_repo_db.best_scores_history[sample.island_id].append(
                    sample.raw_score
                )
                idea_repo_db.scheduler.update_score(
                    sample.island_id, sample.raw_score
                )

        # Merge successful rollout forks into one pool per island and record a
        # lightweight per-idea experiment log without LLM summarization.
        for island_id in range(idea_repo_db.num_islands):
            successful_samples = [
                sample
                for sample in group.samples
                if sample.island_id == island_id
                and sample.eval_success
                and sample.raw_score is not None
                and sample.updated_idea_repo is not None
            ]
            if not successful_samples:
                continue

            base = deepcopy(idea_repo_db.idea_repos[island_id][-1])
            descriptions = {idea.description for idea in base.ideas}
            for sample in successful_samples:
                for idea in sample.updated_idea_repo.ideas:
                    if idea.description in descriptions:
                        continue
                    new_idea = deepcopy(idea)
                    new_idea.id = base.get_next_id()
                    base.ideas.append(new_idea)
                    descriptions.add(new_idea.description)

            for sample in successful_samples:
                selected_idea = sample.updated_idea_repo.find_idea_by_id(
                    sample.idea_id
                )
                if selected_idea is None:
                    continue
                matching_idea = next(
                    (
                        idea
                        for idea in base.ideas
                        if idea.description == selected_idea.description
                    ),
                    None,
                )
                if matching_idea is None:
                    continue
                exp_description = " ".join(
                    sample.exp_description.split()
                )[:200]
                matching_idea.exp_history.append(
                    f"score={sample.raw_score:.4f} — {exp_description}"
                )
                matching_idea.exp_count += 1

            idea_repo_db.idea_repos[island_id].append(base)

        # Advantages are computed inside train_step over the group's rewards;
        # restrict to samples we can actually train (captured tokens) so
        # parse-failures / uncorrelated captures don't skew the baseline. The
        # full group still drove the population updates above and the reward
        # logging below. Mock/baseline backends never set token_ids, so this
        # falls back to the full group and their behavior is unchanged.
        trainable = [s for s in group.samples if s.token_ids is not None]
        result = trainer.train_step(
            rl_trainer.RolloutGroup(step=step, samples=trainable)
            if trainable
            else group
        )
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
    parser.add_argument(
        "--objective",
        choices=("pacevolve++", "grpo", "entropic", "maxk", "none"),
        default=None,
    )
    parser.add_argument(
        "--backend", choices=("mock", "torch", "tinker"), default=None
    )
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--use_idea_repo",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    config_path = os.path.join(
        project_root,
        "tasks",
        args.task_id,
        "config",
        args.dataset_id,
        f"config_{args.run_id}.yaml",
    )
    config, compile_config, eval_configs, _ = run_experiment.load_configs(
        config_path
    )
    _resolve_config_paths(config, compile_config)

    trainer_config = rl_trainer.AdvisorTrainerConfig.from_config(config)
    rl_config = config.get("rl") or {}
    if args.objective is None:
        args.objective = trainer_config.objective
    if args.backend is None:
        args.backend = rl_config.get("backend", "mock")
    if args.n_samples is None:
        args.n_samples = trainer_config.n_samples
    if args.max_steps is None:
        args.max_steps = trainer_config.total_steps
    if args.backend == "torch" and rl_trainer.TORCH_AVAILABLE:
        parser.error(
            "TorchPolicyBackend is a documented unimplemented seam; "
            "use --backend mock"
        )

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
    if args.backend == "tinker":
        import tinker_backend

        backend = tinker_backend.TinkerPolicyBackend(config)
        # The advisor becomes the TRAINED Tinker model; generate_completion
        # accepts a client, so pass the backend-backed client directly.
        advisor = rl_trainer.BackendLLMClient(
            backend, advisor_utils.make_role_config(config, "advisor")
        )
    else:
        if args.backend == "torch":
            logger.warning(
                "Torch is unavailable; falling back to the mock backend."
            )
        backend = rl_trainer.MockPolicyBackend(config)
        advisor = advisor_utils.make_role_config(config, "advisor")["llm"][
            "name"
        ]
    trainer = rl_trainer.AdvisorTrainer(trainer_config, backend)

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
