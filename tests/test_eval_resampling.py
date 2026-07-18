"""Tests for deterministic per-step rectangle-grid evaluation resampling."""

import importlib
import os

import run_advisor_rl
from rl_trainer import RolloutSample
from test_run_advisor_rl_smoke import _config, _harness


eval_rfg = importlib.import_module("tasks.rectangle_free_grid.eval.eval_rfg")


def test_sample_cases_deterministic():
    assert eval_rfg.sample_cases(42) == eval_rfg.sample_cases(42)
    assert eval_rfg.sample_cases(42) != eval_rfg.sample_cases(43)


def test_sample_cases_bounds():
    anchor_shapes = {tuple(sorted(case)) for case in eval_rfg.ANCHOR_CASES}
    for seed in (0, 1, 2, 42, 1000, 556635):
        cases = eval_rfg.sample_cases(seed)
        assert len(cases) == 16
        assert len(set(cases)) == len(cases)
        assert all(n >= 1 and m >= 1 and n * m <= 100000 for n, m in cases)
        assert all(tuple(sorted(case)) not in anchor_shapes for case in cases)


def test_sample_cases_coverage():
    for seed in range(10):
        cases = eval_rfg.sample_cases(seed)
        assert any(n * m >= 50000 for n, m in cases)
        assert any(n * m <= 400 for n, m in cases)
        assert any(min(n, m) <= 10 for n, m in cases)
        assert any(n == m for n, m in cases)


def test_resolve_case_seed_precedence():
    assert eval_rfg.resolve_case_seed(7, {"PACE_EVAL_CASE_SEED": "11"}) == 7
    assert eval_rfg.resolve_case_seed(None, {"PACE_EVAL_CASE_SEED": "11"}) == 11
    assert eval_rfg.resolve_case_seed(None, {}) is None
    assert eval_rfg.resolve_case_seed(None, {"PACE_EVAL_CASE_SEED": ""}) is None


def test_step_seed_set_per_step_and_gated(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["run"] = {"seed": 7}
    config["rl"]["eval_case_resampling"] = True
    order = []
    db, idea_repo_db, prompts, trainer, _, cc, ec, rc, args = _harness(
        config, "none", monkeypatch, order
    )
    seen = []

    def record_sample(step, sample_index, *unused_args, **unused_kwargs):
        seen.append((step, sample_index, os.environ.get("PACE_EVAL_CASE_SEED")))
        return RolloutSample(island_id=0)

    monkeypatch.setattr(run_advisor_rl, "build_rollout_sample", record_sample)
    monkeypatch.delenv("PACE_EVAL_CASE_SEED", raising=False)
    run_advisor_rl.run_evolution(
        config, args, db, idea_repo_db, prompts, trainer,
        "advisor", "implementer", cc, ec, rc,
    )

    assert len(seen) == args.max_steps * args.n_samples
    assert {seed for step, _, seed in seen if step == 0} == {"7000021"}
    assert {seed for step, _, seed in seen if step == 1} == {"7000022"}

    config["rl"]["eval_case_resampling"] = False
    monkeypatch.delenv("PACE_EVAL_CASE_SEED")
    seen.clear()
    run_advisor_rl.run_evolution(
        config, args, db, idea_repo_db, prompts, trainer,
        "advisor", "implementer", cc, ec, rc,
    )
    assert all(seed is None for _, _, seed in seen)

    monkeypatch.setenv("PACE_EVAL_CASE_SEED", "preexisting")
    seen.clear()
    run_advisor_rl.run_evolution(
        config, args, db, idea_repo_db, prompts, trainer,
        "advisor", "implementer", cc, ec, rc,
    )
    assert all(seed == "preexisting" for _, _, seed in seen)
