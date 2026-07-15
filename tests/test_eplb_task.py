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

"""Spec for the EPLB task — the balancedness/validity metrics, the seed
assignment, and the eval-plugin contract (run end to end via subprocess)."""

import importlib
import os
import shutil
import sys

import numpy as np
import pytest
import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EPLB = os.path.join(_REPO, "tasks", "eplb")


def _eval_mod():
    return importlib.import_module("tasks.eplb.eval.eval_eplb")


def _seed_mod():
    return importlib.import_module("tasks.eplb.src.eplb_1")


# --- metric helpers -------------------------------------------------------

def test_device_loads():
    m = _eval_mod()
    np.testing.assert_allclose(m.device_loads([1.0, 2.0, 3.0], [0, 0, 1], 2), [3.0, 3.0])


def test_balancedness_hand_computed():
    m = _eval_mod()
    # device0 = 4+2 = 6, device1 = 3+1 = 4; mean 5, max 6 -> 5/6
    assert m.balancedness([4.0, 3.0, 2.0, 1.0], [0, 1, 0, 1], 2) == pytest.approx(5 / 6)


def test_balancedness_perfect_is_one():
    m = _eval_mod()
    assert m.balancedness([2.0, 2.0], [0, 1], 2) == pytest.approx(1.0)


def test_validate_assignment():
    m = _eval_mod()
    assert m.validate_assignment([0, 1, 0], 3, 2)
    assert not m.validate_assignment([0, 2, 0], 3, 2)   # device out of range
    assert not m.validate_assignment([0, 1], 3, 2)      # wrong length


# --- seed program ---------------------------------------------------------

def test_seed_assignment_is_valid_and_balanced():
    m, seed = _eval_mod(), _seed_mod()
    for loads in m.make_profiles(64, 8, 3, seed=0):
        assignment = seed.assign_experts(loads, 8)
        assert m.validate_assignment(assignment, len(loads), 8)
        assert 0.0 < m.balancedness(loads, assignment, 8) <= 1.0


def test_evaluate_scores_in_unit_range():
    m, seed = _eval_mod(), _seed_mod()
    result = m.evaluate(seed.assign_experts, m.make_profiles(64, 8, 3, seed=0), 8, ref_time=1.0)
    assert result["valid"]
    assert 0.0 <= result["score"] <= 1.0
    assert 0.0 < result["balancedness"] <= 1.0
    assert 0.0 < result["speed"] <= 1.0


def test_config_init_score_matches_seed_measurement():
    m, seed = _eval_mod(), _seed_mod()
    profiles = m.make_profiles(128, 8, 5, seed=0)
    balancedness = np.mean(
        [
            m.balancedness(loads, seed.assign_experts(loads, 8), 8)
            for loads in profiles
        ]
    )
    measured_score = 0.5 * balancedness + 0.5
    with open(os.path.join(_EPLB, "config", "config_1.yaml")) as f:
        config = yaml.safe_load(f)
    assert config["evaluation"]["init_score"] == pytest.approx(
        measured_score, abs=5e-5
    )


def test_evaluate_rejects_invalid_assignment():
    m = _eval_mod()
    result = m.evaluate(lambda loads, nd: [nd] * len(loads),  # every device out of range
                        m.make_profiles(16, 4, 2, seed=0), 4, ref_time=1.0)
    assert not result["valid"]
    assert result["score"] == 0.0


# --- eval-plugin contract -------------------------------------------------

def test_parse_eval_results_roundtrip():
    ev = importlib.import_module("tasks.eplb.eval.eval_utils")
    out = "Candidate: {'score': 0.75, 'balancedness': 0.8, 'speed': 0.7}"
    assert ev.parse_eval_results(out) == pytest.approx(0.75)
    assert ev.parse_eval_results([out]) == pytest.approx(0.75)


def _eplb_config(tmp_path):
    with open(os.path.join(_EPLB, "config", "config_1.yaml")) as f:
        config = yaml.safe_load(f)
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(os.path.join(_EPLB, "src", "eplb_1.py"), src / "eplb_1.py")
    config["paths"]["src_path"] = str(src)
    config["paths"]["target_file_path"] = "eplb_1.py"
    config["paths"]["eval_path"] = os.path.join(_EPLB, "eval")
    config["compilation"]["python_bin"] = sys.executable  # venv python (has numpy)
    return config


def test_eval_utils_end_to_end(tmp_path):
    ev = importlib.import_module("tasks.eplb.eval.eval_utils")
    config = _eplb_config(tmp_path)
    compiled = ev.recompile_library(config)
    assert compiled.returncode == 0
    proc = ev.evaluate_dataset(1, -1, ev.EvalConfig(dataset="synthetic"), config)
    assert proc.returncode == 0
    score = ev.parse_eval_results(proc.stdout)
    assert score is not None and 0.0 <= score <= 1.0
