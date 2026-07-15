# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone evaluator for expert-parallelism load-balancing candidates."""

import argparse
import importlib.util
import sys
import time

import numpy as np


def device_loads(expert_loads, assignment, num_devices) -> np.ndarray:
    loads = np.zeros(num_devices, dtype=float)
    for e, d in enumerate(assignment):
        loads[int(d)] += expert_loads[e]
    return loads


def validate_assignment(assignment, num_experts, num_devices) -> bool:
    if len(assignment) != num_experts:
        return False
    return all(0 <= int(d) < num_devices for d in assignment)


def balancedness(expert_loads, assignment, num_devices) -> float:
    # mean device load / max device load in (0, 1]; 1.0 == perfectly balanced.
    loads = device_loads(expert_loads, assignment, num_devices)
    peak = loads.max()
    return float(loads.mean() / peak) if peak > 0 else 1.0


def make_profiles(num_experts, num_devices, num_profiles, seed) -> list:
    # Deterministic synthetic per-expert loads via a seeded power law (Zipf-like).
    rng = np.random.default_rng(seed)
    return [rng.zipf(2.0, size=num_experts).astype(float)
            for _ in range(num_profiles)]


def evaluate(assign_fn, profiles, num_devices, ref_time) -> dict:
    total_time = 0.0
    profile_balancedness = []

    for expert_loads in profiles:
        start = time.perf_counter()
        assignment = assign_fn(expert_loads, num_devices)
        total_time += time.perf_counter() - start

        if not validate_assignment(
            assignment, len(expert_loads), num_devices
        ):
            return {
                "score": 0.0,
                "balancedness": 0.0,
                "speed": 0.0,
                "valid": False,
            }
        profile_balancedness.append(
            balancedness(expert_loads, assignment, num_devices)
        )

    candidate_time = max(total_time, 1e-9)
    bal = float(np.mean(profile_balancedness))
    speed = min(1.0, ref_time / candidate_time)
    score = 0.5 * bal + 0.5 * speed
    return {
        "score": score,
        "balancedness": bal,
        "speed": speed,
        "valid": True,
    }


def _load_candidate(candidate_path):
    spec = importlib.util.spec_from_file_location("eplb_candidate", candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate from {candidate_path}")
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--num_devices", type=int, default=8)
    parser.add_argument("--num_profiles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ref_time", type=float, default=1.0)
    parser.add_argument("--compile_only", action="store_true")
    args = parser.parse_args()

    try:
        candidate = _load_candidate(args.candidate_path)
        assert hasattr(candidate, "assign_experts")
    except Exception as exc:
        print(f"Failed to load candidate: {exc}", file=sys.stderr)
        return 1

    if args.compile_only:
        print("OK")
        return 0

    profiles = make_profiles(
        args.num_experts, args.num_devices, args.num_profiles, args.seed
    )
    result = evaluate(
        candidate.assign_experts, profiles, args.num_devices, args.ref_time
    )
    print("Candidate: " + repr(result))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
