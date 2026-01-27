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

import argparse
import importlib.util
import itertools
import pickle
import numpy as np
import os
import sys
import traceback
import yaml

def run_simulation(AlgorithmClass, args):
    """Runs simulation for a given algorithm class and returns aggregated results."""
    all_run_results = []
    for _ in range(args.repetitions):
        algo = AlgorithmClass(
            servers=args.servers,
            duplicates=args.duplicates,
            objects=args.objects,
            epsilon=args.epsilon
        )

        init_time = algo.start()

        rep_results = {
            "serverLoadVariance": algo.variance(),
            "pctOfFullBins": algo.pctOfFullBins(),
            "timeAddObjects": init_time
        }
        all_run_results.append(rep_results)

    aggregated_results = {}
    if all_run_results:
        keys = all_run_results[0].keys()
        for key in keys:
            values = [r[key] for r in all_run_results]
            aggregated_results[f"mean_{key}"] = np.mean(values)
            aggregated_results[f"std_{key}"] = np.std(values)
    return aggregated_results

def load_class_from_file(path, class_name="RandomJumpConsistentHashing"):
    """Loads a specific class from a Python source file by its name."""
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {path}")
    module = importlib.util.module_from_spec(spec)
    # Use a unique name for the module in sys.modules to avoid conflicts
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, class_name):
        return getattr(module, class_name)
    else:
        raise AttributeError(f"Class '{class_name}' not found in {path}")

def main():
    """Parses arguments, runs candidate and baseline, and prints the diff."""
    parser = argparse.ArgumentParser(description="Evaluate a consistent hashing algorithm.")
    # These will be passed from the EvalConfig for consistent_hashing
    parser.add_argument("--epsilons", type=list[float], required=False, default=[1])
    parser.add_argument("--servers", type=int, required=False, default=1000)
    parser.add_argument("--objects_list", type=list[int], required=False, default=[3000])
    parser.add_argument("--duplicates_list", type=list[int], required=False, default=[4])
    parser.add_argument("--repetitions", type=int, required=False, default=3)
    parser.add_argument("--config_path", type=str, required=False, default=None)
    parser.add_argument("--dataset", type=str, required=False, default=None)
    parser.add_argument("--output", type=str, required=False, default="trial_results.pkl")

    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for cfg in config['evaluation']['eval_configs']:
            if cfg.get('dataset') == args.dataset:
                eval_config = cfg
                break
        epsilons = eval_config['epsilons']
        objects_list = eval_config['objects_list']
        duplicates_list = eval_config['duplicates_list']
    else:
        epsilons = args.epsilons
        objects_list = args.objects_list
        duplicates_list = args.duplicates_list
    
    # Define paths to candidate and reference implementations
    candidate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'rjch.py'))
    # candidate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rjch_diversified_probe.py'))
    # baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'naive_ch.py'))
    # baseline_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'rjch_ref.py'))

    # Load classes
    CandidateClass = load_class_from_file(candidate_path)
    # BaselineClass = load_class_from_file(baseline_path)

    results = {}
    target_results = {}
    with open("/usr/local/google/home/minghaoyan/Desktop/auto_evo/tasks/consistent_hashing/eval/target_perf.pkl", "rb") as file:
        target_results = pickle.load(file)
    # print(target_results)
    for eps, obj, dup in itertools.product(epsilons, objects_list, duplicates_list):
        args.epsilon = eps
        args.objects = obj
        args.duplicates = dup
        # target_results[(eps, obj, dup)] = {}
        try:

            # Run simulations
            candidate_results = run_simulation(CandidateClass, args)
            # candidate_results = run_simulation(BaselineClass, args)
            # baseline_results = run_simulation(BaselineClass, args)
            print(f"Hyperparam: epsilon={eps}, Num objects={obj}, Num duplicates={dup}")
            results[(eps, obj, dup)] = []
            for metric in ["serverLoadVariance", "pctOfFullBins", "timeAddObjects"]:
                
                candidate_value = candidate_results[f"mean_{metric}"]
                # baseline_value = baseline_results[f"mean_{metric}"]
                baseline_value = target_results[(eps, obj, dup)][metric]
                # target_results[(eps, obj, dup)][metric] = baseline_value
                if metric == "serverLoadVariance":
                    diff_pct = (candidate_value - baseline_value) / (baseline_value - 0.0001) * 100
                elif metric == "pctOfFullBins":
                    diff_pct = candidate_value*100 - baseline_value*100
                elif metric == "timeAddObjects":
                    diff_pct = (candidate_value - baseline_value) / (baseline_value + 1e-6) * 100
                    
                # results_string = f"Metric: {metric}, Candidate: {candidate_value:.4f}, Baseline: {baseline_value:.4f}, Diff: {diff_pct:+.2f}%"
                results_string = f"Metric: {metric}, Candidate: {candidate_value:.4f}, Target Performance: {baseline_value:.4f}, Diff: {diff_pct:+.2f}%"
                results[(eps, obj, dup)].append(results_string)
                print(results_string)

        except Exception as e:
            print(f"Evaluation script failed with error: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    
    # with open("target_perf.pkl", "wb") as file:
    #     pickle.dump(target_results, file)

    with open(args.output, "wb") as file:
        pickle.dump(results, file)

if __name__ == "__main__":
    main()