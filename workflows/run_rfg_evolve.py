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

"""PACEvolve++ per-island ensemble driver for the rectangle-free-grid task.

A shared *advisor* (reasoning model) proposes one strategic idea per candidate;
each *island* has its own frontier open-weight *implementer* that rewrites the
evolvable region; candidates compile+eval locally and register into a thread-safe
tournament-selection population. The global champion migrates into every island so
breakthroughs cross-pollinate. The champion is continuously written back to the
FrontierCS solution.cpp for submission, and metrics stream to Weights & Biases
(per-model cumulative max reward == the paper's Figure-2 plot).

Parallelism: a pool of W worker slots runs candidates concurrently (LLM calls are
network-bound). Each running candidate checks out an isolated working copy from a
pool, so slow reasoning models never block fast ones and workers never clobber each
other's edits. Islands are the *model/population* axis (candidate idx -> island
idx % num_islands), decoupled from the worker slots.

Faithful to PACEvolve++'s *search structure* (advisor->implementer decomposition,
islands, idea history, progress-normalized reward shaping); the RL trainer stays in
its shipped mock/no-op mode since black-box OpenRouter models expose no gradients.
"""

import argparse
import copy
import json
import logging
import os
import queue
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import llm_utils
import program_database
import rl_rewards
import task_utils
import workflow_utils

logger = logging.getLogger("controller")

_PATH_SKIP = {"target_file_path"}  # a bare filename, never a directory to resolve


def load_env_file(root):
    """Load KEY=VALUE lines from <root>/.env into os.environ (no external dep)."""
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    for line in open(env_path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def resolve_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        if k not in _PATH_SKIP and isinstance(v, str) and v and not os.path.isabs(v):
            cfg["paths"][k] = os.path.join(_ROOT, v)
    cfg["compilation"]["python_bin"] = sys.executable
    for key in ("results_path", "build_dir", "log_dir", "transcript_dir"):
        os.makedirs(cfg["paths"][key], exist_ok=True)
    return cfg


def parse_idea(text):
    """Pull the advisor's 'Idea:'/'Why:' lines; fall back to the whole message."""
    if not text:
        return "Improve the construction to raise the mean score."
    idea, why = "", ""
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("idea:"):
            idea = s[5:].strip()
        elif s.lower().startswith("why:"):
            why = s[4:].strip()
    if idea:
        return (idea + (f" (why: {why})" if why else "")).strip()
    return " ".join(text.split())[:400]


def _extract_regime(text):
    import re
    m = re.search(r"Detail:\s*(\{.*\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("by_regime")
    except Exception:
        return None


class Ensemble:
    """Shared state + worker pool for the evolutionary run."""

    def __init__(self, cfg, prompts, eval_utils, run_id):
        self.cfg = cfg
        self.prompts = prompts
        self.eval_utils = eval_utils
        self.run_id = run_id
        self.paths = cfg["paths"]
        self.islands_cfg = cfg["models"]["islands"]
        self.num_islands = len(self.islands_cfg)
        self.advisor_name = cfg["models"]["advisor"]["name"]
        self.advisor_cfg = copy.deepcopy(cfg)
        self.advisor_cfg["llm"] = {**cfg["llm"], **cfg["models"]["advisor"]}
        self.reward_cfg = rl_rewards.RewardShapingConfig.from_config(cfg)
        self.eval_configs = [eval_utils.EvalConfig(dataset=d["dataset"])
                             for d in cfg["evaluation"]["eval_configs"]]
        self.max_workers = int(cfg.get("run", {}).get("max_workers", 20))

        # population (thread-safe tournament islands)
        db_cfg = program_database.ProgramsDatabaseConfig(
            num_islands=self.num_islands,
            tournament_size=cfg["database"]["tournament_size"],
            top_k=cfg["database"]["top_k"],
            max_queue_size=cfg["database"]["max_queue_size"],
        )
        self.seed_region = getattr(prompts, cfg["experiment"]["sota_algo_name"])
        self.db = program_database.ProgramsDatabase(
            config=db_cfg, template=self.seed_region,
            function_to_evolve=cfg["experiment"]["task_id"],
            metric_direction=cfg["evaluation"]["metric_direction"],
        )
        init_score = float(cfg["evaluation"]["init_score"])
        for i in range(self.num_islands):
            self.db.register_program(self.seed_region, i, init_score)

        # island = model + population + shared idea history (guarded by self.lock)
        self.islands = [{"model": self.islands_cfg[i]["name"],
                         "overlay": self.islands_cfg[i], "history": []}
                        for i in range(self.num_islands)]

        # full-file template (fixed library + seed region); used to render a
        # submittable solution.cpp and to reset each worker copy.
        self.template_path = os.path.join(
            self.paths["src_path"], self.paths["target_file_path"])
        self.template_solution = open(self.template_path).read()

        # pool of isolated worker working-copies (one checkout per running candidate)
        self.pool = queue.Queue()
        for w in range(self.max_workers):
            wsrc = os.path.join(self.paths["build_dir"], f"worker_{w}", "src")
            wbuild = os.path.join(self.paths["build_dir"], f"worker_{w}", "build")
            os.makedirs(wsrc, exist_ok=True)
            os.makedirs(wbuild, exist_ok=True)
            wfile = os.path.join(wsrc, self.paths["target_file_path"])
            shutil.copy(self.template_path, wfile)
            self.pool.put({
                "src": wsrc, "build": wbuild, "file": wfile,
                "compile_cfg": task_utils.CompilationConfig(target_file_path=wfile),
            })

        # champion + logging state
        self.lock = threading.Lock()
        self.best_score = init_score
        self.best_region = self.seed_region
        self.best_model = "seed"
        self.island_best = [init_score] * self.num_islands
        self.done = 0
        self.stop = threading.Event()
        self.jsonl = open(os.path.join(self.paths["results_path"], "candidates.jsonl"), "a")
        self.wandb = self._init_wandb(cfg)

    # ---- logging ---------------------------------------------------------
    def _init_wandb(self, cfg):
        wb = cfg.get("wandb", {})
        if not wb.get("enabled"):
            return None
        try:
            import wandb
            return wandb.init(
                project=wb.get("project", "pacevolve-rfg"),
                entity=(wb.get("entity") or None),
                name=f'{wb.get("run_name", "rfg")}-{self.run_id}',
                config={
                    "advisor": self.advisor_name,
                    "islands": [isl["model"] for isl in self.islands],
                    "total_candidates": cfg["experiment"]["max_iters"],
                    "n_samples": cfg["rl"]["n_samples"],
                    "total_steps": cfg["rl"]["total_steps"],
                    "max_workers": self.max_workers,
                    "init_score": cfg["evaluation"]["init_score"],
                    "target_score": cfg["evaluation"]["target_score"],
                },
                reinit=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("wandb init failed (%s); continuing without it.", e)
            return None

    @staticmethod
    def _tag(model):
        return model.split("/")[-1]

    def log_candidate(self, rec):
        with self.lock:
            self.done += 1
            step = self.done
            self.jsonl.write(json.dumps(rec) + "\n")
            self.jsonl.flush()
            best_global = self.best_score
            island_best = self.island_best[rec["island"]]
        if self.wandb is not None:
            payload = {
                "candidate": rec["candidate"], "island": rec["island"],
                "ratio": rec["ratio"] if rec["ratio"] is not None else 0.0,
                "reward": rec["reward"], "compile_ok": int(rec["compile_ok"]),
                "eval_ok": int(rec["eval_ok"]), "elapsed_s": rec["elapsed_s"],
                "best/global": best_global,
                f"best/{self._tag(rec['model'])}": island_best,
            }
            for rg, val in (rec.get("by_regime") or {}).items():
                payload[f"regime/{rg}"] = val
            try:
                self.wandb.log(payload, step=step)
            except Exception as e:  # noqa: BLE001
                logger.warning("wandb.log failed: %s", e)
        return step

    # ---- champion --------------------------------------------------------
    def render_solution(self, region):
        start = self.cfg["compilation"]["edit_start_tag"]
        end = self.cfg["compilation"]["edit_end_tag"]
        pre, rest = self.template_solution.split(f"// {start}", 1)
        _, post = rest.split(f"// {end}", 1)
        return f"{pre}// {start}\n{region}\n// {end}{post}"

    def write_champion(self):
        with self.lock:
            region, score, model = self.best_region, self.best_score, self.best_model
        try:
            full = self.render_solution(region)
            with open(os.path.join(self.paths["results_path"], "champion_solution.cpp"), "w") as f:
                f.write(full)
            frontier = self.paths.get("frontier_solution")
            if frontier:
                frontier = frontier if os.path.isabs(frontier) else os.path.join(_ROOT, frontier)
                if os.path.isdir(os.path.dirname(frontier)):
                    with open(frontier, "w") as f:
                        f.write(full)
            with open(os.path.join(self.paths["results_path"], "champion.json"), "w") as f:
                json.dump({"score": score, "model": model, "region": region}, f, indent=2)
            logger.info("champion written: score=%.6f model=%s", score, model)
        except Exception as e:  # noqa: BLE001
            logger.error("write_champion failed: %s", e)

    def maybe_update_champion(self, region, score, model):
        improved = False
        with self.lock:
            if score is not None and score > self.best_score:
                self.best_score, self.best_region, self.best_model = score, region, model
                improved = True
        if improved:
            for i in range(self.num_islands):  # migrate champion into every island
                self.db.register_program(region, i, score)
            self.write_champion()
        return improved

    # ---- one candidate ---------------------------------------------------
    def _candidate_cfg(self, wdir, island):
        icfg = copy.deepcopy(self.cfg)
        icfg["llm"] = {**self.cfg["llm"], **island["overlay"]}
        icfg["paths"]["src_path"] = wdir["src"]
        icfg["paths"]["build_dir"] = wdir["build"]
        return icfg

    def run_candidate(self, island_id, candidate_idx):
        if self.stop.is_set():
            return None
        island = self.islands[island_id]
        model = island["model"]
        t0 = time.time()
        rec = {"candidate": candidate_idx, "island": island_id, "model": model,
               "ratio": None, "reward": -1.0, "compile_ok": False, "eval_ok": False,
               "valid_all": False, "by_regime": None, "idea": "", "elapsed_s": 0.0,
               "error": None}
        wdir = self.pool.get()
        score = None
        try:
            shutil.copy(self.template_path, wdir["file"])  # clean working copy
            icfg = self._candidate_cfg(wdir, island)
            parent, _ = self.db.get_candidate_for_island(island_id)

            # --- advisor: pick one idea (no code) ---
            with self.lock:
                hist = list(island["history"])[-8:]
            hist_lines = "\n".join(f"- {i[:120]} -> {s:.4f}" for i, s in hist)
            adv_tr = llm_utils.Transcript()
            adv_tr.append(llm_utils.ContentChunk(
                self.prompts.build_advisor_prompt(parent, hist_lines), "user", tags=["advisor"]))
            adv_resp = llm_utils.generate_completion(self.advisor_name, adv_tr, self.advisor_cfg)
            idea = parse_idea(adv_resp)
            rec["idea"] = idea[:300]

            # --- implementer: rewrite region with robust compile/eval loops ---
            tr = llm_utils.Transcript()
            tr.append(llm_utils.ContentChunk(
                self.prompts.build_implementer_prompt(parent, idea), "user", tags=["impl"]))
            resp = llm_utils.generate_completion(model, tr, icfg)
            tr.append(llm_utils.ContentChunk(resp, "model", tags=["impl"]))

            trial = workflow_utils.AlgorithmTrial()
            trial = workflow_utils.edit_until_compile(
                model, trial, tr, wdir["compile_cfg"], icfg,
                loop_config=self.cfg["workflow_loops"]["initial_compile"])
            rec["compile_ok"] = bool(trial.compile_success)
            if trial.compile_success:
                trial = workflow_utils.edit_until_successful_eval(
                    model, trial, tr, wdir["compile_cfg"], self.eval_configs, icfg,
                    candidate_id=candidate_idx,
                    baseline_id=self.cfg["experiment"]["initial_baseline_id"],
                    loop_config=self.cfg["workflow_loops"]["initial_eval"])
                rec["eval_ok"] = all(trial.eval_success) if trial.eval_success else False
                if rec["eval_ok"]:
                    score = self.eval_utils.parse_eval_results(trial.eval_results)
                    rec["by_regime"] = _extract_regime("\n".join(trial.eval_results))

            region = trial.algorithm_implementation or parent
            rec["ratio"] = score
            rec["valid_all"] = score is not None
            rec["reward"] = float(rl_rewards.shape_reward(score, self.reward_cfg))
            if score is not None:
                self.db.register_program(region, island_id, score)
                with self.lock:
                    island["history"].append((idea, score))
                    if score > self.island_best[island_id]:
                        self.island_best[island_id] = score
                self.maybe_update_champion(region, score, model)
        except Exception as e:  # noqa: BLE001 — never let one candidate kill the run
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            logger.exception("candidate %d (island %d) failed", candidate_idx, island_id)
        finally:
            self.pool.put(wdir)
        rec["elapsed_s"] = round(time.time() - t0, 2)
        step = self.log_candidate(rec)
        logger.info(
            "[%d/%d] cand=%d island=%d %s ratio=%s best=%.4f(%s) %.0fs",
            step, self.cfg["experiment"]["max_iters"], candidate_idx, island_id,
            self._tag(model), f"{score:.4f}" if score is not None else "FAIL",
            self.best_score, self._tag(self.best_model), rec["elapsed_s"])
        return rec

    def run(self):
        total = int(self.cfg["experiment"]["max_iters"])
        tasks = [(idx % self.num_islands, idx) for idx in range(total)]
        logger.info("starting run: %d candidates, %d islands, %d workers (%s)",
                    total, self.num_islands, self.max_workers,
                    ", ".join(isl["model"] for isl in self.islands))
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = [ex.submit(self.run_candidate, isl, idx) for (isl, idx) in tasks]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception:  # noqa: BLE001
                    logger.exception("worker future raised")
                if self.stop.is_set():
                    break
        self.write_champion()
        logger.info("run complete: best=%.6f model=%s over %d candidates",
                    self.best_score, self.best_model, self.done)

    def close(self):
        self.write_champion()
        try:
            self.jsonl.close()
        except Exception:
            pass
        if self.wandb is not None:
            try:
                self.wandb.summary["best_score"] = self.best_score
                self.wandb.summary["best_model"] = self.best_model
                self.wandb.finish()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="PACEvolve++ RFG per-island ensemble driver")
    ap.add_argument("--task_id", default="rectangle_free_grid")
    ap.add_argument("--run_id", type=int, default=1)
    ap.add_argument("--dataset_id", default=".")
    ap.add_argument("--max_iters", type=int, default=None, help="override candidate count")
    ap.add_argument("--max_workers", type=int, default=None, help="override worker slots")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    load_env_file(_ROOT)
    if args.dataset_id != ".":
        config_path = os.path.join(_ROOT, "tasks", args.task_id, "config",
                                   args.dataset_id, f"config_{args.run_id}.yaml")
    else:
        config_path = os.path.join(_ROOT, "tasks", args.task_id, "config",
                                   f"config_{args.run_id}.yaml")
    cfg = resolve_config(config_path)
    if args.max_iters is not None:
        cfg["experiment"]["max_iters"] = args.max_iters
    if args.max_workers is not None:
        cfg.setdefault("run", {})["max_workers"] = args.max_workers
    if args.no_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(
                cfg["paths"]["log_dir"], f"rfg_evolve_{timestamp}.log")),
        ],
    )
    # keep third-party http chatter out of the run log
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    prompts = import_module(f"tasks.{args.task_id}.config.prompts")
    eval_utils = import_module(f"tasks.{args.task_id}.eval.eval_utils")

    ens = Ensemble(cfg, prompts, eval_utils, run_id=timestamp)

    def _graceful(signum, frame):
        logger.warning("signal %s received; writing champion and stopping.", signum)
        ens.stop.set()
    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    try:
        ens.run()
    finally:
        ens.close()


if __name__ == "__main__":
    main()
