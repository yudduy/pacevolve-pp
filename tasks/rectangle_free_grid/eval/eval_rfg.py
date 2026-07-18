#!/usr/bin/env python3
"""Standalone evaluator for the rectangle-free-grid task.

Compiles a candidate solution.cpp, runs it on a fixed representative set of (n,m)
grids under a wall-clock limit, scores each with the faithful chk.cc-parity scorer
(score_rfg.cpp), and prints:
  - one flat, ast.literal_eval-parseable line:  Candidate: {'score':..., ...}
    where 'score' is the mean per-case ratio (the quantity FrontierCS averages).
  - one JSON line:                              Detail: {"per_case":[...], ...}
    with per-case + per-regime breakdown for W&B logging by the driver.

Portable: auto-detects g++/clang++; no testlib / no FrontierCS repo dependency.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCORER_SRC = os.path.join(HERE, "score_rfg.cpp")

# Fixed representative spread within n*m <= 100000. Includes the 3 visible cases
# (100x100, 10x10000, 100x1000); weighted toward the regimes with real headroom
# (non-prime-power squares, thin/skewed grids). Deterministic -> stable reward.
DEFAULT_CASES = [
    (2, 2), (10, 10), (100, 100), (150, 150), (200, 200), (256, 256), (316, 316),
    (91, 91), (127, 127), (307, 307),
    (100, 1000), (200, 500), (300, 333), (500, 200),
    (50, 2000), (20, 5000), (10, 10000), (4, 25000), (2, 50000),
]

ANCHOR_CASES = [(100, 100), (10, 10000), (100, 1000), (316, 316), (2, 50000)]
# 3 visible judge samples + large square + extreme thin; held-out progress metric,
# never part of the reward score.


def resolve_case_seed(cli_seed, environ):
    if cli_seed is not None:
        return int(cli_seed)
    env_seed = environ.get("PACE_EVAL_CASE_SEED")
    if env_seed:
        return int(env_seed)
    return None


def sample_cases(seed: int, count: int = 16) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    cases = []
    seen = set(ANCHOR_CASES) | {(m, n) for n, m in ANCHOR_CASES}

    def add(case):
        if len(cases) < count and case not in seen:
            seen.add(case)
            cases.append(case)

    def draw_rectangle():
        aspect = math.exp(rng.uniform(math.log(1.5), math.log(20)))
        area = math.exp(rng.uniform(math.log(1000), math.log(100000)))
        n = max(2, round(math.sqrt(area / aspect)))
        m = max(2, min(100000 // n, round(n * aspect)))
        return n, m

    for _ in range(4):
        side = round(math.exp(rng.uniform(math.log(2), math.log(316))))
        add((side, side))

    for _ in range(5):
        add(draw_rectangle())

    for _ in range(4):
        short = rng.randint(1, 10)
        lower, upper = 20 * short, 100000 // short
        long = round(math.exp(rng.uniform(math.log(lower), math.log(upper))))
        long = max(lower, min(upper, long))
        add(rng.choice(((short, long), (long, short))))

    for _ in range(2):
        n = rng.randint(1, 20)
        add((n, rng.randint(1, 400 // n)))

    area = rng.randint(90000, 100000)
    aspect = math.exp(rng.uniform(math.log(1), math.log(10)))
    n = max(1, round(math.sqrt(area / aspect)))
    m = max(1, min(area // n, 100000 // n))
    add((n, m))

    while len(cases) < count:
        add(draw_rectangle())
    return cases


def regime(n, m):
    lo, hi = min(n, m), max(n, m)
    if lo == hi:
        # square: prime-power-ish (q^2+q+1) constructions are near-optimal
        return "square"
    if lo <= 4 or hi >= 20 * lo:
        return "thin"
    return "rect"


def detect_cxx(explicit=None):
    candidates = ([explicit] if explicit else []) + ["g++", "clang++", "c++"]
    for c in candidates:
        if c and shutil.which(c):
            return c
    raise RuntimeError("no C++ compiler found (need g++/clang++)")


def compile_cpp(cxx, src, out):
    cmd = [cxx, "-O2", "-std=c++17", "-o", out, src]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def ensure_scorer(cxx, build_dir):
    out = os.path.join(build_dir, "score_rfg")
    stale = (not os.path.exists(out)) or (
        os.path.getmtime(out) < os.path.getmtime(SCORER_SRC)
    )
    if stale:
        rc, so, se = compile_cpp(cxx, SCORER_SRC, out)
        if rc != 0:
            raise RuntimeError(f"scorer build failed:\n{se or so}")
    return out


def run_case(solbin, scorer, n, m, tl):
    with tempfile.TemporaryDirectory() as d:
        inf = os.path.join(d, "in")
        outf = os.path.join(d, "out")
        with open(inf, "w") as f:
            f.write(f"{n} {m}\n")
        t0 = time.time()
        try:
            with open(inf) as fi, open(outf, "w") as fo:
                subprocess.run([solbin], stdin=fi, stdout=fo,
                               stderr=subprocess.DEVNULL, timeout=tl)
        except subprocess.TimeoutExpired:
            return {"n": n, "m": m, "valid": False, "reason": "TIMEOUT",
                    "ratio": 0.0, "unbounded": 0.0, "time": tl}
        dt = time.time() - t0
        r = subprocess.run([scorer, inf, outf], capture_output=True, text=True)
        try:
            j = json.loads(r.stdout.strip())
        except Exception:
            j = {"valid": False, "reason": "SCORER_ERR", "ratio": 0.0, "unbounded": 0.0}
        j.update({"n": n, "m": m, "time": round(dt, 3)})
        j.setdefault("ratio", 0.0)
        j.setdefault("unbounded", j.get("ratio", 0.0))
        return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--cxx", default=None)
    ap.add_argument("--tl", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--build_dir", default=None)
    ap.add_argument("--compile_only", action="store_true")
    a = ap.parse_args()
    seed = resolve_case_seed(a.seed, os.environ)

    try:
        cxx = detect_cxx(a.cxx)
    except Exception as e:
        print(f"Candidate: {{'score': -1.0, 'valid_all': False, 'error': {str(e)!r}}}")
        return 1

    build_dir = a.build_dir or tempfile.mkdtemp(prefix="rfg_build_")
    os.makedirs(build_dir, exist_ok=True)
    solbin = os.path.join(build_dir, "solution")

    rc, so, se = compile_cpp(cxx, a.solution, solbin)
    if rc != 0:
        sys.stderr.write(se or so)  # surface compiler errors to the compile-fix loop
        print("Candidate: {'score': -1.0, 'valid_all': False, 'compile': False}")
        return 1
    if a.compile_only:
        print("OK")
        return 0

    try:
        scorer = ensure_scorer(cxx, build_dir)
    except Exception as e:
        print(f"Candidate: {{'score': -1.0, 'valid_all': False, 'error': {str(e)!r}}}")
        return 1

    cases = DEFAULT_CASES if seed is None else sample_cases(seed)
    per = []
    rr = uu = 0.0
    all_valid = True
    worst = (1e9, None)
    reg_sum, reg_cnt = {}, {}
    for (n, m) in cases:
        j = run_case(solbin, scorer, n, m, a.tl)
        rg = regime(n, m)
        per.append({"nm": f"{n}x{m}", "regime": rg, "ratio": round(j["ratio"], 4),
                    "unbounded": round(j["unbounded"], 4), "valid": bool(j.get("valid")),
                    "k": j.get("k"), "U": j.get("U"), "time": j.get("time"),
                    "reason": j.get("reason", "")})
        rr += j["ratio"]
        uu += j["unbounded"]
        if not j.get("valid"):
            all_valid = False
        if j["ratio"] < worst[0]:
            worst = (j["ratio"], f"{n}x{m}")
        reg_sum[rg] = reg_sum.get(rg, 0.0) + j["ratio"]
        reg_cnt[rg] = reg_cnt.get(rg, 0) + 1

    n_cases = len(cases)
    flat = {
        "score": round(rr / n_cases, 6),
        "unbounded": round(uu / n_cases, 6),
        "valid_all": all_valid,
        "n_cases": n_cases,
        "worst_ratio": round(worst[0], 4),
        "worst_case": worst[1],
    }
    detail = {
        "score": flat["score"],
        "by_regime": {k: round(reg_sum[k] / reg_cnt[k], 4) for k in reg_sum},
        "per_case": per,
    }
    if seed is not None:
        anchor_per = []
        anchor_rr = 0.0
        for (n, m) in ANCHOR_CASES:
            j = run_case(solbin, scorer, n, m, a.tl)
            rg = regime(n, m)
            anchor_per.append({
                "nm": f"{n}x{m}", "regime": rg, "ratio": round(j["ratio"], 4),
                "unbounded": round(j["unbounded"], 4), "valid": bool(j.get("valid")),
                "k": j.get("k"), "U": j.get("U"), "time": j.get("time"),
                "reason": j.get("reason", "")})
            anchor_rr += j["ratio"]
        flat["case_seed"] = seed
        flat["anchor_score"] = round(anchor_rr / len(ANCHOR_CASES), 6)
        detail["cases"] = cases
        detail["anchor_per_case"] = anchor_per
    print("Candidate: " + repr(flat))
    print("Detail: " + json.dumps(detail))
    return 0


if __name__ == "__main__":
    sys.exit(main())
