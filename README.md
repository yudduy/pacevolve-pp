# PACEvolve++: Advisor-Model RL for Evolutionary Search

A replication of [PACEvolve++](https://arxiv.org/pdf/2605.07039) — advisor-model reinforcement learning for LLM-driven evolutionary program search — built on the [PACEvolve](https://arxiv.org/pdf/2601.10657) scaffold.

PACEvolve++ adds two things to the base evolutionary loop, both implemented here:

- **Advisor / implementer decomposition (§3.2)** — a small, trainable *advisor* selects an idea without writing code; a stronger *implementer* writes it. The two roles can run on different models (e.g. a local small model for the advisor, a frontier model for the implementer).
- **Search-dynamics-aware phase-adaptive RL (§3.3, Theorem 1)** — progress-normalized reward shaping (Eq. 5), phase-adaptive advantage estimators mixing group-relative and enumeration-exact SLOO branches on a linear α schedule (Eqs. 2–4), and a masked asymmetric clipped surrogate policy loss (Eq. 6), run as a rollout-barrier loop (§3.1). GRPO and entropic (TTT-Discover) baselines ship alongside for like-for-like comparison.

## Repository layout

- `workflows/` — the evolution engine (`run_experiment.py`, population database, prompting utilities) plus the PACEvolve++ layer: `advisor_utils.py`, `rl_rewards.py`, `advantages.py`, `rl_loss.py`, `rl_trainer.py`, and the `run_advisor_rl.py` driver.
- `tasks/` — task definitions. `eplb` (expert-parallelism load balancing, §4.1.1) is fully runnable and GPU-free; `kuairec` and `multi_evolve` are contract-complete skeletons that need external datasets and a GPU evaluator.
- `tests/` — the pytest suite (145 tests), including brute-force checks of the PKPO/SLOO estimators, Theorem-1 property tests, and an end-to-end smoke test of the advisor→implementer→eval→reward→train loop against a fixture task.

The runners resolve tasks dynamically via `tasks/<task_id>/`, so keep that layout when adding tasks.

## Setup

```bash
git clone https://github.com/yudduy/pacevolve-pp.git
cd pacevolve-pp
pip install -r requirements.txt
```

Set API keys for whichever providers you use, and install their clients:

```bash
export GOOGLE_API_KEY="..."     # Gemini
export OPENAI_API_KEY="..."     # OpenAI
export ANTHROPIC_API_KEY="..."  # Anthropic
pip install google-generativeai openai anthropic
```

Local OpenAI-compatible endpoints (Ollama, vLLM) are supported via `client_type: ollama` + `base_url` in a task's config — no key needed.

## Running

Each task is configured by `tasks/<task_id>/config/config_1.yaml` (model, paths, evaluation, `rl` section). Both drivers run from `workflows/`:

```bash
cd workflows

# Base PACEvolve evolutionary loop
python run_experiment.py --task_id eplb

# PACEvolve++ advisor-RL barrier loop
python run_advisor_rl.py --task_id eplb --backend mock --objective pacevolve++ --max_steps 2
```

Give the two roles different models by adding `advisor_llm` / `implementer_llm` sections to the task config; each overlays the base `llm` section. The `rl` section selects the objective (`pacevolve++`, `grpo`, `entropic`, `maxk`, `none`). Real torch RL training of the advisor is a documented `PolicyBackend` seam; the shipped backend is the deterministic mock.

## Tests

From the repository root:

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pytest pyyaml
.venv/bin/pytest
```

## Logs

LLM transcripts and controller logs are written to the paths defined in each task's YAML config (`transcript_*.txt`, `controller_verbose_*.log`).

## Credits & License

The base evolutionary scaffold comes from [MinghaoYan/PACEvolve](https://github.com/MinghaoYan/PACEvolve) (PACEvolve paper implementation); this repository adds the PACEvolve++ advisor-RL layer and tasks on top of it. Licensed under **Apache 2.0** — see `LICENSE`.

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security). It is intended for research and demonstration purposes, not production use.
