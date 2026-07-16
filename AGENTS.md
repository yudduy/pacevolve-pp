# AGENTS.md

PACEvolve++ replication — advisor-model RL for evolutionary program search. Architecture,
paper mapping, and layout live in README.md; this file is the operational delta for agents.

## Commands

```bash
.venv/bin/python -m pytest    # full suite, 143 tests, ~1s — must pass before every commit
```

One-time after clone: `git config core.hooksPath hooks` (pre-commit runs the suite).

## Constraints

- Runners resolve tasks dynamically via `tasks/<task_id>/` — keep that layout when adding tasks.
- `tasks/eplb` is the only fully runnable task (GPU-free). `kuairec` and `multi_evolve` are
  contract-complete skeletons that need external datasets and a GPU evaluator — leave them
  skeletal unless those are provided.
