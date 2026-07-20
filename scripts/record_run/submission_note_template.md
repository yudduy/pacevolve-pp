# PACEvolve++ advisor-RL — rectangle-free grid (Zarankiewicz z(n,m;2,2))

**Approach.** Advisor/implementer decomposition (PACEvolve++, arXiv 2605.07039): a small,
RL-trained *advisor* proposes one high-leverage construction idea per rollout (it writes no
code); a stronger frozen *implementer* turns the idea into a C++ diff; candidates are
compiled and scored, and the advisor is reinforcement-trained on the resulting rewards
(GRPO-style group-relative advantages, importance-sampling surrogate, LoRA rank 32) via a
self-hosted Tinker-compatible server (skyrl-tx, JAX backend, Qwen3-32B tensor-parallel-4).

**Models (credit).**
- *Advisor (strategy, RL-trained):* Qwen3-32B — trained in-loop on this task, `<N_STEPS>` steps,
  n=4 rollouts/step.
- *Implementer (code author):* DeepSeek-V4-Pro (via OpenRouter).
- *Harness:* PACEvolve++ replication; evolutionary islands + per-step case resampling to
  prevent fixed-case overfitting.

**Result (local proxy, honest).** The submitted `solution.cpp` is the pool champion after
re-ranking every archived candidate across a fixed 20-seed panel (per-step max scores use
different seeds and are not comparable). Champion cross-seed mean = `<MEAN>`, median `<MEDIAN>`,
range [`<MIN>`, `<MAX>`], clearing the current leaderboard best (0.5833) on `<GE>`/20 seeds.
Note: the evaluator is time-limited (1.0s) and nondeterministic, so single-run scores wobble.

**Caveat.** `<CAVEAT>`
