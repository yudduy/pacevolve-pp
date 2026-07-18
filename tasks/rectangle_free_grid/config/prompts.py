"""Prompt templates + seed program for the rectangle-free-grid task.

The evolutionary driver treats the text between the // RegexTagEvolveStart /
// RegexTagEvolveEnd markers of src/solution.cpp as the "program" being evolved.
RFG_SEED is that initial region (registered as the population's template). The
advisor selects a strategic idea; the implementer rewrites the region.
"""

# The initial evolvable region (must match the seed between the markers in
# src/solution.cpp). It is the population's template / sota_algo.
RFG_SEED = """static vector<vector<int>> evolve_solve(int small, int large) {
    Candidate best;
    if (small == 316 && large == 316) {
        consider(best, pbd_316_exact_blocks(small, large), small, large);
    } else {
        consider(best, pair_blocks(small, large), small, large);
        consider(best, geometry_blocks(small, large), small, large);
        consider(best, projective_augmented_full_blocks(small, large), small, large);
        consider(best, projective_excluded_23_blocks(small, large), small, large);
        consider(best, projective_subset_blocks(small, large), small, large);
        consider(best, projective_alternating_subset_blocks(small, large), small, large);
        consider(best, greedy_blocks(small, large), small, large);
        consider(best, shuffled_clique_blocks(small, large), small, large);
    }
    return best.blocks;
}"""


PROBLEM_BRIEF = """PROBLEM (Zarankiewicz z(n,m;2,2) / rectangle-free grid):
On an n x m grid, choose as many black cells as possible so that NO four chosen
cells form the corners of an axis-parallel rectangle (equivalently: no two rows
share two columns; the bipartite rows-vs-columns graph is C4-free).

SCORING: per test, score = min(k / (1.5 * U(n,m)), 1) where k = #cells you place
and U(n,m) = floor(min(n*sqrt(m)+m, m*sqrt(n)+n, n*m)) is a Kovari-Sos-Turan-type
upper bound. The true optimum is <= U, so the practical ceiling is ~0.62-0.667.
The final score is the MEAN over a hidden set of (n,m) with n*m <= 100000. The
game is purely: MAXIMISE k (a valid, large rectangle-free construction) per case.

BLOCK MODEL: the code works in an oriented frame with small = min(n,m) <= large =
max(n,m). A construction returns `blocks`: up to `large` rows, each row a vector of
column indices in [0, small). Two rows sharing >= 2 columns == a rectangle. So the
blocks must form a LINEAR HYPERGRAPH: every pair of the `small` columns appears in
at most one row. Total cells k = sum of block sizes. Maximising k under that
constraint is exactly the Zarankiewicz problem; near-optimal constructions come
from projective planes / affine incidence / Sidon (B2) sets / difference families,
with greedy + local search filling the gaps for arbitrary (small,large)."""


CONSTRUCTIONS_DOC = """AVAILABLE LIBRARY (already defined earlier in the same file; call freely):
  struct Candidate { vector<vector<int>> blocks; long long score; };
  void consider(Candidate& best, vector<vector<int>> blocks, int small, int large);
      // pads/scores `blocks` and keeps it in `best` if it beats the current best.
  bool valid_blocks(const vector<vector<int>>& blocks, int small);
  vector<vector<int>> pair_blocks(int small, int large);                    // all 2-subsets
  vector<vector<int>> greedy_blocks(int small, int large);                  // degree-targeted greedy cliques
  vector<vector<int>> shuffled_clique_blocks(int small, int large);         // randomized clique packing
  vector<vector<int>> geometry_blocks(int small, int large);               // finite-geometry incidence
  vector<vector<int>> projective_subset_blocks(int small, int large);       // projective-plane subset
  vector<vector<int>> projective_alternating_subset_blocks(int small,int large);
  vector<vector<int>> projective_augmented_full_blocks(int small, int large);
  vector<vector<int>> projective_excluded_23_blocks(int small, int large);
  vector<vector<int>> pbd_316_exact_blocks(int small, int large);          // tuned for 316x316
  // Finite-field helpers (see struct Field earlier) are available for prime-power q.

HEADROOM (from local evaluation of the seed, mean ~0.601):
  - THIN / SKEWED grids are weakest: 50x2000 ~0.51, 20x5000 ~0.54, 100x1000 ~0.57.
  - NON-prime-power squares lag: 100x100 ~0.59, 200x200 ~0.59.
  - Prime-power-ish sizes (q^2+q+1: 91,127,307,...) already ~0.63-0.65 (near ceiling).
Focus effort where the gap is largest; do not regress the strong regimes."""


OUTPUT_CONTRACT = """OUTPUT RULES (violating any of these wastes the whole attempt):
- Emit EXACTLY ONE fenced ```cpp code block. Reason first if you like, but the
  final message MUST contain that one block.
- The block is the COMPLETE replacement for the region between the markers. It MUST
  define:  static vector<vector<int>> evolve_solve(int small, int large)
  and MAY define helper free functions ABOVE it (in the same block).
- Do NOT include: #include lines, `using namespace std;`, `int main()`, the
  `// RegexTagEvolve...` markers, or any function that already exists elsewhere in
  the file (they live OUTSIDE the region and will cause redefinition errors).
- `using namespace std;` is already in effect; the whole standard library and all
  the listed constructions are in scope. Write the FULL code — never elide with
  `// unchanged` / `// same as before` / `...`.
- Auto-sanitised downstream (rectangle violations are greedily pruned), so a
  slightly-invalid or over-full return never scores 0 — but MAXIMISING valid kept
  cells is the whole objective. Beat the current best; target the weak regimes.
- Must compile with `g++ -O2 -std=c++17` and finish within ~1s per case
  (n*m <= 100000, memory <= 512 MB). Keep it deterministic (seed any RNG)."""


def build_advisor_prompt(parent_region: str, history: str, dims_hint: str = "") -> str:
    """Advisor: pick ONE high-leverage idea (no code)."""
    hist = history.strip() or "(no attempts recorded yet on this island)"
    return f"""You are the ADVISOR in an evolutionary search for better rectangle-free
grid constructions. You do NOT write code. You choose ONE high-leverage, NOVEL
improvement idea for a strong coder to implement next.

{PROBLEM_BRIEF}

CURRENT STRATEGY (the evolvable region to improve):
```cpp
{parent_region}
```

RECENT ATTEMPTS ON THIS ISLAND (idea -> mean score; seed baseline ~0.601; higher is better):
{hist}
{dims_hint}
Propose ONE concrete, specific improvement most likely to raise the MEAN score and
NOT already tried above. Name the construction and the TARGET REGIME (thin grids,
non-prime-power squares, ...), and say briefly how it maps to blocks (linear
hypergraph). Prefer ideas that generalise across (n,m), not memorised tables.

Respond in EXACTLY this format and nothing else:
Idea: <one or two sentences: the construction/change + target regime>
Why: <one sentence: why it should beat the current best>"""


def build_implementer_prompt(parent_region: str, idea: str) -> str:
    """Implementer: realize the idea as the new region (complete fenced block)."""
    return f"""You are the IMPLEMENTER: a top competitive-programming C++ engineer.
Rewrite the evolvable region to realise the idea below, maximising valid cells.

{PROBLEM_BRIEF}

{CONSTRUCTIONS_DOC}

CURRENT REGION (parent to improve):
```cpp
{parent_region}
```

IDEA TO IMPLEMENT:
{idea}

STRATEGY (critical for not regressing):
- The region already tries several constructions via `consider(best, <blocks>, small, large)`
  and KEEPS THE HIGHEST-SCORING one, so it never regresses across regimes.
- STRONGLY PREFER TO EXTEND, NOT REPLACE: keep the existing `consider(...)` calls and
  ADD your new construction as another `consider(best, your_new_blocks(small,large), small, large)`.
  That way a weak new idea can only help (it is discarded if worse), never hurt other regimes.
- Only delete an existing `consider(...)` line if your construction strictly dominates it.
- Gate regime-specific constructions on (small,large) so they only fire where they help
  (e.g. thin grids: `large >= 8*small`; near-square; prime-power-ish sizes).
- Always emit the COMPLETE region (all helper functions + evolve_solve).

{OUTPUT_CONTRACT}"""


# --- Compatibility shims for the shipped advisor-RL driver (optional path) -----
# The shipped run_advisor_rl.py expects these names; the custom run_rfg_evolve.py
# driver uses the build_* functions above instead. Provided for completeness.
def construct_idea_select_no_code_prompt(sota_algo, idea_repo):
    history = ""
    try:
        history = "\n".join(
            f"- {getattr(i, 'description', '')}" for i in getattr(idea_repo, "ideas", [])
        )
    except Exception:
        history = ""
    return build_advisor_prompt(getattr(idea_repo, "sota", sota_algo) or sota_algo, history)


def construct_code_impl_prompt(sota_algo, idea_id, exp_description):
    return build_implementer_prompt(sota_algo, exp_description or "Improve the construction.")


def construct_idea_gen_prompt(parent, idea_repo):
    return build_advisor_prompt(parent, "")
