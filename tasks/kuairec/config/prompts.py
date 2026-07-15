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

"""Prompt contract for the KuaiRec sequential-recommendation task."""

KUAIREC = '''
def build_recommender(num_items, embedding_dim=64):
    """Build a compact FuXi-linear-style next-item ranking model."""
    import torch

    class FuXiLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.items = torch.nn.Embedding(
                num_items + 1, embedding_dim, padding_idx=0
            )
            self.mixer = torch.nn.Linear(embedding_dim, embedding_dim)
            self.bias = torch.nn.Parameter(torch.zeros(num_items))

        def forward(self, item_sequences):
            mask = item_sequences.ne(0)
            embedded = self.items(item_sequences) * mask.unsqueeze(-1)
            counts = mask.cumsum(dim=1).clamp_min(1).unsqueeze(-1)
            prefix_state = embedded.cumsum(dim=1) / counts
            prefix_state = torch.tanh(self.mixer(prefix_state))
            last = mask.sum(dim=1).clamp_min(1) - 1
            state = prefix_state[
                torch.arange(item_sequences.shape[0], device=item_sequences.device),
                last,
            ]
            return state @ self.items.weight[1:].T + self.bias

    return FuXiLinear()
'''


BACKGROUND = """
### Background on KuaiRec sequential recommendation

The task ranks the next item for each user from an ordered KuaiRec interaction
history. A candidate implements `build_recommender(num_items,
embedding_dim=64)` and returns a PyTorch model whose forward pass accepts a
padded integer tensor of item histories and returns one score per catalog item.
Item id zero is padding; catalog items are numbered from one through
`num_items`.

Candidates are trained for exactly 16 epochs with sampled-softmax negatives and
must finish training plus evaluation within 1,200 seconds on the evaluator GPU.
Quality is measured by NDCG@10, HR@10, and MRR. The optimization score is their
arithmetic mean, so higher is better.
"""


TASK_INTRO = """
You are an expert in efficient sequential recommendation. Improve next-item
ranking on KuaiRec while respecting the fixed 16-epoch sampled-softmax training
protocol and the 1,200-second end-to-end budget.
"""


CODING_REQ = """
While completing the coding task, you MUST:
- Answer with exactly one Markdown Python code block.
- Define `build_recommender(num_items, embedding_dim=64)` with the exact
  signature and return a `torch.nn.Module`.
- The model's forward method must accept padded item-history tensors and return
  a score for each of the `num_items` catalog items.
- Include the `build_recommender` function only; do not add edit tags, a
  top-level class, the trainer, evaluator, or global imports. Put imports and
  any helper class inside the function.
- Do not access files, networks, test labels, or dataset-specific identifiers.
- Keep the model compatible with 16-epoch sampled-softmax GPU training and the
  1,200-second total evaluation budget.
"""


def construct_idea_gen_prompt(sota, idea_repo):
    return f"""
{TASK_INTRO}

{BACKGROUND}

### Current state-of-the-art
```python
{sota}
```

### Idea repository
{idea_repo}

## Your Task

Propose three distinct, testable ideas for improving next-item ranking. For
each, explain the hypothesis, model or optimization mechanism, expected effect
on NDCG@10/HR@10/MRR, runtime cost, and difference from prior experiments.

Use this format:
** Idea 1 **
Hypothesis: <idea>
Reasoning: <reasoning>

Continue the same format for Ideas 2 and 3. Do not write code yet.
"""


def construct_idea_select_prompt(sota, idea_repo):
    return f"""
{TASK_INTRO}

{BACKGROUND}

### Current state-of-the-art
```python
{sota}
```

### Idea repository
{idea_repo}

## Your Task

Select one existing idea and propose a concrete experiment that has not already
been run. Use observed results, balance ranking quality against the runtime
budget, and do not invent an idea identifier.

Start with exactly this format:
Idea ID: <id>
Experiment description: <desc>

Then implement the experiment.

{CODING_REQ}
"""


def construct_idea_select_no_code_prompt(sota, idea_repo):
    return f"""
{TASK_INTRO}

{BACKGROUND}

### Current state-of-the-art
```python
{sota}
```

### Idea repository
{idea_repo}

## Your Task

Select one existing idea and describe a concrete, untried experiment. Use the
experiment history and do not write code yet.

Respond using exactly this format:
Idea ID: <id>
Experiment description: <desc>
"""


def construct_code_impl_prompt(sota, idea_id, exp_description):
    return f"""
{TASK_INTRO}

{BACKGROUND}

### Current state-of-the-art
```python
{sota}
```

## Your Task

Implement this selected experiment:

Idea ID: {idea_id}
Experiment description: {exp_description}

{CODING_REQ}
"""


def construct_mutation_prompt(sota, ablation_list):
    ablation_descriptions = "\n".join(ablation_list)
    return f"""
{TASK_INTRO}

{BACKGROUND}

### Current state-of-the-art
```python
{sota}
```

### Experiment knowledge base
{ablation_descriptions}

## Your Task

Brainstorm several materially different mutations informed by the experiment
history. Select a useful untried option and briefly state its expected ranking
quality and runtime tradeoff before implementing it.

{CODING_REQ}
"""


SUMMARIZE_EVAL_PROMPT = """
## Your Task

Summarize this experiment in one short paragraph followed by exactly one short
bullet point for the knowledge base. Begin the bullet with the mean score,
NDCG@10, HR@10, MRR, and runtime, then identify the implemented change and the
lesson supported by those results. Be concrete and avoid generic claims.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
Results report NDCG@10, HR@10, MRR, their arithmetic mean, and total runtime for
the KuaiRec next-item-ranking evaluation.

### Understanding metrics
Higher is better for all ranking metrics. The optimization score is the
arithmetic mean of NDCG@10, HR@10, and MRR. Training uses 16 sampled-softmax
epochs, and training plus evaluation must complete within 1,200 seconds.
"""
