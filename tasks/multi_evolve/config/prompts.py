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

"""Prompt contract for the Multi-Evolve protein-fitness task."""

MULTI_EVOLVE = '''
def predict_fitness(train_features, train_fitness, test_features):
    """Fit a ridge model on mutation features and predict test fitness."""
    import numpy as np

    x_train = np.asarray(train_features, dtype=float)
    x_test = np.asarray(test_features, dtype=float)
    y_train = np.asarray(train_fitness, dtype=float).reshape(-1)
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    train_design = np.column_stack(
        [np.ones(len(x_train)), (x_train - mean) / scale]
    )
    test_design = np.column_stack(
        [np.ones(len(x_test)), (x_test - mean) / scale]
    )
    penalty = np.eye(train_design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.pinv(
        train_design.T @ train_design + 1e-2 * penalty
    ) @ train_design.T @ y_train
    return test_design @ weights
'''


BACKGROUND = """
### Background on Multi-Evolve protein fitness extrapolation

The supplied training records are single and double protein mutants represented
by numeric mutation-feature vectors and measured fitness values. The held-out
records are higher-order mutants. A candidate implements
`predict_fitness(train_features, train_fitness, test_features)` and must learn
from the low-order mutants without using held-out labels or dataset-specific
hard-coding.

Each dataset is scored with Pearson correlation and Precision@5. Precision@5 is
the fraction of the five highest predicted variants that are also among the
five variants with highest measured fitness (or all variants when there are
fewer than five). The dataset score is
`0.7 * PearsonR + 0.3 * Precision@5`, and the final metric is the arithmetic
mean of that combined score across datasets. Higher is better.
"""


TASK_INTRO = """
You are an expert protein-engineering and machine-learning researcher. Improve
extrapolation from observed single and double mutants to unseen higher-order
mutants while keeping the candidate compact and robust across datasets.
"""


CODING_REQ = """
While completing the coding task, you MUST:
- Answer with exactly one Markdown Python code block.
- Define `predict_fitness(train_features, train_fitness, test_features)` with
  the exact signature and return one numeric prediction per test row.
- Include the `predict_fitness` function only; do not add edit tags, a class,
  an evaluator, or global imports. Put imports inside the function.
- Use only Python's standard library and NumPy. Do not access files, networks,
  environment variables, test labels, or dataset names.
- Fit only on the supplied single/double-mutant training data and produce
  finite predictions for the higher-order-mutant test features.
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

Propose three distinct, testable ideas for improving higher-order fitness
extrapolation. For each, explain the hypothesis, mechanism, expected effect on
PearsonR and Precision@5, and how it differs from prior experiments.

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
been run. Use observed results, balance exploration against exploitation, and
do not invent an idea identifier.

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
history. Select a useful untried option and briefly state why it should improve
correlation, top-five recovery, or both before implementing it.

{CODING_REQ}
"""


SUMMARIZE_EVAL_PROMPT = """
## Your Task

Summarize this experiment in one short paragraph followed by exactly one short
bullet point for the knowledge base. Begin the bullet with the combined score,
PearsonR, and Precision@5, then identify the implemented change and the lesson
supported by the results. Be concrete and avoid generic claims.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
Results report PearsonR, Precision@5, and their combined score for each protein
fitness dataset, followed by the mean across datasets.

### Understanding metrics
Higher is better. Each dataset score is `0.7 * PearsonR + 0.3 * Precision@5`.
The optimization target is the arithmetic mean of this score across datasets.
"""
