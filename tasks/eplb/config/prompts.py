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

EPLB = '''
def assign_experts(expert_loads, num_devices):
    """Assign each expert to a device. Returns a list mapping expert index ->
    device index in [0, num_devices). Greedy longest-processing-time baseline."""
    order = sorted(range(len(expert_loads)), key=lambda e: -expert_loads[e])
    device_load = [0.0] * num_devices
    assignment = [0] * len(expert_loads)
    for e in order:
        d = min(range(num_devices), key=lambda dev: device_load[dev])
        assignment[e] = d
        device_load[d] += expert_loads[e]
    return assignment
'''


BACKGROUND = """
### Background on Expert-Parallelism Load Balancing

Expert-parallel mixture-of-experts systems place experts on parallel devices.
For each synthetic activation profile, `expert_loads[e]` is the work assigned
to expert `e`. The candidate maps every expert to exactly one integer device in
`[0, num_devices)`. Its goal is to minimize the maximum aggregate load on any
device without making the assignment routine slow.

Balancedness is mean device load divided by maximum device load, so it lies in
`(0, 1]` and is best at `1.0`. Speed is the reference runtime divided by the
candidate runtime, capped at `1.0`. The evaluation score is
`0.5 * balancedness + 0.5 * speed`; larger is better. Invalid assignments score
zero. Profiles are deterministic, seeded, Zipf-like expert activation counts.
"""


TASK_INTRO = """
You are an expert systems researcher improving expert-parallelism load
balancing. Develop a fast assignment heuristic that distributes expert
activation load evenly across the available devices.
"""


CODING_REQ = """
While completing the coding task, you MUST:
- Answer with exactly one Markdown Python code block.
- Define `assign_experts(expert_loads, num_devices)` with the exact signature.
- Return a list mapping every expert index to an integer device index in
  `[0, num_devices)`.
- Include the `assign_experts` function only; do not add edit tags, a class, the
  evaluator, or global imports. Put any stdlib or NumPy imports inside the
  function.
- Use only Python's standard library and NumPy. Do not use GPUs, model training,
  external packages, files, randomness, or profile-specific hard-coding.
- Keep the implementation fast as well as balanced because the score is the
  mean of balancedness and speed.
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

Propose three distinct, testable ideas for improving load balancedness or
assignment speed. For each idea, explain its hypothesis, mechanism, expected
tradeoff, and why it differs from experiments already in the repository.

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
been run. Balance exploration against exploitation and use observed experiment
results rather than inventing an idea identifier.

Start the selection using exactly this format:
Idea ID: <id>
Experiment description: <desc>

Then implement that experiment.

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

Select one existing idea and describe a concrete experiment that has not
already been run. Balance exploration against exploitation and do not invent
an idea identifier. Do not write code yet.

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
history, select the most useful untried option, and implement it. State the
rationale and expected balancedness/speed tradeoff briefly before the code.

{CODING_REQ}
"""


SUMMARIZE_EVAL_PROMPT = """
## Your Task

Summarize this experiment in one short paragraph followed by exactly one short
bullet point for the knowledge base. Begin the bullet with the candidate's
score, balancedness, and speed, then name the implemented change and the lesson
supported by those results. Be concrete and avoid generic claims.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The result reports `score`, `balancedness`, `speed`, and `valid` for synthetic
expert-activation profiles.

### Understanding metrics
Higher is better. Balancedness is mean device load divided by peak device load;
speed is reference time divided by candidate time, capped at one; score is the
equal-weight mean of balancedness and speed. Invalid assignments score zero.
"""
