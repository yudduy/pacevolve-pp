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

"""Test-only fake task prompts. Exposes the full symbol set the engine and the
advisor/implementer split require, so wiring tests can drive a real prompt flow
without any network or task-specific content.
"""

# Seed program named by experiment.sota_algo_name in the fixture config.
FAKE_SOTA = "def solve(x):\n    return x\n"

CODING_REQ = (
    "Return exactly one markdown-fenced Python code block implementing solve(x). "
    "You may include a line '# SCORE: <float>' to signal the fake evaluator's score."
)


def construct_idea_gen_prompt(sota_algorithm, idea_repo):
    return (
        "Generate ideas for the fake task.\n"
        f"SoTA:\n{sota_algorithm}\n"
        f"Idea Repo:\n{idea_repo}\n"
        "Format each as:\nHypothesis: ...\nReasoning: ..."
    )


def construct_idea_select_prompt(sota_algorithm, idea_repo):
    return (
        "Select an idea and implement it (code inline) for the fake task.\n"
        f"SoTA:\n{sota_algorithm}\n"
        f"Idea Repo:\n{idea_repo}\n"
        "Format:\nIdea ID: <id>\nExperiment description: <desc>\n"
        f"{CODING_REQ}"
    )


def construct_idea_select_no_code_prompt(sota_algorithm, idea_repo):
    return (
        "Select an idea to test for the fake task. DO NOT write code.\n"
        f"SoTA:\n{sota_algorithm}\n"
        f"Idea Repo:\n{idea_repo}\n"
        "Format:\nIdea ID: <id>\nExperiment description: <desc>"
    )


def construct_code_impl_prompt(sota_algorithm, idea_id, exp_description):
    return (
        "Implement the selected idea for the fake task.\n"
        f"SoTA:\n{sota_algorithm}\n"
        f"Idea ID: {idea_id}\n"
        f"Experiment description: {exp_description}\n"
        f"{CODING_REQ}"
    )


def construct_mutation_prompt(sota_algorithm, ablation_list):
    return (
        "Mutate the fake task solution.\n"
        f"SoTA:\n{sota_algorithm}\n"
        f"Past experiments:\n{ablation_list}\n"
        f"{CODING_REQ}"
    )


SUMMARIZE_EVAL_PROMPT = (
    "Provide exactly 1 bullet summarizing this iteration, starting with '- '."
)

EVAL_DESCRIPTION_PROMPT = "### Candidate results\nThe fake evaluator reports a score."
