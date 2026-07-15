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

import copy
import logging

import idea_select_utils
import llm_utils
import workflow_utils


logger = logging.getLogger("controller")


def make_role_config(config: dict, role: str) -> dict:
    new = copy.deepcopy(config)
    role_llm = config.get(f"{role}_llm")
    if role_llm:
        new["llm"] = {**config["llm"], **role_llm}
    return new


def select_idea_no_code(
    advisor,
    transcript,
    prompts_module,
    sota_algo,
    idea_repo,
    config: dict,
    max_attempts: int = 3,
):
    prompt = None
    for attempt in range(max_attempts):
        prompt_tag = f"idea_select_no_code_{attempt}"
        response_tag = f"idea_select_response_{attempt}"
        prompt = prompts_module.construct_idea_select_no_code_prompt(
            sota_algo, idea_repo
        )
        transcript.append(
            llm_utils.ContentChunk(prompt, "user", tags=[prompt_tag])
        )
        response = llm_utils.generate_completion(advisor, transcript, config)
        transcript.append(
            llm_utils.ContentChunk(response, "model", tags=[response_tag])
        )

        if response is None:
            transcript.hide_by_tag([prompt_tag, response_tag])
            continue

        idea_id = idea_select_utils.extract_idea_id(response)
        _, exp_description = idea_select_utils.parse_selected_idea(response)
        if idea_id is None or exp_description is None:
            transcript.hide_by_tag([prompt_tag, response_tag])
            continue

        return idea_id, exp_description, response, prompt

    return None, None, None, prompt


def implement_idea(
    implementer,
    transcript,
    prompts_module,
    sota_algo,
    idea_id,
    exp_description,
    compile_config,
    eval_configs,
    config,
    candidate_id,
    baseline_id,
):
    prompt = prompts_module.construct_code_impl_prompt(
        sota_algo, idea_id, exp_description
    )
    transcript.append(
        llm_utils.ContentChunk(prompt, "user", tags=["code_impl_prompt"])
    )
    response = llm_utils.generate_completion(implementer, transcript, config)
    transcript.append(
        llm_utils.ContentChunk(
            response, "model", tags=["code_impl_response"]
        )
    )

    trial = workflow_utils.AlgorithmTrial()
    trial = workflow_utils.edit_until_compile(
        implementer,
        trial,
        transcript,
        compile_config,
        config,
        loop_config=config["workflow_loops"]["initial_compile"],
        use_idea_repo=False,
    )
    trial.idea_id = idea_id
    if not trial.compile_success:
        return trial

    trial = workflow_utils.edit_until_successful_eval(
        implementer,
        trial,
        transcript,
        compile_config,
        eval_configs,
        config,
        candidate_id,
        baseline_id,
        loop_config=config["workflow_loops"]["initial_eval"],
    )
    trial.idea_id = idea_id
    return trial
