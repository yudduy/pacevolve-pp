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

"""run_experiment.load_configs is imported and called by run_advisor_rl, so its
error path must fail cleanly (SystemExit) rather than NameError on an unbound
module-level logger."""

import textwrap

import pytest

import run_experiment


def test_load_configs_bad_task_exits_cleanly(tmp_path):
    cfg = tmp_path / "config_1.yaml"
    cfg.write_text(textwrap.dedent("""
        llm: {name: fake}
        experiment: {task_id: nonexistent_task_xyz}
        paths: {src_path: /tmp, target_file_path: x.py}
        evaluation: {eval_configs: [{dataset: d}]}
    """))
    # The task's eval_utils cannot import -> load_configs logs and sys.exit(1).
    with pytest.raises(SystemExit):
        run_experiment.load_configs(str(cfg))
