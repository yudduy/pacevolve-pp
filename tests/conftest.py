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

"""Pytest bootstrap: make the flat ``workflows/`` modules and the ``tasks``
namespace package importable, and merge the test-only fixture tasks into the
same ``tasks.*`` namespace so ``importlib.import_module('tasks.fake_task...')``
resolves during tests without touching the engine.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKFLOWS = os.path.join(_REPO_ROOT, "workflows")
_FIXTURES = os.path.join(_REPO_ROOT, "tests", "fixtures")

# workflows/ must be importable as top-level modules (import llm_utils, ...).
# _REPO_ROOT provides the real `tasks` namespace portion; _FIXTURES provides the
# fake-task portion. Namespace packages merge portions across all sys.path roots.
for _p in (_WORKFLOWS, _REPO_ROOT, _FIXTURES):
    if _p not in sys.path:
        sys.path.insert(0, _p)
