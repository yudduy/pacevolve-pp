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

# RegexTagCustomPruningAlgorithmStart
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
# RegexTagCustomPruningAlgorithmEnd
