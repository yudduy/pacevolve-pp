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

RJCH = '''
    def _add_one_object_algo(self, object_id):
        """
        Places a single object onto a server using the naive consistent
        hashing approach (linear scan on the serversArray).
        """

        history_list = self.objectsHistory.setdefault(object_id, [])

        # Hash object to a starting position in serversArray
        obj_hash_input = f"object_{object_id}"
        start_index = mmh3.hash(obj_hash_input, seed=object_id) % self.serversArrayLen

        found_slot = False
        # Search circularly from the start_index for the next slot with a server
        for k in range(self.serversArrayLen):
            current_index = (start_index + k) % self.serversArrayLen
            history_list.append(current_index)

            server_id = self.serversArray[current_index]

            if server_id != -1:  # Found a server
                if not self.fullFlag[server_id]:  # Check if the server is not full
                    # Assign object to this server
                    self.serversContains[server_id].add(object_id)
                    self.servers[server_id] += 1
                    if self.servers[server_id] == self.loadCap:
                        self.fullFlag[server_id] = True
                        self.totalFull += 1
                    self.objectsCurrent[object_id] = current_index
                    found_slot = True
                    break  # Object placed
                # else: server is full, continue scanning for the next server
        # End of linear scan loop

        if not found_slot:
             # This would mean all servers are full or no server slots were found,
             raise RuntimeError(f"Failed to place object {object_id}: No available server found in serversArray.")
'''
CODING_REQ = """
While completing your task, you MUST:
- Enclose your code in triple backticks to properly format the code in Markdown.
- You MUST indent all functions one block like in the example since these methods are going to be embedded in a well-defined Python class. 
- You MUST NOT import any libraries at the beginning of the code due to indentation issue, you code block is embedded in a Python class. If you need to import Python native libraries, do it inside the function. We have already imported mmh3, random, time, and `import numpy as np` for you so you don't need to import those.
- You MUST NOT change the function signature (arguments, return value, and name). The system expects to call a function with signature `_add_one_object_algo(self, object_id)`.
- Your solution should contain the function defined above and the function ONLY. DO NOT import packages globally, DO NOT wrap those functions in a class, DO NOT define a class, DO NOT define an __init__ method.
- You DO NOT need to escape special characters, such as newlines, with a double backslash when writing code. The system understands the traditional "backslash-n" character style.
- If you want to introduce additional, tunable hyperparameters, you will need to hard-code them and run multiple experiments (one for each parameter configuration).
- Each triple-backtick enclosed code block in your output must contain valid Python and be a valid implementation.
- Be aware of the performance implication of your code changes, do NOT introduce changes that will cause the code to run much slower.
"""

RJCH_DOCS = f"""
### Codebase documentation
Your code will be run as a modification of the consistent hashing algorithm.

#### Writing code

{CODING_REQ}
"""

TASK_INTRO = """
You are an expert researcher who specializes in algorithm development and optimization. We are studying consistent hashing, an important problem for many applications. Our goal is to optimize the hashing algorithm.
"""


BACKGROUND = """
### Background on Consistent Hashing
Consistent hashing is a technique used in distributed systems to decide where to store or route data (e.g., cache entries, user sessions, work items) across a cluster of servers. The primary goal is to minimize the impact of server additions or removals on the overall system. In a large-scale system, servers can fail, be taken down for maintenance, or new servers can be added to increase capacity. Consistent hashing ensures that such changes only require a minimal amount of data or request remapping.

Traditional hashing methods (like hash(key) % N, where N is the number of servers) can cause massive disruption when N changes. For instance, if a server is added or removed, nearly all keys might be reassigned to different servers, leading to significant data movement, cache misses, or session invalidations. Consistent hashing is designed to avoid this widespread reshuffling.

### How Consistent Hashing Works
The core idea is to map both the servers and the keys onto an abstract range, often visualized as a ring. Here's a common conceptual model:

Hashing to the Ring: A hash function is used to map both server identifiers (e.g., IP addresses, hostnames) and item keys to points on the ring. The range of the hash function typically wraps around (e.g., 0 to 2^32 - 1).
Server Placement: Each server is placed at one or more positions on the ring based on the hash of its identifier. Using multiple positions per server (virtual nodes) helps improve load distribution.
Key Mapping: To determine which server is responsible for a given key, the key is also hashed onto the ring. The system then walks clockwise (or counter-clockwise, as long as it's consistent) along the ring from the key's position until a server's position is encountered. That server is deemed responsible for the key.
When a server is added, it takes over a portion of the keys from the server immediately following it on the ring. When a server is removed, its keys are distributed among the server that follows it. Crucially, keys mapped to other servers are not affected.

### Benefits and Properties
Minimized Disruption: When a server is added or removed, only a fraction of the keys (roughly 1/N, where N is the number of servers) need to be remapped. This is the primary advantage over simple modulo hashing.
Load Balancing: While not perfectly uniform, consistent hashing generally distributes keys relatively evenly across the available servers, especially when virtual nodes are used.
Scalability: New servers can be added to the system, and they will automatically start handling their share of the keys, improving capacity.
Fault Tolerance: The impact of a single server failure is isolated to the keys it was handling.

Consistent Hashing Evaluation
The effectiveness of a consistent hashing scheme is evaluated based on several factors:
Balance: How evenly is the load (number of keys, requests) distributed across the servers? Uneven distribution can lead to hotspots. The use of virtual nodes is a common technique to improve balance.
Efficiency: How long does it take to add and remove an object from a bin?
"""



# - Instead of Linear probing, using a second hash to jump to a new bin can provide better load distribution and faster object insertion, compared to sequential probing.
# - When implementing probing strategies, if you encounter high latency, consider using bitwise operations, such has bit shift (>>), XOR (^), and (&) to speed up the implementation.

def construct_mutation_prompt(sota_algorithm, ablation_list):
  ablation_descriptions = "\n".join(ablation_list)
  prompt = f"""
We are conducting an evolutionary optimization process for the load balancing problem in consistent hashing.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

# Knowledge base
- Cascaded overflow is a liability in practice because overloaded servers often fail and pass their loads to the nearest clockwise server. Cascaded overflow can trigger an avalanche of server failures as an enormous load bounces around the circle, crashing servers wherever it goes.
- Linear probing with random probes fails because it effectively rearranges the unit circle. Bin i always overflows into h(i), preserving the cascaded overflow effect.- Results showed increased cost on all dataset. Deterministic selection likely chooses too many local connections and does not allow enough graph exploration.
- Using Murmurhash is much faster than picking an index uniformly at random from the entire serversArray for each object, however, this usually leads to higher percentage of bins full. Think about how to get the best of both worlds.
- When implementing probing strategies, if you encounter high latency, consider using bitwise operations, such has bit shift (>>), XOR (^), and (&) to speed up the implementation.
{ablation_descriptions}

## Your Task

You will make small, reasonable changes to this algorithm to reduce bins full percentage, server load variance, and time to add an object (We provide you with performance metrics targets, note that they are not necessarily the performance of the current SoTA). The experiment strategy is to improve the current state-of-the-art algorithm by evaluating dozens or hundreds of candidates with small perturbations. Try to strike a balance between safe, easy changes that are very likely to improve performance ("exploit-heavy" candidates) and more exploratory changes that help us understand the space of possible algorithms ("explore-heavy" candidates).

You must consider the results of past experiments when designing your candidate. For example, if the notes show poor performance from a strategy, does it make sense to try the opposite? Is there a hyperparameter that needs to be tuned properly? Is there any additional information that you can compute or obtain? These are just suggestions - feel free to come up with additional directions to explore.

Your task is to analyze the current state-of-the-art algorithm, construct an algorithm candidate by editing the current state-of-the-art method, and write the final Python output code for the candidate.

{CODING_REQ}

Please follow these steps:

1. Explanation of the current state-of-the-art.
2. Brainstorm several possible ideas. Try to be creative while also considering the results of past experiments. Provide a reasoning for each idea.
3. Think through which idea is the most promising one to implement. Explain your reasoning, select the best idea, and describe your proposed modification.
4. Code implementation of the candidate.
  """
  return prompt

SUMMARIZE_EVAL_PROMPT = """
## Your Task

Your task is to provide a final concise summary of this entire experiment iteration. This summary will be added to our knowledge base and used to inform future experiments. First, summarize the key findings in a short paragraph. This paragraph will not be used in the knowledge base: it just serves to help you organize your thoughts. Then, provide **exactly 2 bullet points** summarizing the key findings and your final lesson. Each bullet MUST start on a new line and begin with a hyphen (-). Keep your bullets SHORT - they do not need to be complete sentences; they just need to be clear and detailed. DO NOT include obviously true or trivial statements, such as "per-dataset hyperparameter tuning is important." Instead, focus on the key findings that will inform future experiments. 

Here is an example of a good summary:
- Candidate introduced random sampling of neighbors, randomly selecting 2*M nodes before applying the existing pruning logic.
- Results show increased cost on all datasets (X% at Y% recall on sift1m). Sampling likely removes too many good neighbors.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents the cost of the candidate relative to the cost of the baseline. A positive value (X%) means that the candidate is X% more expensive and a negative value (< 0%) means that the candidate is BETTER than the baseline.

Remember that algorithms are evaluated by percentage of bins full and server load variance. The objective of our experiments is to reduce bins full percentage and server load variance.

### Understanding metrics
Variance in server loads: Reducing server load variance by more than 10% is considered good, and by more than 20% is considered very significant.
Percentage of bins full: Reducing pctOfFullBins by more than 5% is considered good, and by more than 10% is considered very significant. Note in some cases pctOfFullBins of baseline is <5%, in those cases, this metric is less significant.
Time to add an object: You should aim to be as fast as the target, and should avoid any latency greater than or equal to 0.0015, which likely suggests your algorithm is either too complex or poorly implemented.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
Would you like to tune any hyperparameters?

Note that you MUST NOT write dataset-specific code (e.g., choose hyperparameters based on the dataset name). We will consider the best hyperparameter configuration on a per-dataset basis when evaluating algorithms.

Note that you CANNOT tune servers, duplicates, objects, and epsilon, those are set during the eval phase.

If yes, explain your reasoning and respond with ONE candidate that you would like to try. Do not write any code yet - we will write the implementation in the next step. If you do not want to tune any hyperparameters, simply respond "No."
"""

HPARAM_IMPLEMENT_PROMPT = f"""
### Hyperparameter implementation
Please write the implementation of your hyperparameter candidate. Respond with a markdown-formatted code block that implements your improved algorithm.

{CODING_REQ}
"""

UPDATE_BASELINE_PROMPT = f"""
Should we update the baseline algorithm? Please answer yes or no then explain your reasoning. If the answer is yes, respond with a code block containing the candidate that we should use as the new baseline algorithm - this will most likely be the candidate that performed the best overall in hyperparameter ablations. If no, simply respond "No."

{CODING_REQ}
"""
