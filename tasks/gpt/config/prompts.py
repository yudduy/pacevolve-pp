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

GPT = """
Please Paste the GPT training code here. Sample training code can be obtained from https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py
"""

RJCH_DOCS = f"""
### Codebase documentation
Your code will be used to train a language model.

"""

TASK_INTRO = """
You are an expert researcher who specializes in LLM. We are studying how to improve the training efficiency of a GPT-2 scale LLM. Our goal is to improve the Pareto frontier of validation loss and training time. You will be given a fixed corpus and stepsize to train the model on.
"""


BACKGROUND = """
***

### Background on GPT-2

The Generative Pre-trained Transformer 2 (GPT-2) is a large language model (LLM) designed to generate coherent and human-like text. Its primary task is **next-word prediction**. Given a sequence of words, the model learns to predict the most probable next word. This simple objective, when trained on a massive and diverse corpus of text from the internet, enables the model to learn grammar, facts, reasoning abilities, and different styles of writing.
***

***

### Benefits and Properties

* **Unsupervised Learning**: GPT-2 can be trained on vast amounts of raw text without needing explicit human-made labels, making it possible to leverage internet-scale datasets.
* **Contextual Understanding**: The self-attention mechanism is exceptionally effective at capturing long-range dependencies, allowing the model to maintain context over long passages of text.
* **Scalability**: The Transformer architecture scales remarkably well. Increasing the model size (more layers, larger embeddings) and training data consistently leads to better performance and the emergence of new capabilities not explicitly trained for (e.g., translation, summarization).

***

### GPT-2 Evaluation
How well can a GPT-2 model learn a language? You will be evaluated based on its performance during training and its efficiency, using standard industry metrics.

* **Validation Loss**: This is the model's Cross-Entropy Loss calculated on a held-out "validation" dataset that it does not see during training. This metric is the primary indicator of how well the model is **generalizing** to new text, rather than just memorizing the training data. A lower validation loss is better.
* **Training Time**: The total wall-clock time required to train the model for a specified number of steps or epochs. This metric is crucial for measuring computational efficiency and the overall cost of developing the model. A shorter training time is better.
"""

def construct_mutation_prompt(sota_algorithm, ablation_list):
  ablation_descriptions = "\n".join(ablation_list)
  prompt = f"""
We are conducting an evolutionary optimization process for training a language model.

{BACKGROUND}

{RJCH_DOCS}

### Current state-of-the-art
The current state-of-the-art algorithm is as follows:
```python
{sota_algorithm}
```

# Knowledge base
- Tuned hyperparameters like the learning rate, batch size, Adam epsilon, and logit softcap improved performance.
- Muon optimizer improved validation loss.
- Architectural changes including untied embeddings and output heads, padded embeddings, ReLU² activations, zero-initialized projections, and QK-normalization improved performance.
- Various skip connections, including U-net patterns for values and embeddings improved performance.
- Adding "Value Embeddings," eventually splitting and sparsifying them improved performance.
- Replaced dense causal attention with FlexAttention to dramatically increase context length and added techniques like window warmup and block sliding windows.
- Merged the QKV (Query, Key, Value) weights and implemented a long-short attention pattern to improve performance.
- Optimized distributed training by improving gradient communication with techniques like reduce_scatter and overlapping computation with communication.
- Bfloat16 for activations improved performance.
- Adjusted the learning rate schedule with momentum warmup and a modified decay cooldown improved performance.
- Aligned training batch starts with End-of-Sequence (EoS) tokens for better data alignment improved performance.

{ablation_descriptions}

## Your Task

You will make small, reasonable changes to improve the performance of language model training (We provide you with performance metrics targets, note that they are not necessarily the performance of the current SoTA algorithms). The experiment strategy is to improve the current state-of-the-art algorithm by evaluating dozens or hundreds of candidates with small perturbations. Try to strike a balance between safe, easy changes that are very likely to improve performance ("exploit-heavy" candidates) and more exploratory changes that help us understand the space of possible algorithms ("explore-heavy" candidates).

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
Also, be concrete about which improvements to which current techniques lead to which results. Do not use vague terms such as SoTA since SoTA is constantly evolving and future audience may not know what you are referring to. Your statements must be self-contained. 
In the second bullet point, include the best result from the current trial.

Here is an example of a good summary:
- Candidate introduced random sampling of neighbors, randomly selecting 2*M nodes before applying the existing pruning logic.
- Results: Validation loss: xxx Training time: xxx. Results show improvements with crossing X and Y. They are likely correlated and have higher order connetions.
"""


EVAL_DESCRIPTION_PROMPT = """
### Candidate results
The table presents the validation loss and training time.

### Understanding metrics
A lower validation loss is better. A lower training time is better.
"""

HPARAM_PROMPT = """
## Hyperparameter tuning
Would you like to tune any hyperparameters?

Note that you MUST NOT write dataset-specific code (e.g., choose hyperparameters based on the dataset name). We will consider the best hyperparameter configuration on a per-dataset basis when evaluating algorithms.

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
