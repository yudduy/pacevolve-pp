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
# RegexTagCustomPruningAlgorithmEnd
