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
# RegexTagCustomPruningAlgorithmEnd
