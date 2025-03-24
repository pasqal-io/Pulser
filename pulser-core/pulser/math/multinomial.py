# Copyright 2025 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility function for sampling."""
import numpy as np


def multinomial(n_samples: int, probabilities: np.ndarray) -> np.ndarray:
    """Multinomial samples from the distribution given by `probabilities`.

    Unlike `np.random.multinomial`, this doesn't assert that the probabilities
    sum to 1, and returns the indices of the samples instead of
    aggregated counts as a large array.

    Args:
        n_samples: Number of samples to return.
        probabilities: Probability distribution. Must sum to 1.

    Returns:
        Indices of samples with replacement.
    """
    rnd = np.random.rand(n_samples)

    cumsums = np.cumsum(probabilities)

    return np.searchsorted(cumsums, rnd)
