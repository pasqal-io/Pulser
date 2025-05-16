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
import numpy as np

from pulser.math.multinomial import multinomial


def test_multinomial_small():
    np.random.seed(123)
    assert np.allclose(
        multinomial(10, [0.3, 0.1, 0.6]),
        np.array([2, 0, 0, 2, 2, 2, 2, 2, 2, 1]),
    )


def test_multinomial_large():
    np.random.seed(123)

    distribution = np.array([0.001] * 100 * 2 + [0.7] + [0.001] * 100 * 1)

    assert abs(1 - sum(distribution)) < 1e-10

    assert np.allclose(
        multinomial(10, distribution),
        np.array([200, 200, 200, 200, 200, 200, 281, 200, 200, 200]),
    )
