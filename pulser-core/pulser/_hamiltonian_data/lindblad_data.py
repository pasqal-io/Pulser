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
"""Definition of a set of Lindblad collapse operators."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LindbladData:
    """Some data about the Lindblad operators used by the simulation."""

    op_matrix_names: list[str]
    local_collapse_ops: list[tuple[int | float | complex, str | np.ndarray]]
    depolarizing_pauli_2ds: dict[str, list[tuple[int | complex, str]]]
