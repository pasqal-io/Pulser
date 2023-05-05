# Copyright 2023 Pulser Development Team
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
"""Defines the backend configuration classes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy


@dataclass
class BackendConfig:
    """The base backend configuration.

    Attributes:
        backend_options: A dictionary of backend specific options.
    """

    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmulatorConfig(BackendConfig):
    """The configuration for emulator backends.

    Attributes:
        backend_options: A dictionary of backend-specific options.
        evaluation_times: The times at which results are returned. Choose
            between:

            - "Full": The times are set to be the ones used to define the
              Hamiltonian to the solver.

            - "Minimal": The times are set to only include initial and final
              times.

            - A list of times in µs if you wish to only include those specific
              times.

            - A float to act as a sampling rate for the resulting state.
        initial_state: The initial state from which emulation starts.
            Choose between:

            - "all-ground" for all atoms in the ground state
            - An array of floats with a shape compatible with the system
        with_modulation: Whether to emulate the sequence with the programmed
            input or the expected output.
        noise_model: An optional noise model to emulate the sequence with.
    """

    sampling_rate: float = 1.0
    evaluation_times: float | list[float] | Literal[
        "Full", "Minimal"
    ] = "Minimal"
    initial_state: Literal["all-ground"] | numpy.ndarray = "all-ground"
    with_modulation: bool = False
    noise_model: Any = None  # TODO: Define NoiseModel class
