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
from typing import Any, Literal, Sequence, get_args

import numpy as np

from pulser.backend.noise_model import NoiseModel

EVAL_TIMES_LITERAL = Literal["Full", "Minimal", "Final"]


@dataclass(frozen=True)
class BackendConfig:
    """The base backend configuration.

    Attributes:
        backend_options: A dictionary of backend specific options.
    """

    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmulatorConfig(BackendConfig):
    """The configuration for emulator backends.

    Attributes:
        backend_options: A dictionary of backend-specific options.
        sampling_rate: The fraction of samples to extract from the pulse
            sequence for emulation.
        evaluation_times: The times at which results are returned. Choose
            between:

            - "Full": The times are set to be the ones used to define the
              Hamiltonian to the solver.

            - "Minimal": The times are set to only include initial and final
              times.

            - "Final": Returns only the result at the end of the sequence.

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
    evaluation_times: float | Sequence[float] | EVAL_TIMES_LITERAL = "Full"
    initial_state: Literal["all-ground"] | Sequence[complex] = "all-ground"
    with_modulation: bool = False
    noise_model: NoiseModel = field(default_factory=NoiseModel)

    def __post_init__(self) -> None:
        if not (0 < self.sampling_rate <= 1.0):
            raise ValueError(
                "The sampling rate (`sampling_rate` = "
                f"{self.sampling_rate}) must be greater than 0 and "
                "less than or equal to 1."
            )

        if isinstance(self.evaluation_times, str):
            if self.evaluation_times not in get_args(EVAL_TIMES_LITERAL):
                raise ValueError(
                    "If provided as a string, 'evaluation_times' must be one "
                    f"of the following options: {get_args(EVAL_TIMES_LITERAL)}"
                )
        elif isinstance(self.evaluation_times, float):
            if not (0 < self.evaluation_times <= 1.0):
                raise ValueError(
                    "If provided as a float, 'evaluation_times' must be"
                    " greater than 0 and less than or equal to 1."
                )
        elif isinstance(self.evaluation_times, (list, tuple, np.ndarray)):
            if np.min(self.evaluation_times, initial=0) < 0:
                raise ValueError(
                    "If provided as a sequence of values, "
                    "'evaluation_times' must not contain negative values."
                )
        else:
            raise TypeError(
                f"'{type(self.evaluation_times)}' is not a valid"
                " type for 'evaluation_times'."
            )

        if isinstance(self.initial_state, str):
            if self.initial_state != "all-ground":
                raise ValueError(
                    "If provided as a string, 'initial_state' must be"
                    " 'all-ground'."
                )
        elif not isinstance(self.initial_state, (tuple, list, np.ndarray)):
            raise TypeError(
                f"'{type(self.initial_state)}' is not a valid type for"
                " 'initial_state'."
            )

        if not isinstance(self.noise_model, NoiseModel):
            raise TypeError(
                "'noise_model' must be a NoiseModel instance,"
                f" not {type(self.noise_model)}."
            )
