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
from typing import (
    Any,
    Generic,
    Literal,
    Sequence,
    SupportsFloat,
    TypeVar,
    cast,
    get_args,
)

import numpy as np
from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.backend.state import State
from pulser.backend.observable import Observable
from pulser.noise_model import NoiseModel

EVAL_TIMES_LITERAL = Literal["Full", "Minimal", "Final"]

StateType = TypeVar("StateType", bound=State)


class BackendConfig:
    """The base backend configuration."""

    _backend_options: dict[str, Any]

    def __init__(self, **backend_options: Any) -> None:
        """Initializes the backend config."""
        self._backend_options = backend_options
        # TODO: Deprecate use of backend_options kwarg
        # TODO: Filter for accepted kwargs

    def __getattr__(self, name: str) -> Any:
        if (
            # Needed to avoid recursion error
            "_backend_options" in self.__dict__
            and name in self._backend_options
        ):
            return self._backend_options[name]
        raise AttributeError  # TODO:


class EmulationConfig(BackendConfig, Generic[StateType]):
    """Configurates an emulation on a backend."""

    # TODO: Complete docstring
    observables: Sequence[Observable]
    default_evaluation_times: np.ndarray | Literal["Full"]
    initial_state: StateType | None
    with_modulation: bool
    interaction_matrix: pm.AbstractArray | None
    prefer_device_noise_model: bool
    noise_model: NoiseModel

    def __init__(
        self,
        *,
        observables: Sequence[Observable] = (),
        # Default evaluation times for observables that don't specify one
        default_evaluation_times: Sequence[SupportsFloat] | Literal["Full"] = (
            1.0,
        ),
        initial_state: StateType | None = None,  # Default is ggg...
        with_modulation: bool = False,
        interaction_matrix: ArrayLike | None = None,
        prefer_device_noise_model: bool = False,
        noise_model: NoiseModel = NoiseModel(),
        **backend_options: Any,
    ) -> None:
        """Initializes the EmulationConfig."""
        for obs in observables:
            if not isinstance(obs, Observable):
                raise TypeError(
                    "All entries in 'observables' must be instances of "
                    f"Observable. Instead, got instance of type {type(obs)}."
                )

        if default_evaluation_times != "Full":
            eval_times_arr = np.array(default_evaluation_times, dtype=float)
            if np.any((eval_times_arr < 0.0) | (eval_times_arr > 1.0)):
                raise ValueError(
                    "All evaluation times must be between 0. and 1."
                )
            default_evaluation_times = cast(Sequence[float], eval_times_arr)

        if initial_state is not None and not isinstance(initial_state, State):
            raise TypeError(
                "When defined, 'initial_state' must be an instance of State;"
                f" got object of type {type(initial_state)} instead."
            )

        # TODO: Validate interaction matrix

        if not isinstance(noise_model, NoiseModel):
            raise TypeError(
                "'noise_model' must be a NoiseModel instance,"
                f" not {type(noise_model)}."
            )

        super().__init__(
            observables=tuple(observables),
            default_evaluation_times=default_evaluation_times,
            initial_state=initial_state,
            with_modulation=bool(with_modulation),
            interaction_matrix=interaction_matrix,
            prefer_device_noise_model=bool(prefer_device_noise_model),
            noise_model=noise_model,
            **backend_options,
        )

    def is_evaluation_time(self, t: float, tol: float = 1e-6) -> bool:
        """Assesses whether a relative time is an evaluation time."""
        return 0.0 <= t <= 1.0 and (
            self.default_evaluation_times == "Full"
            or self.is_time_in_evaluation_times(
                t, self.default_evaluation_times, tol=tol
            )
        )

    @staticmethod
    def is_time_in_evaluation_times(
        t: float, evaluation_times: ArrayLike, tol: float = 1e-6
    ) -> bool:
        """Checks if a time is within a collection of evaluation times."""
        return bool(
            np.any(np.abs(np.array(evaluation_times, dtype=float) - t) <= tol)
        )


# Legacy class


@dataclass
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
        prefer_device_noise_model: If the sequence's device has a default noise
            model, this option signals the backend to prefer it over the noise
            model given with this configuration.
        noise_model: An optional noise model to emulate the sequence with.
            Ignored if the sequence's device has default noise model and
            `prefer_device_noise_model=True`.
    """

    backend_options: dict[str, Any] = field(default_factory=dict)
    sampling_rate: float = 1.0
    evaluation_times: float | Sequence[float] | EVAL_TIMES_LITERAL = "Full"
    initial_state: Literal["all-ground"] | Sequence[complex] = "all-ground"
    with_modulation: bool = False
    prefer_device_noise_model: bool = False
    noise_model: NoiseModel = field(default_factory=NoiseModel)

    def __post_init__(self) -> None:
        # TODO: Raise deprecation warning
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
