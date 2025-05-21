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

import copy
import json
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal,
    Sequence,
    SupportsFloat,
    Type,
    TypeVar,
    cast,
    get_args,
)

import numpy as np
from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.backend.observable import Callback, Observable
from pulser.backend.operator import Operator, OperatorRepr
from pulser.backend.state import State, StateRepr
from pulser.json.abstract_repr.backend import _deserialize_emulation_config
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr
from pulser.noise_model import NoiseModel

EVAL_TIMES_LITERAL = Literal["Full", "Minimal", "Final"]

StateType = TypeVar("StateType", bound=State)


class BackendConfig:
    """The base backend configuration."""

    _backend_options: dict[str, Any]
    # Whether to warn if unexpected kwargs are received
    _enforce_expected_kwargs: ClassVar[bool] = True

    def __init__(self, **backend_options: Any) -> None:
        """Initializes the backend config."""
        cls_name = self.__class__.__name__
        if self._enforce_expected_kwargs and (
            invalid_kwargs := (
                set(backend_options)
                - (self._expected_kwargs() | {"backend_options"})
            )
        ):
            raise ValueError(
                f"{cls_name!r} received unexpected keyword arguments: "
                f"{invalid_kwargs}; only the following keyword "
                f"arguments are expected: {self._expected_kwargs()}. "
            )
        # Store the abstract repr of the config in _backend_options
        # Prevents potential issues with mutable arguments
        self._backend_options = copy.deepcopy(backend_options)
        if "backend_options" in backend_options:
            with warnings.catch_warnings():
                warnings.filterwarnings("always")
                warnings.warn(
                    f"The 'backend_options' argument of {cls_name!r} "
                    "has been deprecated. Please provide the options "
                    f"as keyword arguments directly to '{cls_name}()'.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self._backend_options.update(backend_options["backend_options"])

    def _expected_kwargs(self) -> set[str]:
        return set()

    def __getattr__(self, name: str) -> Any:
        if (
            # Needed to avoid recursion error
            "_backend_options" in self.__dict__
            and name in self._backend_options
        ):
            return self._backend_options[name]
        raise AttributeError(f"{name!r} has not been passed to {self!r}.")


class EmulationConfig(BackendConfig, Generic[StateType]):
    """Configures an emulation on a backend.

    Args:
        observables: A sequence of observables to compute at specific
            evaluation times. The observables without specified evaluation
            times will use this configuration's 'default_evaluation_times'.
        callbacks: A general callback that is not an observable. Observables
            must be fed into the observables arg, since they all interact
            with the Results, and are subject to additional validation.
            Unlike observables, these are called at every emulation step.
        default_evaluation_times: The default times at which observables
            are computed. Can be a sequence of unique relative times between 0
            (the start of the sequence) and 1 (the end of the sequence), in
            ascending order. Can also be specified as "Full", in which case
            every step in the emulation will also be an evaluation time.
        initial_state: The initial state from which emulation starts. If
            specified, the state type needs to be compatible with the emulator
            backend. If left undefined, defaults to starting with all qudits
            in the ground state.
        with_modulation: Whether to emulate the sequence with the programmed
            input or the expected output.
        interaction_matrix: An optional interaction matrix to replace the
            interaction terms in the Hamiltonian. For an N-qudit system,
            must be an NxN symmetric matrix where entry (i, j) dictates
            the interaction coefficient between qudits i and j, ie it replaces
            the C_n/r_{ij}^n term.
        prefer_device_noise_model: If True, uses the noise model of the
            sequence's device (if the sequence's device has one), regardless
            of the noise model given with this configuration.
        noise_model: An optional noise model to emulate the sequence with.
            Ignored if the sequence's device has default noise model and
            `prefer_device_noise_model=True`.

    Note:
        Additional parameters may be provided. It is up to the emulation
        backend that receives a configuration with extra parameters to assess
        whether it recognizes them and how it will use them. To know all
        parameters expected by an EmulatorBackend, consult its associated
        EmulationConfig subclass found under 'EmulatorBackend.default_config'.

    """

    callbacks: Sequence[Callback]
    observables: Sequence[Observable]
    default_evaluation_times: np.ndarray | Literal["Full"]
    initial_state: StateType | None
    with_modulation: bool
    interaction_matrix: pm.AbstractArray | None
    prefer_device_noise_model: bool
    noise_model: NoiseModel
    # Whether to warn if unexpected kwargs are received
    _enforce_expected_kwargs: ClassVar[bool] = False

    _state_type: ClassVar[Type[State]]
    _operator_type: ClassVar[Type[Operator]]

    def __init__(
        self,
        *,
        callbacks: Sequence[Callback] = (),
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
        obs_tags = []
        if not observables and not callbacks:
            warnings.warn(
                f"{self.__class__.__name__!r} was initialized without any "
                "observables. The corresponding emulation results will be"
                " empty.",
                stacklevel=2,
            )
        for cb in callbacks:
            if isinstance(cb, Observable):
                raise TypeError(
                    "All entries in 'callbacks' must not be instances of "
                    "Observable, since those go in 'observables'."
                )
            if not isinstance(cb, Callback):
                raise TypeError(
                    "All entries in 'callbacks' must be instances of "
                    f"Callback. Instead, got instance of type {type(cb)}."
                )
        for obs in observables:
            if not isinstance(obs, Observable):
                raise TypeError(
                    "All entries in 'observables' must be instances of "
                    f"Observable. Instead, got instance of type {type(obs)}."
                )
            obs_tags.append(obs.tag)
        repeated_tags = [k for k, v in Counter(obs_tags).items() if v > 1]
        if repeated_tags:
            raise ValueError(
                "Some of the provided 'observables' share identical tags. Use "
                "'tag_suffix' when instantiating multiple instances of the "
                "same observable so they can be distinguished. "
                f"Repeated tags found: {repeated_tags}"
            )

        if default_evaluation_times != "Full":
            eval_times_arr = Observable._validate_eval_times(
                list(map(float, default_evaluation_times))
            )
            default_evaluation_times = cast(Sequence[float], eval_times_arr)

        if initial_state is not None and not isinstance(initial_state, State):
            raise TypeError(
                "When defined, 'initial_state' must be an instance of State;"
                f" got object of type {type(initial_state)} instead."
            )

        if interaction_matrix is not None:
            interaction_matrix = pm.AbstractArray(interaction_matrix)
            _shape = interaction_matrix.shape
            if len(_shape) != 2 or _shape[0] != _shape[1]:
                raise ValueError(
                    "'interaction_matrix' must be a square matrix. Instead, "
                    f"an array of shape {_shape} was given."
                )
            if (
                initial_state is not None
                and _shape[0] != initial_state.n_qudits
            ):
                raise ValueError(
                    f"The received interaction matrix of shape {_shape} is "
                    "incompatible with the received initial state of "
                    f"{initial_state.n_qudits} qudits."
                )
            matrix_arr = interaction_matrix.as_array(detach=True)
            if not np.allclose(matrix_arr, matrix_arr.transpose()):
                raise ValueError(
                    "The received interaction matrix is not symmetric."
                )
            if np.any(np.diag(matrix_arr) != 0):
                warnings.warn(
                    "The received interaction matrix has non-zero values in "
                    "its diagonal; keep in mind that these values are "
                    "ignored.",
                    stacklevel=2,
                )

        if not isinstance(noise_model, NoiseModel):
            raise TypeError(
                "'noise_model' must be a NoiseModel instance,"
                f" not {type(noise_model)}."
            )

        super().__init__(
            callbacks=tuple(callbacks),
            observables=tuple(observables),
            default_evaluation_times=default_evaluation_times,
            initial_state=initial_state,
            with_modulation=bool(with_modulation),
            interaction_matrix=interaction_matrix,
            prefer_device_noise_model=bool(prefer_device_noise_model),
            noise_model=noise_model,
            **backend_options,
        )

    def _expected_kwargs(self) -> set[str]:
        return super()._expected_kwargs() | {
            "callbacks",
            "observables",
            "default_evaluation_times",
            "initial_state",
            "with_modulation",
            "interaction_matrix",
            "prefer_device_noise_model",
            "noise_model",
        }

    def is_evaluation_time(self, t: float, tol: float = 1e-6) -> bool:
        """Assesses whether a relative time is an evaluation time."""
        return (
            self.default_evaluation_times == "Full" and 0.0 <= t <= 1.0
        ) or (
            self.is_time_in_evaluation_times(
                t, self.default_evaluation_times, tol=tol
            )
        )

    @staticmethod
    def is_time_in_evaluation_times(
        t: float, evaluation_times: ArrayLike, tol: float = 1e-6
    ) -> bool:
        """Checks if a time is within a collection of evaluation times."""
        return 0.0 <= t <= 1.0 and bool(
            np.any(np.abs(np.array(evaluation_times, dtype=float) - t) <= tol)
        )

    def _to_abstract_repr(self) -> dict[str, Any]:
        return self._backend_options

    def to_abstract_repr(self, skip_validation: bool = False) -> str:
        """Serialize `EmulationConfig` to a JSON formatted str."""
        obj_str = json.dumps(self, cls=AbstractReprEncoder)
        if not skip_validation:
            validate_abstract_repr(obj_str, "config")
        return obj_str

    @classmethod
    def from_abstract_repr(cls, obj_str: str) -> EmulationConfig:
        """Deserialize an EmulationConfig from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the sequence encoded
                in the abstract JSON format.

        Returns:
            EmulationConfig: The EmulationConfig instance.
        """
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized EmulationConfig must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )
        validate_abstract_repr(obj_str, "config")
        return _deserialize_emulation_config(
            json.loads(obj_str),
            cls,
            getattr(cls, "_state_type", StateRepr),
            getattr(cls, "_operator_type", OperatorRepr),
        )


# Legacy class


@dataclass
class EmulatorConfig(BackendConfig):
    """The configuration for emulator backends.

    Warning:
        This class will be deprecated in favor of EmulationConfig once all
        backends migrate to it.

    Args:
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
    initial_state: Literal["all-ground"] | Sequence[complex] | np.ndarray = (
        "all-ground"
    )
    with_modulation: bool = False
    prefer_device_noise_model: bool = False
    noise_model: NoiseModel = field(default_factory=NoiseModel)

    def __post_init__(self) -> None:
        # TODO: Deprecate once QutipBackendV2 is feature complete
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
