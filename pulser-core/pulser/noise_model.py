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
"""Defines a noise model class for emulator backends."""
from __future__ import annotations

import json
import warnings
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Union, cast, get_args

import numpy as np
from numpy.typing import ArrayLike

import pulser.json.abstract_repr as pulser_abstract_repr
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr

__all__ = ["NoiseModel"]

NoiseTypes = Literal[
    "leakage",
    "doppler",
    "amplitude",
    "SPAM",
    "dephasing",
    "relaxation",
    "depolarizing",
    "eff_noise",
]

_NOISE_TYPE_PARAMS: dict[NoiseTypes, tuple[str, ...]] = {
    "leakage": ("with_leakage",),
    "doppler": ("temperature",),
    "amplitude": ("laser_waist", "amp_sigma"),
    "SPAM": ("p_false_pos", "p_false_neg", "state_prep_error"),
    "dephasing": ("dephasing_rate", "hyperfine_dephasing_rate"),
    "relaxation": ("relaxation_rate",),
    "depolarizing": ("depolarizing_rate",),
    "eff_noise": ("eff_noise_rates", "eff_noise_opers"),
}

_PARAM_TO_NOISE_TYPE: dict[str, NoiseTypes] = {
    param: noise_type
    for noise_type, params in _NOISE_TYPE_PARAMS.items()
    for param in params
}

# Parameter characterization

_POSITIVE = {
    "dephasing_rate",
    "hyperfine_dephasing_rate",
    "relaxation_rate",
    "depolarizing_rate",
    "temperature",
}
_STRICT_POSITIVE = {
    "runs",
    "samples_per_run",
    "laser_waist",
}
_PROBABILITY_LIKE = {
    "state_prep_error",
    "p_false_pos",
    "p_false_neg",
    "amp_sigma",
}

_BOOLEAN = {"with_leakage"}

_LEGACY_DEFAULTS = {
    "runs": 15,
    "samples_per_run": 5,
    "state_prep_error": 0.005,
    "p_false_pos": 0.01,
    "p_false_neg": 0.05,
    "temperature": 50.0,
    "laser_waist": 175.0,
    "amp_sigma": 5e-2,
    "relaxation_rate": 0.01,
    "dephasing_rate": 0.05,
    "hyperfine_dephasing_rate": 1e-3,
    "depolarizing_rate": 0.05,
}


@dataclass(init=False, repr=False, frozen=True)
class NoiseModel:
    """Specifies the noise model parameters for emulation.

    Supported noise types:

    - **leakage**: Adds an error state 'x' to the computational
        basis, that can interact with the other states via an
        effective noise channel. Must be defined with an effective
        noise channel, but is incompatible with dephasing and
        depolarizing noise channels.
    - **relaxation**: Noise due to a decay from the Rydberg to
      the ground state (parametrized by ``relaxation_rate``),
      commonly characterized experimentally by the T1 time.
    - **dephasing**: Random phase (Z) flip (parametrized
      by ``dephasing_rate``), commonly characterized
      experimentally by the T2* time.
    - **depolarizing**: Quantum noise where the state is
      turned into the maximally mixed state with rate
      ``depolarizing_rate``. While it does not describe a
      physical phenomenon, it is a commonly used tool to test
      the system under a uniform combination of phase flip (Z) and
      bit flip (X) errors.
    - **eff_noise**: General effective noise channel defined by the
      set of collapse operators ``eff_noise_opers`` and their
      corresponding rates ``eff_noise_rates``.
    - **doppler**: Local atom detuning due to termal motion of the
      atoms and Doppler effect with respect to laser frequency.
      Parametrized by the ``temperature`` field.
    - **amplitude**: Gaussian damping due to finite laser waist and
      laser amplitude fluctuations. Parametrized by ``laser_waist``
      and ``amp_sigma``.
    - **SPAM**: SPAM errors. Parametrized by ``state_prep_error``,
      ``p_false_pos`` and ``p_false_neg``.

    Args:
        runs: When reconstructing the Hamiltonian from random noise is
            necessary, this determines how many times that happens. Not
            to be confused with the number of times the resulting
            bitstring distribution is sampled when calculating bitstring
            counts.
        samples_per_run: Number of samples per noisy Hamiltonian. Useful
            for cutting down on computing time, but unrealistic.
        state_prep_error: The state preparation error probability.
        p_false_pos: Probability of measuring a false positive.
        p_false_neg: Probability of measuring a false negative.
        temperature: Temperature, set in µK, of the atoms in the array.
            Also sets the standard deviation of the speed of the atoms.
        laser_waist: Waist of the gaussian lasers, set in µm, for global
            pulses. Assumed to be the same for all global channels.
        amp_sigma: Dictates the fluctuation in amplitude of a channel from
            run to run as a standard deviation of a normal distribution
            centered in 1. Assumed to be the same for all channels (though
            each channel has its own randomly sampled value in each run).
        relaxation_rate: The rate of relaxation from the Rydberg to the
            ground state (in 1/µs). Corresponds to 1/T1.
        dephasing_rate: The rate of a dephasing occuring (in 1/µs) in a
            Rydberg state superpostion. Only used if a Rydberg state is
            involved. Corresponds to 1/T2*.
        hyperfine_dephasing_rate: The rate of dephasing occuring (in 1/µs)
            between hyperfine ground states. Only used if the hyperfine
            state is involved.
        depolarizing_rate: The rate (in 1/µs) at which a depolarizing
            error occurs.
        eff_noise_rates: The rate associated to each effective noise operator
            (in 1/µs).
        eff_noise_opers: The operators for the effective noise model.
        with_leakage: Whether or not to include an error state in the
            computations (default to False).
    """

    noise_types: tuple[NoiseTypes, ...]
    runs: int | None
    samples_per_run: int | None
    state_prep_error: float
    p_false_pos: float
    p_false_neg: float
    temperature: float
    laser_waist: float | None
    amp_sigma: float
    relaxation_rate: float
    dephasing_rate: float
    hyperfine_dephasing_rate: float
    depolarizing_rate: float
    eff_noise_rates: tuple[float, ...]
    eff_noise_opers: tuple[ArrayLike, ...]
    with_leakage: bool

    def __init__(
        self,
        runs: int | None = None,
        samples_per_run: int | None = None,
        state_prep_error: float | None = None,
        p_false_pos: float | None = None,
        p_false_neg: float | None = None,
        temperature: float | None = None,
        laser_waist: float | None = None,
        amp_sigma: float | None = None,
        relaxation_rate: float | None = None,
        dephasing_rate: float | None = None,
        hyperfine_dephasing_rate: float | None = None,
        depolarizing_rate: float | None = None,
        eff_noise_rates: tuple[float, ...] = (),
        eff_noise_opers: tuple[ArrayLike, ...] = (),
        with_leakage: bool = False,
    ) -> None:
        """Initializes a noise model."""

        def to_tuple(obj: tuple) -> tuple:
            if isinstance(obj, (tuple, list, np.ndarray)):
                obj = tuple(to_tuple(el) for el in obj)
            return obj

        param_vals = dict(
            runs=runs,
            samples_per_run=samples_per_run,
            state_prep_error=state_prep_error,
            p_false_neg=p_false_neg,
            p_false_pos=p_false_pos,
            temperature=temperature,
            laser_waist=laser_waist,
            amp_sigma=amp_sigma,
            relaxation_rate=relaxation_rate,
            dephasing_rate=dephasing_rate,
            hyperfine_dephasing_rate=hyperfine_dephasing_rate,
            depolarizing_rate=depolarizing_rate,
            eff_noise_rates=to_tuple(eff_noise_rates),
            eff_noise_opers=to_tuple(eff_noise_opers),
            with_leakage=with_leakage,
        )
        true_noise_types: set[NoiseTypes] = {
            _PARAM_TO_NOISE_TYPE[p_]
            for p_ in param_vals
            if param_vals[p_] and p_ in _PARAM_TO_NOISE_TYPE
        }
        self._check_leakage_noise(true_noise_types)
        self._check_eff_noise(
            cast(tuple, param_vals["eff_noise_rates"]),
            cast(tuple, param_vals["eff_noise_opers"]),
            "eff_noise" in true_noise_types,
            with_leakage=cast(bool, param_vals["with_leakage"]),
        )

        # Get rid of unnecessary None's
        for p_ in _POSITIVE | _PROBABILITY_LIKE:
            param_vals[p_] = param_vals[p_] or 0.0

        relevant_params = self._find_relevant_params(
            true_noise_types,
            cast(float, param_vals["state_prep_error"]),
            cast(float, param_vals["amp_sigma"]),
            cast(Union[float, None], param_vals["laser_waist"]),
        )

        relevant_param_vals = {
            p: param_vals[p]
            for p in param_vals
            if param_vals[p] is not None or p in relevant_params
        }
        self._validate_parameters(relevant_param_vals)

        object.__setattr__(
            self, "noise_types", tuple(sorted(true_noise_types))
        )
        non_zero_relevant_params = [
            p for p in relevant_params if param_vals[p]
        ]
        for param_, val_ in param_vals.items():
            object.__setattr__(self, param_, val_)
            if val_ and param_ not in relevant_params:
                warnings.warn(
                    f"{param_!r} is not used by any active noise type "
                    f"in {self.noise_types} when the only defined parameters "
                    f"are {non_zero_relevant_params}.",
                    stacklevel=2,
                )

    @staticmethod
    def _find_relevant_params(
        noise_types: Collection[NoiseTypes],
        state_prep_error: float,
        amp_sigma: float,
        laser_waist: float | None,
    ) -> set[str]:
        relevant_params: set[str] = set()
        for nt_ in noise_types:
            relevant_params.update(_NOISE_TYPE_PARAMS[nt_])
            if (
                nt_ == "doppler"
                or (nt_ == "amplitude" and amp_sigma != 0.0)
                or (nt_ == "SPAM" and state_prep_error != 0.0)
            ):
                relevant_params.update(("runs", "samples_per_run"))
        # Disregard laser_waist when not defined
        if laser_waist is None:
            relevant_params.discard("laser_waist")
        return relevant_params

    @staticmethod
    def _check_leakage_noise(noise_types: Collection[NoiseTypes]) -> None:
        # Can't define "dephasing", "depolarizing" with "leakage"
        if "leakage" not in noise_types:
            return
        if "eff_noise" not in noise_types:
            raise ValueError(
                "At least one effective noise operator must be defined to"
                " simulate leakage."
            )

    @staticmethod
    def _check_noise_types(noise_types: Sequence[NoiseTypes]) -> None:
        for noise_type in noise_types:
            if noise_type not in get_args(NoiseTypes):
                raise ValueError(
                    f"'{noise_type}' is not a valid noise type. "
                    + "Valid noise types: "
                    + ", ".join(get_args(NoiseTypes))
                )

    @staticmethod
    def _check_eff_noise(
        eff_noise_rates: Sequence[float],
        eff_noise_opers: Sequence[ArrayLike],
        check_contents: bool,
        with_leakage: bool,
    ) -> None:
        if len(eff_noise_opers) != len(eff_noise_rates):
            raise ValueError(
                f"The operators list length({len(eff_noise_opers)}) "
                "and rates list length"
                f"({len(eff_noise_rates)}) must be equal."
            )
        for rate in eff_noise_rates:
            if not isinstance(rate, float):
                raise TypeError(
                    "eff_noise_rates is a list of floats,"
                    f" it must not contain a {type(rate)}."
                )

        if not check_contents:
            return

        if not eff_noise_opers or not eff_noise_rates:
            raise ValueError(
                "The effective noise parameters have not been filled."
            )

        if np.any(np.array(eff_noise_rates) < 0):
            raise ValueError("The provided rates must be greater than 0.")

        # Check the validity of operators
        min_shape = 2 if not with_leakage else 3
        possible_shapes = [
            (min_shape, min_shape),
            (min_shape + 1, min_shape + 1),
        ]
        for op in eff_noise_opers:
            # type checking
            try:
                operator = np.array(op, dtype=complex)
            except TypeError as e1:
                try:
                    operator = np.array(
                        op.to("Dense").data_as("ndarray"),  # type: ignore
                        dtype=complex,
                    )
                except AttributeError:
                    raise TypeError(
                        f"Operator {op!r} is not castable to a Numpy array."
                    ) from e1
            if operator.ndim != 2:
                raise ValueError(f"Operator '{op!r}' is not a 2D array.")

            if operator.shape not in possible_shapes:
                raise ValueError(
                    f"With{'' if with_leakage else 'out'} leakage, operator's "
                    f"shape must be {possible_shapes[0]}, "
                    f"not {operator.shape}."
                )

    @staticmethod
    def _validate_parameters(param_vals: dict[str, Any]) -> None:
        for param in param_vals:
            is_valid = True
            value = param_vals[param]
            if param in _POSITIVE:
                is_valid = value >= 0
                comp = "greater than or equal to zero"
            elif param in _STRICT_POSITIVE:
                is_valid = value is not None and value > 0
                comp = "greater than zero"
            elif param in _PROBABILITY_LIKE:
                is_valid = 0 <= value <= 1
                comp = (
                    "greater than or equal to zero and smaller than "
                    "or equal to one"
                )
            elif param in _BOOLEAN:
                is_valid = isinstance(value, bool)
                comp = "a boolean"
            if not is_valid:
                raise ValueError(f"'{param}' must be {comp}, not {value}.")

    def _to_abstract_repr(self) -> dict[str, Any]:
        all_fields = asdict(self)
        all_fields.pop("with_leakage")
        eff_noise_rates = all_fields.pop("eff_noise_rates")
        eff_noise_opers = all_fields.pop("eff_noise_opers")
        all_fields["eff_noise"] = list(zip(eff_noise_rates, eff_noise_opers))
        return all_fields

    def __repr__(self) -> str:
        relevant_params = self._find_relevant_params(
            self.noise_types,
            self.state_prep_error,
            self.amp_sigma,
            self.laser_waist,
        )
        relevant_params.add("noise_types")
        params_list = []
        for f in fields(self):
            if f.name in relevant_params:
                params_list.append(f"{f.name}={getattr(self, f.name)!r}")
        return f"{self.__class__.__name__}({', '.join(params_list)})"

    def to_abstract_repr(self) -> str:
        """Serializes the noise model into an abstract JSON object."""
        abstr_str = json.dumps(self, cls=AbstractReprEncoder)
        validate_abstract_repr(abstr_str, "noise")
        return abstr_str

    @staticmethod
    def from_abstract_repr(obj_str: str) -> NoiseModel:
        """Deserialize a noise model from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the noise model
                encoded in the abstract JSON format.
        """
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized noise model must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )

        # Avoids circular imports
        return (
            pulser_abstract_repr.deserializer.deserialize_abstract_noise_model(
                obj_str
            )
        )
