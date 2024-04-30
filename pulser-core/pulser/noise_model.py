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
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

import pulser.json.abstract_repr as pulser_abstract_repr
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.abstract_repr.validation import validate_abstract_repr

__all__ = ["NoiseModel"]

NOISE_TYPES = Literal[
    "doppler",
    "amplitude",
    "SPAM",
    "dephasing",
    "relaxation",
    "depolarizing",
    "eff_noise",
]


@dataclass(frozen=True)
class NoiseModel:
    """Specifies the noise model parameters for emulation.

    Select the desired noise types in `noise_types` and, if necessary,
    modifiy the default values of related parameters.
    Non-specified parameters will have reasonable default values which
    are only taken into account when the related noise type is selected.

    Args:
        noise_types: Noise types to include in the emulation.
            Available options:

            - "relaxation": Noise due to a decay from the Rydberg to
              the ground state (parametrized by `relaxation_rate`), commonly
              characterized experimentally by the T1 time.

            - "dephasing": Random phase (Z) flip (parametrized
              by `dephasing_rate`), commonly characterized experimentally
              by the T2* time.

            - "depolarizing": Quantum noise where the state is
              turned into the maximally mixed state with rate
              `depolarizing_rate`. While it does not describe a physical
              phenomenon, it is a commonly used tool to test the system
              under a uniform combination of phase flip (Z) and
              bit flip (X) errors.

            - "eff_noise": General effective noise channel defined by
              the set of collapse operators `eff_noise_opers`
              and the corresponding rates distribution
              `eff_noise_rates`.

            - "doppler": Local atom detuning due to termal motion of the
              atoms and Doppler effect with respect to laser frequency.
              Parametrized by the `temperature` field.

            - "amplitude": Gaussian damping due to finite laser waist and
              laser amplitude fluctuations. Parametrized by `laser_waist`
              and `amp_sigma`.

            - "SPAM": SPAM errors. Parametrized by
              `state_prep_error`, `p_false_pos` and `p_false_neg`.

        runs: Number of runs needed (each run draws a new random noise).
        samples_per_run: Number of samples per noisy run. Useful for
            cutting down on computing time, but unrealistic.
        state_prep_error: The state preparation error probability.
        p_false_pos: Probability of measuring a false positive.
        p_false_neg: Probability of measuring a false negative.
        temperature: Temperature, set in µK, of the atoms in the array.
            Also sets the standard deviation of the speed of the atoms.
        laser_waist: Waist of the gaussian laser, set in µm, for global
            pulses.
        amp_sigma: Dictates the fluctuations in amplitude as a standard
            deviation of a normal distribution centered in 1.
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
    """

    noise_types: tuple[NOISE_TYPES, ...] = ()
    runs: int = 15
    samples_per_run: int = 5
    state_prep_error: float = 0.005
    p_false_pos: float = 0.01
    p_false_neg: float = 0.05
    temperature: float = 50.0
    laser_waist: float = 175.0
    amp_sigma: float = 5e-2
    relaxation_rate: float = 0.01
    dephasing_rate: float = 0.05
    hyperfine_dephasing_rate: float = 1e-3
    depolarizing_rate: float = 0.05
    eff_noise_rates: tuple[float, ...] = field(default_factory=tuple)
    eff_noise_opers: tuple[ArrayLike, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        positive = {
            "dephasing_rate",
            "hyperfine_dephasing_rate",
            "relaxation_rate",
            "depolarizing_rate",
        }
        strict_positive = {
            "runs",
            "samples_per_run",
            "temperature",
            "laser_waist",
        }
        probability_like = {
            "state_prep_error",
            "p_false_pos",
            "p_false_neg",
            "amp_sigma",
        }
        # The two share no common terms
        assert not strict_positive.intersection(probability_like)

        for f in fields(self):
            is_valid = True
            param = f.name
            value = getattr(self, param)
            if param in positive:
                is_valid = value is None or value >= 0
                comp = "None or greater than or equal to zero"
            if param in strict_positive:
                is_valid = value > 0
                comp = "greater than zero"
            elif param in probability_like:
                is_valid = 0 <= value <= 1
                comp = (
                    "greater than or equal to zero and smaller than "
                    "or equal to one"
                )
            if not is_valid:
                raise ValueError(f"'{param}' must be {comp}, not {value}.")

        def to_tuple(obj: tuple) -> tuple:
            if isinstance(obj, (tuple, list, np.ndarray)):
                obj = tuple(to_tuple(el) for el in obj)
            return obj

        # Turn lists and arrays into tuples
        for f in fields(self):
            if f.name == "noise_types" or "eff_noise" in f.name:
                object.__setattr__(
                    self, f.name, to_tuple(getattr(self, f.name))
                )

        self._check_noise_types()
        self._check_eff_noise()

    def _check_noise_types(self) -> None:
        for noise_type in self.noise_types:
            if noise_type not in get_args(NOISE_TYPES):
                raise ValueError(
                    f"'{noise_type}' is not a valid noise type. "
                    + "Valid noise types: "
                    + ", ".join(get_args(NOISE_TYPES))
                )

    def _check_eff_noise(self) -> None:
        if len(self.eff_noise_opers) != len(self.eff_noise_rates):
            raise ValueError(
                f"The operators list length({len(self.eff_noise_opers)}) "
                "and rates list length"
                f"({len(self.eff_noise_rates)}) must be equal."
            )
        for rate in self.eff_noise_rates:
            if not isinstance(rate, float):
                raise TypeError(
                    "eff_noise_rates is a list of floats,"
                    f" it must not contain a {type(rate)}."
                )

        if "eff_noise" not in self.noise_types:
            # Stop here if effective noise is not selected
            return

        if not self.eff_noise_opers or not self.eff_noise_rates:
            raise ValueError(
                "The effective noise parameters have not been filled."
            )

        if np.any(np.array(self.eff_noise_rates) < 0):
            raise ValueError("The provided rates must be greater than 0.")

        # Check the validity of operators
        for op in self.eff_noise_opers:
            # type checking
            try:
                operator = np.array(op, dtype=complex)
            except Exception:
                raise TypeError(
                    f"Operator {op!r} is not castable to a Numpy array."
                )
            if operator.ndim != 2:
                raise ValueError(f"Operator '{op!r}' is not a 2D array.")

            if operator.shape != (2, 2):
                raise NotImplementedError(
                    f"Operator's shape must be (2,2) not {operator.shape}."
                )

    def _to_abstract_repr(self) -> dict[str, Any]:
        all_fields = asdict(self)
        eff_noise_rates = all_fields.pop("eff_noise_rates")
        eff_noise_opers = all_fields.pop("eff_noise_opers")
        all_fields["eff_noise"] = list(zip(eff_noise_rates, eff_noise_opers))
        return all_fields

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
