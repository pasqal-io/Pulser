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

from dataclasses import dataclass, field, fields
from typing import Literal, get_args

import numpy as np

NOISE_TYPES = Literal[
    "doppler", "amplitude", "SPAM", "dephasing", "depolarizing", "eff_noise"
]


@dataclass
class NoiseModel:
    """Specifies the noise model parameters for emulation.

    Select the desired noise types in `noise_types` and, if necessary,
    modifiy the default values of related parameters.
    Non-specified parameters will have reasonable default value which
    is only taken into account when the related noise type is selected.

    Args:
        noise_types: Noise types to include in the emulation. Available
            options:

            - "dephasing": Random phase (Z) flip (parametrized
              by `dephasing_prob`).
            - "depolarizing": Quantum noise where the state is
              turned into a mixed state I/2 with probability
              `depolarizing_prob`.
            - "eff_noise": General effective noise channel defined by
              the set of collapse operators `eff_noise_opers`
              and the corresponding probability distribution
              `eff_noise_probs`.
            - "doppler": Local atom detuning due to termal motion of the
              atoms and Doppler effect with respect to laser frequency.
              Parametrized by the `temperature` field.
            - "amplitude": Gaussian damping due to finite laser waist and
                laser amplitude fluctuations. Parametrized by `laser_waist`
                and `amp_sigma`.
            - "SPAM": SPAM errors. Parametrized by `state_prep_error`,
                `p_false_pos` and `p_false_neg`.

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
        dephasing_prob: The probability of a dephasing error occuring.
        depolarizing_prob: The probability of a depolarizing error occuring.
        eff_noise_probs: The probability associated to each effective noise
            operator.
        eff_noise_opers: The operators for the effective noise model. The
            first operator must be the identity.
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
    dephasing_prob: float = 0.05
    depolarizing_prob: float = 0.05
    eff_noise_probs: list[float] = field(default_factory=list)
    eff_noise_opers: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
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
            "dephasing_prob",
            "depolarizing_prob",
            "amp_sigma",
        }
        # The two share no common terms
        assert not strict_positive.intersection(probability_like)

        for f in fields(self):
            is_valid = True
            param = f.name
            value = getattr(self, param)
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
        dephasing_on = "dephasing" in self.noise_types
        depolarizing_on = "depolarizing" in self.noise_types
        eff_noise_on = "eff_noise" in self.noise_types
        eff_noise_conflict = dephasing_on + depolarizing_on + eff_noise_on > 1
        if eff_noise_conflict:
            raise NotImplementedError(
                "Depolarizing, dephasing and effective noise channels"
                "cannot be simultaneously selected."
            )

    def _check_eff_noise(self) -> None:
        if len(self.eff_noise_opers) != len(self.eff_noise_probs):
            raise ValueError(
                f"The operators list length({len(self.eff_noise_opers)}) "
                "and probabilities list length"
                f"({len(self.eff_noise_probs)}) must be equal."
            )
        for prob in self.eff_noise_probs:
            if not isinstance(prob, float):
                raise TypeError(
                    "eff_noise_probs is a list of floats,"
                    f" it must not contain a {type(prob)}."
                )

        if "eff_noise" not in self.noise_types:
            # Stop here if effective noise is not selected
            return

        if not self.eff_noise_opers or not self.eff_noise_probs:
            raise ValueError(
                "The general noise parameters have not been filled."
            )

        prob_distr = np.array(self.eff_noise_probs)
        lower_bound = np.any(prob_distr < 0.0)
        upper_bound = np.any(prob_distr > 1.0)
        sum_p = not np.isclose(sum(prob_distr), 1.0)

        if sum_p or lower_bound or upper_bound:
            raise ValueError(
                "The distribution given is not a probability distribution."
            )

        # Check the validity of operators
        for operator in self.eff_noise_opers:
            # type checking
            if not isinstance(operator, np.ndarray):
                raise TypeError(f"{operator} is not a Numpy array.")
            if operator.shape != (2, 2):
                raise NotImplementedError(
                    "Operator's shape must be (2,2) " f"not {operator.shape}."
                )
        # Identity position
        identity = np.eye(2)
        if np.any(self.eff_noise_opers[0] != identity):
            raise NotImplementedError(
                "You must put the identity matrix at the "
                "beginning of the operator list."
            )
        # Completeness relation checking
        sum_op = np.zeros((2, 2), dtype=complex)
        for prob, op in zip(self.eff_noise_probs, self.eff_noise_opers):
            sum_op += prob * op @ op.conj().transpose()

        if not np.all(np.isclose(sum_op, identity)):
            raise ValueError(
                "The completeness relation is not verified."
                f" Ended up with {sum_op} instead of {identity}."
            )
