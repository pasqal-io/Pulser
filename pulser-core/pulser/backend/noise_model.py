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

import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Literal, get_args

import numpy as np

NOISE_TYPES = Literal[
    "doppler", "amplitude", "SPAM", "dephasing", "depolarizing", "eff_noise"
]


@dataclass(frozen=True)
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
              by `dephasing_rate`).
            - "depolarizing": Quantum noise where the state is
              turned into a mixed state I/2 with rate
              `depolarizing_rate`.
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
        dephasing_rate: The rate of a dephasing error occuring (in rad/µs).
        dephasing_prob: (Deprecated) The rate of a dephasing error occuring
            (in rad/µs). Use `dephasing_rate` instead.
        depolarizing_rate: The rate (in rad/µs) at which a depolarizing
            error occurs.
        depolarizing_prob: (Deprecated) The rate (in rad/µs) at which a
            depolarizing error occurs. Use `depolarizing_rate` instead.
        eff_noise_rates: The rate associated to each effective noise operator
            (in rad/µs).
        eff_noise_probs: (Deprecated) The rate associated to each effective
            noise operator (in rad/µs). Use `eff_noise_rate` instead.
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
    dephasing_rate: float = 0.05
    depolarizing_rate: float = 0.05
    eff_noise_rates: list[float] = field(default_factory=list)
    eff_noise_opers: list[np.ndarray] = field(default_factory=list)
    dephasing_prob: float | None = None
    depolarizing_prob: float | None = None
    eff_noise_probs: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        default_field_value = {
            field.name: field.default for field in fields(self)
        }
        for noise in ["dephasing", "depolarizing", "eff_noise"]:
            # Probability and rates should be the same
            prob_name = f"{noise}_prob{'s' if noise=='eff_noise' else ''}"
            rate_name = f"{noise}_rate{'s' if noise=='eff_noise' else ''}"
            prob, rate = (getattr(self, prob_name), getattr(self, rate_name))
            if len(prob) > 0 if noise == "eff_noise" else prob is not None:
                warnings.warn(
                    f"{prob_name} is deprecated. Use {rate_name} instead.",
                    DeprecationWarning,
                )
                if prob != rate:
                    if (
                        len(rate) > 0
                        if noise == "eff_noise"
                        else rate != default_field_value[rate_name]
                    ):
                        raise ValueError(
                            f"If both defined, `{rate_name}` and `{prob_name}`"
                            " must be equal."
                        )
                    warnings.warn(
                        f"Setting {rate_name} with the value from "
                        f"{prob_name}.",
                        UserWarning,
                    )
                    self._change_attribute(rate_name, prob)
            self._change_attribute(prob_name, getattr(self, rate_name))
        assert self.dephasing_prob == self.dephasing_rate
        assert self.depolarizing_prob == self.depolarizing_rate
        assert self.eff_noise_probs == self.eff_noise_rates
        positive = {
            "dephasing_rate",
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

        self._check_noise_types()
        self._check_eff_noise()

    def _change_attribute(self, attr_name: str, new_value: Any) -> None:
        object.__setattr__(self, attr_name, new_value)

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
        for operator in self.eff_noise_opers:
            # type checking
            if not isinstance(operator, np.ndarray):
                raise TypeError(f"{operator} is not a Numpy array.")
            if operator.shape != (2, 2):
                raise NotImplementedError(
                    "Operator's shape must be (2,2) " f"not {operator.shape}."
                )
