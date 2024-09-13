# Copyright 2020 Pulser Development Team
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
"""Contains the SimConfig class that sets the configuration of a simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import sqrt
from typing import Any, Optional, Tuple, Type, TypeVar, Union, cast

import qutip

from pulser.noise_model import _LEGACY_DEFAULTS, NoiseModel, NoiseTypes

MASS = 1.45e-25  # kg
KB = 1.38e-23  # J/K
KEFF = 8.7  # µm^-1

T = TypeVar("T", bound="SimConfig")

SUPPORTED_NOISES: dict = {
    "ising": {
        "amplitude",
        "dephasing",
        "relaxation",
        "depolarizing",
        "doppler",
        "eff_noise",
        "SPAM",
        "leakage",
    },
    "XY": {"dephasing", "depolarizing", "eff_noise", "SPAM", "leakage"},
}

# Maps the noise model parameters with a different name in SimConfig
_DIFF_NOISE_PARAMS = {
    "noise_types": "noise",
    "state_prep_error": "eta",
    "p_false_pos": "epsilon",
    "p_false_neg": "epsilon_prime",
}


def doppler_sigma(temperature: float) -> float:
    """Standard deviation for Doppler shifting due to thermal motion.

    Arg:
        temperature: The temperature in K.
    """
    return KEFF * sqrt(KB * temperature / MASS)


@dataclass(frozen=True)
class SimConfig:
    """Specifies a simulation's configuration.

    Note:
        Being a frozen dataclass, the configuration chosen upon instantiation
        cannot be changed later on.

    Args:
        noise: Types of noises to be used in the
            simulation. You may specify just one, or a tuple of the allowed
            noise types:

            - "leakage": Adds an error state 'x' to the computational
              basis, that can interact with the other states via an
              effective noise channel (which must be defined).
            - "relaxation": Relaxation from the Rydberg to the ground state.
            - "dephasing": Random phase (Z) flip.
            - "depolarizing": Quantum noise where the state (rho) is
              turned into a mixed state I/2 at a rate gamma (in rad/µs).
            - "eff_noise": General effective noise channel defined by
              the set of collapse operators **eff_noise_opers** and the
              corresponding rates **eff_noise_rates** (in rad/µs).
            - "doppler": Local atom detuning due to finite speed of the
              atoms and Doppler effect with respect to laser frequency.
            - "amplitude": Gaussian damping due to finite laser waist
            - "SPAM": SPAM errors. Defined by **eta**, **epsilon** and
              **epsilon_prime**.

        eta: Probability of each atom to be badly prepared.
        epsilon: Probability of false positives.
        epsilon_prime: Probability of false negatives.
        runs: Number of runs needed : each run draws a new random
            noise.
        samples_per_run: Number of samples per noisy run.
            Useful for cutting down on computing time, but unrealistic.
        temperature: Temperature, set in µK, of the Rydberg array.
            Also sets the standard deviation of the speed of the atoms.
        laser_waist: Waist of the gaussian laser, set in µm, in global
            pulses.
        amp_sigma: Dictates the fluctuations in amplitude as a standard
            deviation of a normal distribution centered in 1.
        solver_options: Options for the qutip solver.
    """

    noise: Union[NoiseTypes, tuple[NoiseTypes, ...]] = ()
    runs: int = cast(int, _LEGACY_DEFAULTS["runs"])
    samples_per_run: int = cast(int, _LEGACY_DEFAULTS["samples_per_run"])
    temperature: float = _LEGACY_DEFAULTS["temperature"]
    laser_waist: float = _LEGACY_DEFAULTS["laser_waist"]
    amp_sigma: float = _LEGACY_DEFAULTS["amp_sigma"]
    eta: float = _LEGACY_DEFAULTS["state_prep_error"]
    epsilon: float = _LEGACY_DEFAULTS["p_false_pos"]
    epsilon_prime: float = _LEGACY_DEFAULTS["p_false_neg"]
    relaxation_rate: float = _LEGACY_DEFAULTS["relaxation_rate"]
    dephasing_rate: float = _LEGACY_DEFAULTS["dephasing_rate"]
    hyperfine_dephasing_rate: float = _LEGACY_DEFAULTS[
        "hyperfine_dephasing_rate"
    ]
    depolarizing_rate: float = _LEGACY_DEFAULTS["depolarizing_rate"]
    eff_noise_rates: list[float] = field(default_factory=list, repr=False)
    eff_noise_opers: list[qutip.Qobj] = field(default_factory=list, repr=False)
    solver_options: Optional[qutip.Options] = None

    @classmethod
    def from_noise_model(cls: Type[T], noise_model: NoiseModel) -> T:
        """Creates a SimConfig from a NoiseModel."""
        kwargs: dict[str, Any] = dict(noise=noise_model.noise_types)
        relevant_params = NoiseModel._find_relevant_params(
            noise_model.noise_types,
            noise_model.state_prep_error,
            noise_model.amp_sigma,
            noise_model.laser_waist,
        )
        for param in relevant_params:
            kwargs[_DIFF_NOISE_PARAMS.get(param, param)] = getattr(
                noise_model, param
            )
        kwargs.pop("with_leakage", None)
        return cls(**kwargs)

    def to_noise_model(self) -> NoiseModel:
        """Creates a NoiseModel from the SimConfig."""
        relevant_params = NoiseModel._find_relevant_params(
            cast(Tuple[NoiseTypes, ...], self.noise),
            self.eta,
            self.amp_sigma,
            self.laser_waist,
        )
        kwargs = {}
        for param in relevant_params:
            kwargs[param] = getattr(self, _DIFF_NOISE_PARAMS.get(param, param))
        if "temperature" in kwargs:
            kwargs["temperature"] *= 1e6  # Converts back to µK
        return NoiseModel(**kwargs)

    def __post_init__(self) -> None:
        # only one noise was given as argument : convert it to a tuple
        if isinstance(self.noise, str):
            self._change_attribute("noise", (self.noise,))

        # Converts temperature from µK to K
        if not isinstance(self.temperature, (int, float)):
            raise TypeError(
                f"'temperature' must be a float, not {type(self.temperature)}."
            )
        self._change_attribute("temperature", self.temperature / 1e6)

        NoiseModel._check_noise_types(cast(Tuple[NoiseTypes], self.noise))
        self._check_spam_dict()
        self._check_eff_noise()
        NoiseModel._validate_parameters(
            {f.name: getattr(self, f.name) for f in fields(self)}
        )

    @property
    def with_leakage(self) -> bool:
        """Whether or not 'leakage' is included in the noise types."""
        return "leakage" in self.noise

    @property
    def spam_dict(self) -> dict[str, float]:
        """A dictionary combining the SPAM error parameters."""
        return {
            "eta": self.eta,
            "epsilon": self.epsilon,
            "epsilon_prime": self.epsilon_prime,
        }

    @property
    def doppler_sigma(self) -> float:
        """Standard deviation for Doppler shifting due to thermal motion."""
        return doppler_sigma(self.temperature)

    def __str__(self, solver_options: bool = False) -> str:
        lines = [
            "Options:",
            "----------",
            f"Number of runs:        {self.runs}",
            f"Samples per run:       {self.samples_per_run}",
        ]
        if self.noise:
            lines.append("Noise types:           " + ", ".join(self.noise))
        if "SPAM" in self.noise:
            lines.append(f"SPAM dictionary:       {self.spam_dict}")
        if "eff_noise" in self.noise:
            lines.append(
                f"Effective noise rates:       {self.eff_noise_rates}"
            )
            lines.append(
                f"Effective noise operators:       {self.eff_noise_opers}"
            )
        if "doppler" in self.noise:
            lines.append(f"Temperature:           {self.temperature*1.e6}µK")
        if "amplitude" in self.noise:
            lines.append(f"Laser waist:           {self.laser_waist}μm")
            lines.append(f"Amplitude standard dev.:  {self.amp_sigma}")
        if "relaxation" in self.noise:
            lines.append(f"Relaxation rate: {self.relaxation_rate}")
        if "dephasing" in self.noise:
            lines.append(
                f"Dephasing rate: {self.dephasing_rate} (Rydberg), "
                f"{self.hyperfine_dephasing_rate} (Hyperfine)"
            )
        if "depolarizing" in self.noise:
            lines.append(f"Depolarizing rate: {self.depolarizing_rate}")
        if solver_options:
            lines.append(
                "Solver Options: \n" + f"{str(self.solver_options)[10:-1]}"
            )
        return "\n".join(lines).rstrip()

    def _check_spam_dict(self) -> None:
        for param, value in self.spam_dict.items():
            if value > 1 or value < 0:
                raise ValueError(
                    f"SPAM parameter {param} = {value} must be"
                    + " greater than 0 and less than 1."
                )

    def _change_attribute(self, attr_name: str, new_value: Any) -> None:
        object.__setattr__(self, attr_name, new_value)

    def _check_eff_noise(self) -> None:
        # Check the validity of operators
        for operator in self.eff_noise_opers:
            # type checking
            if not isinstance(operator, qutip.Qobj):
                raise TypeError(f"{operator} is not a Qobj.")
            if operator.type != "oper":
                raise TypeError(
                    "Operators are supposed to be of Qutip type 'oper'."
                )
        NoiseModel._check_eff_noise(
            self.eff_noise_rates,
            self.eff_noise_opers,
            "eff_noise" in self.noise,
            self.with_leakage,
        )

    @property
    def supported_noises(self) -> dict:
        """Return the noises implemented on pulser."""
        return SUPPORTED_NOISES
