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

from dataclasses import dataclass, field
from sys import version_info
from typing import Any, Optional, Union

import numpy as np
import qutip

if version_info[:2] >= (3, 8):  # pragma: no cover
    from typing import Literal, get_args
else:  # pragma: no cover
    try:
        from typing_extensions import Literal, get_args  # type: ignore
    except ImportError:
        raise ImportError(
            "Using pulser with Python version 3.7 requires the"
            " `typing_extensions` module. Install it by running"
            " `pip install typing-extensions`."
        )

NOISE_TYPES = Literal[
    "doppler", "amplitude", "SPAM", "dephasing", "depolarizing", "eff_noise"
]
MASS = 1.45e-25  # kg
KB = 1.38e-23  # J/K
KEFF = 8.7  # µm^-1


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

            - "dephasing": Random phase (Z) flip
            - "depolarizing": Quantum noise where the state(rho) is
              turned into a mixed state I/2 with probability p,
              and left unchanged with probability 1-p.
            - "eff_noise": General effective noise channel defined by
              the set of collapse operators **eff_noise_opers**
              and the corresponding probability distribution
              **eff_noise_probs**.
            - "doppler": Local atom detuning due to finite speed of the
              atoms and Doppler effect with respect to laser frequency
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

    noise: Union[NOISE_TYPES, tuple[NOISE_TYPES, ...]] = ()
    runs: int = 15
    samples_per_run: int = 5
    temperature: float = 50.0
    laser_waist: float = 175.0
    amp_sigma: float = 5e-2
    eta: float = 0.005
    epsilon: float = 0.01
    epsilon_prime: float = 0.05
    dephasing_prob: float = 0.05
    depolarizing_prob: float = 0.05
    eff_noise_probs: list[float] = field(default_factory=list, repr=False)
    eff_noise_opers: list[qutip.Qobj] = field(default_factory=list, repr=False)
    solver_options: Optional[qutip.Options] = None
    spam_dict: dict[str, float] = field(
        init=False, default_factory=dict, repr=False
    )
    doppler_sigma: float = field(
        init=False, default=KEFF * np.sqrt(KB * 50.0e-6 / MASS)
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.amp_sigma < 1.0:
            raise ValueError(
                "The standard deviation in amplitude (amp_sigma="
                f"{self.amp_sigma}) must be greater than or equal"
                " to 0. and smaller than 1."
            )
        self._process_temperature()
        self._change_attribute(
            "spam_dict",
            {
                "eta": self.eta,
                "epsilon": self.epsilon,
                "epsilon_prime": self.epsilon_prime,
            },
        )
        self._check_noise_types()
        self._check_spam_dict()
        self._calc_sigma_doppler()
        self._check_eff_noise()

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
                f"General noise distribution:       {self.eff_noise_probs}"
            )
            lines.append(
                f"General noise operators:       {self.eff_noise_opers}"
            )
        if "doppler" in self.noise:
            lines.append(f"Temperature:           {self.temperature*1.e6}µK")
            lines.append(f"Amplitude standard dev.:  {self.amp_sigma}")
        if "amplitude" in self.noise:
            lines.append(f"Laser waist:           {self.laser_waist}μm")
        if "dephasing" in self.noise:
            lines.append(f"Dephasing probability: {self.dephasing_prob}")
        if "depolarizing" in self.noise:
            lines.append(f"Depolarizing probability: {self.depolarizing_prob}")
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

    def _process_temperature(self) -> None:
        # checks value of temperature field and converts it to K from muK
        if self.temperature <= 0:
            raise ValueError(
                "Temperature field"
                + f" (`temperature` = {self.temperature}) must be"
                + " greater than 0."
            )
        self._change_attribute("temperature", self.temperature * 1.0e-6)

    def _check_noise_types(self) -> None:
        # only one noise was given as argument : convert it to a tuple
        if isinstance(self.noise, str):
            self._change_attribute("noise", (self.noise,))
        for noise_type in self.noise:
            if noise_type not in get_args(NOISE_TYPES):
                raise ValueError(
                    f"{noise_type} is not a valid noise type. "
                    + "Valid noise types: "
                    + ", ".join(get_args(NOISE_TYPES))
                )
        dephasing_on = "dephasing" in self.noise
        depolarizing_on = "depolarizing" in self.noise
        eff_noise_on = "eff_noise" in self.noise
        eff_noise_conflict = dephasing_on + depolarizing_on + eff_noise_on > 1
        if eff_noise_conflict:
            raise NotImplementedError(
                "Depolarizing, dephasing and eff_noise channels"
                "cannot be activated at the same time in"
                " one simulation."
            )

    def _calc_sigma_doppler(self) -> None:
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self._change_attribute(
            "doppler_sigma", KEFF * np.sqrt(KB * self.temperature / MASS)
        )

    def _change_attribute(self, attr_name: str, new_value: Any) -> None:
        object.__setattr__(self, attr_name, new_value)

    def _check_eff_noise(self) -> None:
        # Check the validity of the distribution of probability
        if "eff_noise" in self.noise:
            if len(self.eff_noise_opers) != len(self.eff_noise_probs):
                raise ValueError(
                    f"The operators list length({len(self.eff_noise_opers)}) "
                    "and probabilities list length"
                    f"({len(self.eff_noise_probs)}) must be equal."
                )
            if self.eff_noise_opers == [] or self.eff_noise_probs == []:
                raise ValueError(
                    "The general noise parameters have not been filled."
                )

            for prob in self.eff_noise_probs:
                if not isinstance(prob, float):
                    raise TypeError(
                        "eff_noise_probs is a list of floats,"
                        f" it must not contain a {type(prob)}."
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

                if type(operator) != qutip.qobj.Qobj:
                    raise TypeError(f"{operator} is not a Qobj.")
                if operator.type != "oper":
                    raise TypeError(
                        "Operators are supposed to be of type oper."
                    )
                if operator.shape != (2, 2):
                    raise NotImplementedError(
                        "Operator's shape must be (2,2) "
                        f"not {operator.shape}."
                    )
            # Identity position
            identity = qutip.qeye(2)
            if self.eff_noise_opers[0] != identity:
                raise NotImplementedError(
                    "You must put the identity matrix at the "
                    "beginning of the operator list."
                )
            # Completeness relation checking
            sum_op = qutip.Qobj(shape=(2, 2))
            length = len(self.eff_noise_probs)
            for i in range(length):
                sum_op += (
                    self.eff_noise_probs[i]
                    * self.eff_noise_opers[i]
                    * self.eff_noise_opers[i].dag()
                )

            if sum_op != identity:
                raise ValueError(
                    "The completeness relation is not verified."
                    f" Ended up with {sum_op} instead of {identity}."
                )
