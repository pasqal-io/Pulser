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
"""Contains SimConfig class that sets the configuration of a simulation."""

from __future__ import annotations

from sys import version_info
from dataclasses import dataclass, field

import numpy as np
import qutip

if version_info[:2] == (3, 7):  # pragma: no cover
    try:
        from typing_extensions import Literal, get_args
    except ImportError:
        raise ImportError(
            "Using pulser with Python version 3.7 requires the"
            " `typing_extensions` module. Install it by running"
            " `pip install typing-extensions`.")
else:  # pragma: no cover
    from typing import Literal, get_args, Union  # type: ignore

NOISE_TYPES = Literal['doppler', 'amplitude', 'SPAM', 'dephasing']
MASS = 1.45e-25  # kg
KB = 1.38e-23  # J/K
KEFF = 8.7  # µm^-1


@dataclass(frozen=True)
class SimConfig:
    """Includes additional parameters to simulation.

    Note:
        The configuration chosen upon instantiation cannot be changed
        later on.

    Keyword Arguments:
        noise (Union[NOISE_TYPES, tuple[NOISE_TYPES]]): Types of noises
            to be used in the simulation. You may specify just one, or a
            tuple of the allowed noise types:
            -   'dephasing': Random phase (Z) flip
            -   'doppler': Local atom detuning due to finite speed of the
                atoms and Doppler effect with respect to laser frequency
            -   'amplitude': Gaussian damping due to finite laser waist
            -   'SPAM': SPAM errors. Adds:
                --  eta: Probability of each atom to be badly prepared
                --  epsilon: Probability of false positives
                --  epsilon_prime: Probability of false negatives.
        runs (int): Number of runs needed : each run draws a new random
            noise.
        samples_per_run (int): Number of samples per noisy run.
            Useful for cutting down on computing time, but unrealistic.
        temperature (float): Temperature, set in µK, of the Rydberg array.
            Also sets the standard deviation of the speed of the atoms.
        laser_waist (float): Waist of the gaussian laser, set in µm,
            in global pulses.
        solver_options (qutip.Options): Options for the qutip solver.
    """
    noise: Union[NOISE_TYPES, tuple[NOISE_TYPES, ...]] = ()
    runs: int = 15
    samples_per_run: int = 5
    temperature: float = 50.
    laser_waist: float = 175.
    eta: float = 0.005
    epsilon: float = 0.01
    epsilon_prime: float = 0.05
    solver_options: qutip.Options = qutip.Options(max_step=5)
    spam_dict: dict[str, float] = field(init=False, default_factory=dict,
                                        repr=False)
    doppler_sigma: float = field(init=False,
                                 default=KEFF * np.sqrt(KB * 50.e-6 / MASS))

    def __post_init__(self) -> None:
        self._process_temperature()
        self.__dict__["spam_dict"] = {'eta': self.eta, 'epsilon': self.epsilon,
                                      'epsilon_prime': self.epsilon_prime}
        self._check_noise_types()
        self._check_spam_dict()
        self._calc_sigma_doppler()

    def __str__(self) -> str:
        lines = [
            "Options:",
            "----------",
            "Noise types:         " + ", ".join(self.noise),
            f"Spam dictionary:     {self.spam_dict}",
            f"Temperature:         {self.temperature}K",
            f"Number of runs:      {self.runs}",
            f"Samples per runs:    {self.samples_per_run}",
            f"Laser waist:         {self.laser_waist}μm",
            "Solver Options:",
            f"{str(self.solver_options)[10:-1]}",
            ]
        return "\n".join(lines)

    def _check_spam_dict(self) -> None:
        for param, value in self.spam_dict.items():
            if value > 1 or value < 0:
                raise ValueError(f"SPAM parameter {param} = {value} must be"
                                 + " greater than 0 and less than 1.")

    def _process_temperature(self) -> None:
        # checks value of temperature field and converts it to K from muK
        if self.temperature <= 0:
            raise ValueError("Temperature field"
                             + f" (`temperature` = {self.temperature}) must be"
                             + " greater than 0.")
        self.__dict__["temperature"] *= 1.e-6

    def _check_noise_types(self) -> None:
        # only one noise was given as argument : convert it to a tuple
        if isinstance(self.noise, str):
            self.__dict__["noise"] = (self.noise, )
        for noise_type in self.noise:
            if noise_type not in get_args(NOISE_TYPES):
                raise ValueError(
                    f"{noise_type} is not a valid noise type. " +
                    "Valid noise types: " + ", ".join(get_args(NOISE_TYPES))
                 )

    def _calc_sigma_doppler(self) -> None:
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self.__dict__["doppler_sigma"] = KEFF * np.sqrt(
            KB * self.temperature / MASS)
