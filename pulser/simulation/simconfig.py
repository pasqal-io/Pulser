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

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Literal, get_args

import numpy as np
import qutip


NOISE_TYPES = Literal['doppler', 'amplitude', 'SPAM', 'dephasing']
mass = 1.45e-25
kb = 1.38e-23
keff = 8.7


@dataclass
class SimConfig:
    """Include additional parameters to simulation.

        The user chooses the settings when creating a SimConfig. Settings
        shouldn't be changed after instantiation of a SimConfig.

        Keyword arguments:
            noise (list[NOISE_TYPES]): Types of noises to be used in the
                simulation. Choose among:
                -   'dephasing': Random phase (Z) flip
                -   'doppler': Noisy doppler runs
                -   'amplitude': Noisy gaussian beam
                -   'SPAM': SPAM errors. Adds:
                    --  eta: Probability of each atom to be badly prepared
                    --  epsilon: Probability of false positives
                    --  epsilon_prime: Probability of false negatives.
            runs (int): Number of runs needed : each run draws a new random
                noise.
            samples_per_run (int): Number of samples per noisy run.
                Useful for cutting down on computing time, but unrealistic.
            temperature (float): Temperature, set in muK, of the Rydberg array.
                Also sets the standard deviation of the speed of the atoms.
            laser_waist (float): Waist of the gaussian laser in global pulses.
            solver_options (qutip.Options): options for the qutip solver.
        """
    noise: list[NOISE_TYPES] = field(default_factory=list)
    runs: int = 15
    samples_per_run: int = 5
    temperature: float = 50.
    laser_waist: float = 175.
    eta: float = 0.005
    epsilon: float = 0.01
    epsilon_prime: float = 0.05
    solver_options: qutip.Options = qutip.Options(max_step=5)

    def __post_init__(self):
        self._process_temperature()
        self._check_noise_types()
        self.spam_dict = {'eta': self.eta, 'epsilon': self.epsilon,
                          'epsilon_prime': self.epsilon_prime}
        self._check_spam_dict()
        self._calc_sigma_doppler()

    def __str__(self) -> str:
        s = f"""Options:
----------
Noise types:         {self.noise}
Spam dictionary:     {self.spam_dict}
Temperature:         {self.temperature}K
Number of runs:      {self.runs}
Samples per runs:    {self.samples_per_run}
Laser waist:         {self.laser_waist}μm

Solver Options:
{str(self.solver_options)[10:-1]}"""
        return s

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
        self.temperature *= 1.e-6

    def _check_noise_types(self) -> None:
        for noise_type in self.noise:
            if noise_type not in get_args(NOISE_TYPES):
                raise ValueError(str(noise_type)+" is not a valid noise type."
                                 "Valid noise types : " + get_args(NOISE_TYPES)
                                 )

    def _calc_sigma_doppler(self) -> None:
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self.doppler_sigma: float = keff * np.sqrt(
            kb * self.temperature / mass)
