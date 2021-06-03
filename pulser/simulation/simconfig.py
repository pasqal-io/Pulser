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

import numpy as np
import qutip
from typing import Optional, Union, cast, List
from numpy.typing import ArrayLike

noise_dict_set = {'doppler', 'amplitude', 'SPAM', 'dephasing'}


class SimConfig:
    """Include additional parameters to simulation. Will be necessary for
        noisy simulations.

        Keyword arguments:
            noise_types (string list): types of noises to be used in the
                simulation.
            samples_per_run (int value): number of samples per noisy run.
                Useful for cutting down on computing time, but unrealistic.
            runs (int value): number of runs needed : each run
                draws a new random noise.
            laser_waist (float): waist of the gaussian laser in global pulses.
            solver_options (qutip.Options): options for the qutip solver.
        """
    def __init__(self, noise_types: List[str] = [], runs: int = 15,
                 samples_per_run: int = 5, temperature: float = 50.,
                 laser_waist: float = 175.,
                 solver_options: qutip.Options = qutip.Options(max_step=5)):
        self.noise = noise_types
        self.temperature = temperature
        self.runs = runs
        self.samples_per_run = samples_per_run
        self.laser_waist = laser_waist
        self.solver_options = solver_options
        self.init_spam()

    def __str__(self) -> str:
        s = ""
        s += "Options:\n"
        s += "----------       \n"
        s += "noise types:       " + str(self.noise) + "\n"
        s += "spam dictionary:   " + str(self.spam_dict) + "\n"
        s += "temperature:       " + str(self.temperature) + "K" + "\n"
        s += "number of runs:    " + str(self.runs) + "\n"
        s += "samples per runs:  " + str(self.samples_per_run) + "\n"
        s += "laser waist:       " + str(self.laser_waist) + "μm \n"
        s += "\n" + "Solver Options: \n"+str(self.solver_options)[10:-1]+"\n"
        return s

    @property
    def noise(self) -> List[str]:
        return self._noise

    @noise.setter
    def noise(self, noise_types: List[str]) -> None:
        self._noise: List[str] = []
        for noise_type in noise_types:
            self._add_noise(noise_type)

    def _add_noise(self, noise_type: str) -> None:
        """Adds a noise model to the SimConfig instance, to be used
            in Simulation.
            Args:
                noise_type (str): Choose among:
                    'dephasing': random phase (Z) flip
                    'doppler': Noisy doppler runs
                    'amplitude': Noisy gaussian beam
                    'SPAM': SPAM errors. Adds:
                        eta: Probability of each atom to be badly prepared
                        epsilon: false positives
                        epsilon_prime: false negatives
        """
        # Check proper input:
        if noise_type not in noise_dict_set:
            raise ValueError('Not a valid noise type')
        self._noise.append(noise_type)

    @property
    def spam_dict(self) -> dict[str, float]:
        return self._spam_dict

    def set_spam(self, **values: float) -> None:
        """Allows the user to change SPAM parameters in dictionary"""
        for param in values:
            if param not in {'eta', 'epsilon', 'epsilon_prime'}:
                raise ValueError('Not a valid SPAM parameter')
            if values[param] > 1 or values[param] < 0:
                raise ValueError('Invalid value : must be between 0 and 1')
            self._spam_dict[param] = values[param]

    def init_spam(self) -> None:
        self._spam_dict: dict[str, float] = \
            {'eta': 0.005, 'epsilon': 0.01, 'epsilon_prime': 0.05}

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        # value set in microkelvin
        self._temperature = value * 10.**-6
        self._calc_sigma_doppler()

    @property
    def doppler_sigma(self) -> float:
        return self._doppler_sigma

    def _calc_sigma_doppler(self) -> None:
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self._doppler_sigma: float = \
            8.7 * np.sqrt(1.38e-23 * self._temperature / 1.45e-25)

    def remove_all_noise(self) -> None:
        """Removes noise from simulation"""
        self._noise = []
