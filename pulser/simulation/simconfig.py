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

from typing import Optional
from typing_extensions import Literal, get_args
import collections.abc as abc

import numpy as np
import qutip


NOISE_TYPES = Literal['doppler', 'amplitude', 'SPAM', 'dephasing']


class SimConfig:
    """Include additional parameters to simulation.

        Keyword arguments:
            noise (list[NOISE_TYPES]): Types of noises to be used in the
                simulation.
            samples_per_run (int value): Number of samples per noisy run.
                Useful for cutting down on computing time, but unrealistic.
            runs (int value): Number of runs needed : each run
                draws a new random noise.
            laser_waist (float): Waist of the gaussian laser in global pulses.
            solver_options (qutip.Options): options for the qutip solver.
        """
    def __init__(self, noise: list[NOISE_TYPES] = [],
                 runs: int = 15, samples_per_run: int = 5,
                 temperature: float = 50., laser_waist: float = 175.,
                 solver_options: qutip.Options = qutip.Options(max_step=5)):
        self.noise = noise
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
        s += "Noise types:       " + str(self.noise) + "\n"
        s += "Spam dictionary:   " + str(self.spam_dict) + "\n"
        s += "Temperature:       " + str(self.temperature) + "K" + "\n"
        s += "Number of runs:    " + str(self.runs) + "\n"
        s += "Samples per runs:  " + str(self.samples_per_run) + "\n"
        s += "Laser waist:       " + str(self.laser_waist) + "μm \n"
        s += "\n" + "Solver Options: \n"+str(self.solver_options)[10:-1]+"\n"
        return s

    @property
    def noise(self) -> list[NOISE_TYPES]:
        return self._noise

    @noise.setter
    def noise(self, noise_types: abc.Sequence[NOISE_TYPES]) -> None:
        self._noise: list[NOISE_TYPES] = []
        for noise_type in noise_types:
            self._add_noise(noise_type)

    def _add_noise(self, noise_type: NOISE_TYPES) -> None:
        """Adds a noise model to the SimConfig instance.

            Args:
                noise_type (NOISE_TYPES): Choose among:
                -   'dephasing': Random phase (Z) flip
                -   'doppler': Noisy doppler runs
                -   'amplitude': Noisy gaussian beam
                -   'SPAM': SPAM errors. Adds:
                    --  eta: Probability of each atom to be badly prepared
                    --  epsilon: Probability of false positives
                    --  epsilon_prime: Probability of false negatives.
        """
        # Check proper input:
        if noise_type not in get_args(NOISE_TYPES):
            raise ValueError('Not a valid noise type')
        self._noise.append(noise_type)

    @property
    def spam_dict(self) -> dict[str, float]:
        return self._spam_dict

    def set_spam(self, eta: Optional[float] = None,
                 epsilon: Optional[float] = None,
                 epsilon_prime: Optional[float] = None) -> None:
        """Allows the user to change SPAM parameters in dictionary"""
        values = {'eta': eta, 'epsilon': epsilon,
                  'epsilon_prime': epsilon_prime}
        for param in values:
            val = values[param]
            if val is not None:
                if val > 1. or val < 0.:
                    raise ValueError('Invalid value : must be between 0 and 1')
                self._spam_dict[param] = val

    def init_spam(self) -> None:
        self._spam_dict: dict[str, float] = \
            {'eta': 0.005, 'epsilon': 0.01, 'epsilon_prime': 0.05}

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        # value set in microkelvin
        self._temperature = value * 1e-6
        self._calc_sigma_doppler()

    @property
    def doppler_sigma(self) -> float:
        return self._doppler_sigma

    def _calc_sigma_doppler(self) -> None:
        # sigma = keff Deltav, keff = 8.7mum^-1, Deltav = sqrt(kB T / m)
        self._doppler_sigma: float = 8.7 * np.sqrt(
            1.38e-23 * self._temperature / 1.45e-25)

    def remove_all_noise(self) -> None:
        """Removes noise from simulation."""
        self._noise = []
