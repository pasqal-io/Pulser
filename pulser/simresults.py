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

import qutip
import numpy as np


class SimulationResults():
    """Result of a simulation run of a pulse sequence.
    """

    def __init__(self, run_output, dim, size):
        if not isinstance(run_output, list):
            raise TypeError("Received simulation output is not a list of Qobj")
        if not all(isinstance(x, qutip.Qobj) for x in run_output):
            raise TypeError("Received output elements are not all Qobj")
        self.states = run_output
        self.dim = dim
        self.size = size

    def expect(self, obs_list):
        """Calculate the expectation value of a list of observables.

        Args:
            obs_list (list): A list of observables whose
                      expectation value will be calculated. Each member will
                      be transformed into a qutip.Qobj instance.
        """
        if not isinstance(obs_list, list):
            raise TypeError("`obs_list` must be a list of operators")

        for i, obs in enumerate(obs_list):
            if obs.shape != (self.dim**self.size, self.dim**self.size):
                raise ValueError('Incompatible shape of observable')
            if not isinstance(obs, qutip.Qobj):
                # Transfrom to qutip.Qobj and take dims from state
                dim_list = [self.states[0].dims[0], self.states[0].dims[0]]
                obs_list[i] = qutip.Qobj(obs, dims=dim_list)

        return [qutip.expect(obs, self.states) for obs in obs_list]

    def sample_final_state(self, N_samples=1000):
        """Calculate the expectation value of a list of observables.

        Args:
            N_samples (int): Number of samples to take.
        """

        weights = np.abs(self.states[-1])**2
        N = len(weights)
        dist = np.random.binomial(N_samples, weights)
        return {np.binary_repr(i, N): dist[i][0] for i in np.nonzero(dist)[0]}
