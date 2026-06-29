# Copyright 2025 Pulser Development Team
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
"""Definition of a noise trajectory."""

from dataclasses import dataclass

import numpy as np

import pulser.math as pm
from pulser.register.base_register import BaseRegister, QubitId

ChannelName = str


@dataclass(frozen=True)
class NoiseTrajectory:
    """Defines a noise trajectory.

    Args:
        bad_atoms: Whether each atom is present or bad.
            False means it's present, True means it's bad.
        doppler_detune: The time-independent doppler detuning error per qubit.
        amp_fluctuations:
            The time-independent amplitude fluctuation per channel.
        det_fluctuations:
            The time-independent detuning fluctuation per non-DMM channel.
        det_phases:
            The random phase for each frequency component in the
            time-dependent detuning noise. The amplitude for each component
            is taken from the noise model and is non-random.
        register: The qubit register positions including noise.
        interaction_matrix:
            Packed interaction matrix for the two body term in the
            Hamiltonian. Should be of shape (2,N,N) for XY,
            encoding the C3 and C6 term in that order.
            Should be of shape (1,N,N) otherwise.
        dmm_det_fluctuation:
            The time-independent detuning fluctuations per DMM channel.
    """

    bad_atoms: dict[QubitId, bool]
    doppler_detune: dict[QubitId, float]
    amp_fluctuations: dict[ChannelName, float]
    det_fluctuations: dict[ChannelName, float]
    det_phases: dict[ChannelName, np.ndarray]
    register: BaseRegister
    interaction_matrix: pm.AbstractArray
    dmm_det_fluctuation: dict[ChannelName, float]
