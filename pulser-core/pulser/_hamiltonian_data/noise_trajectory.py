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
    """Defines a noise trajectory."""

    bad_atoms: dict[QubitId, bool]
    doppler_detune: dict[QubitId, float]
    amp_fluctuations: dict[ChannelName, float]
    det_fluctuations: dict[ChannelName, float]
    det_phases: dict[ChannelName, np.ndarray]
    register: BaseRegister
    interaction_matrix: pm.AbstractArray
