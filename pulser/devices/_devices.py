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
"""Definitions of real devices."""
import numpy as np

from pulser.channels import Raman, Rydberg
from pulser.devices._device_datacls import Device

Chadoq2 = Device(
    name="Chadoq2",
    dimensions=2,
    rydberg_level=70,
    max_atom_num=100,
    max_radial_distance=50,
    min_atom_distance=4,
    _channels=(
        ("rydberg_global", Rydberg.Global(2 * np.pi * 20, 2 * np.pi * 2.5)),
        ("rydberg_local", Rydberg.Local(2 * np.pi * 20, 2 * np.pi * 10)),
        ("raman_local", Raman.Local(2 * np.pi * 20, 2 * np.pi * 10)),
    ),
)

IroiseMVP = Device(
    name="IroiseMVP",
    dimensions=2,
    rydberg_level=60,
    max_atom_num=100,
    max_radial_distance=60,
    min_atom_distance=5,
    _channels=(
        (
            "rydberg_global",
            Rydberg.Global(
                max_abs_detuning=2 * np.pi * 4,
                max_amp=2 * np.pi * 3,
                phase_jump_time=500,
            ),
        ),
    ),
)
