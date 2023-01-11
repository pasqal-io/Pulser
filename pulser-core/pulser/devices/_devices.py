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
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices._device_datacls import Device

Chadoq2 = Device(
    name="Chadoq2",
    dimensions=2,
    rydberg_level=70,
    max_atom_num=100,
    max_radial_distance=50,
    min_atom_distance=4,
    supports_slm_mask=True,
    channel_objects=(
        Rydberg.Global(
            max_abs_detuning=2 * np.pi * 20,
            max_amp=2 * np.pi * 2.5,
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
        ),
        Rydberg.Local(
            max_abs_detuning=2 * np.pi * 20,
            max_amp=2 * np.pi * 10,
            min_retarget_interval=220,
            fixed_retarget_t=0,
            max_targets=1,
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
        ),
        Raman.Local(
            max_abs_detuning=2 * np.pi * 20,
            max_amp=2 * np.pi * 10,
            min_retarget_interval=220,
            fixed_retarget_t=0,
            max_targets=1,
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
        ),
    ),
)

IroiseMVP = Device(
    name="IroiseMVP",
    dimensions=2,
    rydberg_level=60,
    max_atom_num=100,
    max_radial_distance=60,
    min_atom_distance=5,
    channel_objects=(
        Rydberg.Global(
            max_abs_detuning=2 * np.pi * 4,
            max_amp=2 * np.pi * 3,
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
            mod_bandwidth=4,
            eom_config=RydbergEOM(
                limiting_beam=RydbergBeam.RED,
                max_limiting_amp=40 * 2 * np.pi,
                intermediate_detuning=700 * 2 * np.pi,
                mod_bandwidth=24,
                controlled_beams=(RydbergBeam.BLUE,),
            ),
        ),
    ),
)
