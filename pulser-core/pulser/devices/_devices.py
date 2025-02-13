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
"""Examples of realistic devices."""
import numpy as np

from pulser.channels import DMM, Raman, Rydberg
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.devices._device_datacls import Device
from pulser.register.special_layouts import TriangularLatticeLayout

DigitalAnalogDevice = Device(
    name="DigitalAnalogDevice",
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
    dmm_objects=(
        DMM(
            clock_period=4,
            min_duration=16,
            max_duration=2**26,
            bottom_detuning=-2 * np.pi * 20,
            total_bottom_detuning=-2 * np.pi * 2000,
        ),
    ),
    short_description="A device with digital and analog capabilites.",
)

AnalogDevice = Device(
    name="AnalogDevice",
    dimensions=2,
    rydberg_level=60,
    max_atom_num=80,
    max_radial_distance=38,
    min_atom_distance=5,
    max_sequence_duration=6000,
    max_runs=2000,
    requires_layout=True,
    accepts_new_layouts=True,
    optimal_layout_filling=0.45,
    channel_objects=(
        Rydberg.Global(
            max_abs_detuning=2 * np.pi * 20,
            max_amp=2 * np.pi * 2,
            clock_period=4,
            min_duration=16,
            mod_bandwidth=8,
            eom_config=RydbergEOM(
                limiting_beam=RydbergBeam.RED,
                max_limiting_amp=30 * 2 * np.pi,
                intermediate_detuning=450 * 2 * np.pi,
                mod_bandwidth=40,
                controlled_beams=(RydbergBeam.BLUE,),
                custom_buffer_time=240,
            ),
        ),
    ),
    pre_calibrated_layouts=(TriangularLatticeLayout(61, 5),),
    short_description="A realistic device for analog sequence execution.",
)
