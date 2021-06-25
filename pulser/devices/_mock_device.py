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

from pulser.devices._device_datacls import Device
from pulser.channels import Rydberg, Raman, Microwave


MockDevice = Device(
    name="MockDevice",
    dimensions=2,
    max_atom_num=2000,
    max_radial_distance=1000,
    min_atom_distance=1,
    _channels=(
        (
            "rydberg_global",
            Rydberg.Global(1000, 200, clock_period=1, min_duration=1),
        ),
        (
            "rydberg_local",
            Rydberg.Local(
                1000, 200, 0, max_targets=2000, clock_period=1, min_duration=1
            ),
        ),
        (
            "raman_global",
            Raman.Global(1000, 200, clock_period=1, min_duration=1),
        ),
        (
            "raman_local",
            Raman.Local(
                1000, 200, 0, max_targets=2000, clock_period=1, min_duration=1
            ),
        ),
        (
            "mw_global",
            Microwave.Global(1000, 200, clock_period=1, min_duration=1),
        ),
    ),
)
