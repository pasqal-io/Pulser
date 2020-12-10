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
"""Definitions of Pasqal's devices."""

from pulser.devices._pasqal_device import PasqalDevice
from pulser.channels import Raman, Rydberg


Chadoq2 = PasqalDevice(
            name="Chadoq2",
            max_dimensionality=2,
            max_atom_num=100,
            max_radial_distance=50,
            min_atom_distance=4,
            channel_names=(
                "rydberg_global",
                "rydberg_local",
                "rydberg_local2",
                "raman_local",
                ),
            channel_objs=(
                Rydberg.Global(50, 2.5),
                Rydberg.Local(50, 10, 100),
                Rydberg.Local(50, 10, 100),
                Raman.Local(50, 10, 100),
                )
            )
