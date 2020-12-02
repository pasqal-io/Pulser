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

import pytest

from pulser.channels import Raman, Rydberg


def test_init():
    with pytest.raises(ValueError, match="set retarget time for local"):
        Raman('Local', 50, 10)

    with pytest.raises(ValueError, match="'Global' or 'Local'"):
        Raman('total', 20, 1)
        Rydberg(2, 1, 0)

    with pytest.raises(ValueError, match="detuning has to be positive"):
        Raman.Global(-1, 1)

    with pytest.raises(ValueError, match="amplitude has to be positive"):
        Rydberg.Global(1, 0)

    with pytest.raises(TypeError):
        Raman.Local(1, 19, 100, max_targets=1.5)

    with pytest.raises(ValueError, match='at least 1'):
        Raman.Local(1, 19, 100, max_targets=0)

    Raman.Local(10, 2, retarget_time=1000)
    Rydberg.Global(50, 2.5)


def test_repr():
    raman = Raman.Local(10, 2, retarget_time=1000, max_targets=4)
    r1 = ("Raman.Local(Max Absolute Detuning: 10 MHz, Max Amplitude: 2 MHz,"
          " Target time: 1000 ns, Max targets: 4, Basis: 'digital')")
    assert raman.__str__() == r1

    ryd = Rydberg.Global(50, 2.5)
    r2 = ("Rydberg.Global(Max Absolute Detuning: 50 MHz, "
          "Max Amplitude: 2.5 MHz, Basis: 'ground-rydberg')")
    assert ryd.__str__() == r2
