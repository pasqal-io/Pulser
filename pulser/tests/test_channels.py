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

import pulser
from pulser.channels import Raman, Rydberg


def test_device_channels():
    for dev in pulser.devices._valid_devices:
        for i, (id, ch) in enumerate(dev.channels.items()):
            assert id == dev._channels[i][0]
            assert isinstance(id, str)
            assert ch == dev._channels[i][1]
            assert isinstance(ch, pulser.channels.Channel)
            assert ch.name in ['Rydberg', 'Raman']
            assert ch.basis in ['digital', 'ground-rydberg']
            assert ch.addressing in ['Local', 'Global']
            assert ch.max_abs_detuning >= 0
            assert ch.max_amp > 0
            if ch.addressing == 'Local':
                assert ch.retarget_time >= 0
                assert ch.retarget_time == int(ch.retarget_time)
                assert ch.max_targets >= 1
                assert ch.max_targets == int(ch.max_targets)


def test_repr():
    raman = Raman.Local(10, 2, retarget_time=1000, max_targets=4)
    r1 = ("Raman.Local(Max Absolute Detuning: 10 MHz, Max Amplitude: 2 MHz,"
          " Target time: 1000 ns, Max targets: 4, Basis: 'digital')")
    assert raman.__str__() == r1

    ryd = Rydberg.Global(50, 2.5)
    r2 = ("Rydberg.Global(Max Absolute Detuning: 50 MHz, "
          "Max Amplitude: 2.5 MHz, Basis: 'ground-rydberg')")
    assert ryd.__str__() == r2
