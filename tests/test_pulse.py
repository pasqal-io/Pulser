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
import dataclasses

import numpy as np
import pytest

from pulser import Pulse
from pulser.channels import Rydberg
from pulser.channels.eom import RydbergBeam, RydbergEOM
from pulser.waveforms import BlackmanWaveform, ConstantWaveform, RampWaveform

cwf = ConstantWaveform(100, -10)
bwf = BlackmanWaveform(200, 3)
rwf = RampWaveform(200, 0, 1)

pls = Pulse(bwf, bwf, 2 * np.pi)
pls2 = Pulse.ConstantPulse(100, 1, -10, -np.pi)
pls3 = Pulse.ConstantAmplitude(1, cwf, -np.pi)
pls4 = Pulse.ConstantDetuning(bwf, -10, 0)


def test_creation():
    with pytest.raises(TypeError):
        Pulse(10, 0, 0, post_phase_shift=2)
        Pulse(cwf, 1, 0)
        Pulse(0, bwf, 1)
        Pulse(bwf, cwf, bwf)
        Pulse(bwf, cwf, 0, post_phase_shift=cwf)

    with pytest.raises(ValueError, match="The duration of"):
        Pulse(bwf, cwf, 0)

    with pytest.raises(ValueError, match="All samples of an amplitude"):
        Pulse(cwf, cwf, 0)
        Pulse.ConstantAmplitude(-1, cwf, 0)
        Pulse.ConstantPulse(100, -1, 0, 0)

    assert pls.phase == 0
    assert pls2 == pls3
    assert pls != pls4
    assert pls4.detuning != cwf
    assert pls4.amplitude == pls.amplitude


def test_str():
    assert pls2.__str__() == (
        "Pulse(Amp=1 rad/µs, Detuning=-10 rad/µs, " + "Phase=3.14)"
    )
    pls_ = Pulse(bwf, rwf, 1)
    msg = "Pulse(Amp=Blackman(Area: 3), Detuning=Ramp(0->1 rad/µs), Phase=1)"
    assert pls_.__str__() == msg


def test_repr():
    pls_ = Pulse(bwf, rwf, 1, post_phase_shift=-np.pi)
    msg = (
        "Pulse(amp=BlackmanWaveform(200 ns, Area: 3), "
        + "detuning=RampWaveform(200 ns, 0->1 rad/µs), "
        + "phase=1, post_phase_shift=3.14)"
    )
    assert pls_.__repr__() == msg


def test_draw(patch_plt_show):
    pls_ = Pulse.ConstantDetuning(bwf, -10, 1, post_phase_shift=-np.pi)
    pls_.draw()


@pytest.fixture
def eom_channel():
    eom_config = RydbergEOM(
        mod_bandwidth=24,
        max_limiting_amp=100,
        limiting_beam=RydbergBeam.RED,
        intermediate_detuning=700,
        controlled_beams=tuple(RydbergBeam),
    )
    return Rydberg.Global(None, None, mod_bandwidth=4, eom_config=eom_config)


def test_fall_time(eom_channel):
    assert eom_channel.eom_config.rise_time == 20
    assert eom_channel.rise_time == 120

    pulse = Pulse.ConstantPulse(1000, 1, 0, 0)
    assert pulse.fall_time(eom_channel, in_eom_mode=False) == 240
    assert pulse.fall_time(eom_channel, in_eom_mode=True) == 40


def test_full_duration(eom_channel):
    with pytest.raises(TypeError, match="must be a channel object instance"):
        pls.get_full_duration("eom_channel")

    channel1 = Rydberg.Global(None, None)
    assert not channel1.supports_eom()
    with pytest.raises(
        ValueError, match="does not support EOM mode operation"
    ):
        pls.get_full_duration(channel1, in_eom_mode=True)

    assert pls.get_full_duration(channel1) == pls.duration
    channel2 = dataclasses.replace(channel1, mod_bandwidth=4)
    assert pls.get_full_duration(channel2) == pls.duration + pls.fall_time(
        channel2
    )

    assert pls.get_full_duration(
        eom_channel, in_eom_mode=True
    ) == pls.duration + pls.fall_time(eom_channel, in_eom_mode=True)
