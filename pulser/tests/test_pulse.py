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

from unittest.mock import patch

import numpy as np
import pytest

from pulser import Pulse
from pulser.waveforms import ConstantWaveform, BlackmanWaveform, RampWaveform

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


def test_draw():
    pls_ = Pulse.ConstantDetuning(bwf, -10, 1, post_phase_shift=-np.pi)
    with patch("matplotlib.pyplot.show"):
        pls_.draw()
