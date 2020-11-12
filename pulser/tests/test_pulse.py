from unittest.mock import patch

import numpy as np
import pytest

from pulser import Pulse
from pulser.waveforms import ConstantWaveform, BlackmanWaveform, RampWaveform

cwf = ConstantWaveform(100, -10)
bwf = BlackmanWaveform(200, 3)
rwf = RampWaveform(200, 0, 1)

pls = Pulse(bwf, bwf, 2*np.pi)
pls2 = Pulse.ConstantPulse(100, 1, -10, -np.pi)
pls3 = Pulse.ConstantAmplitude(1, cwf, 1)
pls4 = Pulse.ConstantDetuning(bwf, -10, 0)


def test_creation():
    with pytest.raises(TypeError):
        Pulse(10, 0, 0, post_phase_shift=2)
        Pulse(cwf, 1, 0)
        Pulse(0, bwf, 1)
        Pulse(bwf, cwf, bwf)
        Pulse(bwf, cwf, 0, post_phase_shift=cwf)

    with pytest.raises(ValueError, match="durations don't match"):
        Pulse(bwf, cwf, 0)

    with pytest.raises(ValueError, match="has always to be non-negative."):
        Pulse(cwf, cwf, 0)
        Pulse.ConstantAmplitude(-1, cwf, 0)
        Pulse.ConstantPulse(100, -1, 0, 0)

    assert pls.phase == 0
    assert pls2.amplitude == pls3.amplitude
    assert pls2.detuning == pls3.detuning
    assert pls2.phase == np.pi
    assert pls3.phase == 1
    assert pls4.detuning != cwf
    assert pls4.amplitude == pls.amplitude


def test_str():
    assert pls2.__str__() == "Pulse(Amp=1 MHz, Detuning=-10 MHz, Phase=3.14)"
    pls_ = Pulse(bwf, rwf, np.pi)
    msg = "Pulse(Amp=Blackman(Area: 3), Detuning=Ramp(0->1 MHz), Phase=3.14)"
    assert pls_.__str__() == msg


def test_repr():
    pls_ = Pulse(bwf, rwf, 1, post_phase_shift=-np.pi)
    msg = ("Pulse(amp=BlackmanWaveform(200 ns, Area: 3), " +
           "detuning=RampWaveform(200 ns, 0->1 MHz), " +
           "phase=1, post_phase_shift=3.14)")
    assert pls_.__repr__() == msg


def test_draw():
    pls_ = Pulse.ConstantDetuning(bwf, -10, 1, post_phase_shift=-np.pi)
    with patch('matplotlib.pyplot.show'):
        pls_.draw()
