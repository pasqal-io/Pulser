import pytest
import numpy as np

from pulser import Pulse
from pulser.waveforms import ConstantWaveform, BlackmanWaveform

cwf = ConstantWaveform(100, -10)
bwf = BlackmanWaveform(200, 3)


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

    pls = Pulse(bwf, bwf, 2*np.pi)
    pls2 = Pulse.ConstantPulse(100, 1, -10, -np.pi)
    pls3 = Pulse.ConstantAmplitude(1, cwf, 1)
    pls4 = Pulse.ConstantDetuning(bwf, -10, 0)
    assert pls.phase == 0
    assert pls2.amplitude == pls3.amplitude
    assert pls2.detuning == pls3.detuning
    assert pls2.phase == np.pi
    assert pls3.phase == 1
    assert pls4.detuning != cwf
    assert pls4.amplitude == pls.amplitude


def test_str():
    pls = Pulse.ConstantDetuning(bwf, -10, np.pi)
    str = "Pulse(Amp=Blackman(Area: 3), Detuning=-10 MHz, Phase=3.14)"
    assert pls.__str__() == str


def test_repr():
    pls = Pulse.ConstantDetuning(bwf, -10, 1, post_phase_shift=-np.pi)
    str = ("Pulse(amp=BlackmanWaveform(200 ns, Area: 3), " +
           "detuning=ConstantWaveform(200 ns, -10 MHz), " +
           "phase=1, post_phase_shift=3.14)")
    assert pls.__repr__() == str
