import pytest
import numpy as np
from unittest.mock import patch

from pulser.waveforms import (ConstantWaveform, RampWaveform, GaussianWaveform,
                              BlackmanWaveform, ArbitraryWaveform,
                              CompositeWaveform)

np.random.seed(20201105)

constant = ConstantWaveform(100, -3)
ramp = RampWaveform(2e3, 5, 19)
arb_samples = np.random.random(50)
arbitrary = ArbitraryWaveform(arb_samples)
gaussian = GaussianWaveform(70, 10, 30, offset=5)
blackman = BlackmanWaveform(40, np.pi)
composite = CompositeWaveform(blackman, constant, arbitrary)


def test_duration():
    with pytest.raises(TypeError, match='needs to be castable to an int'):
        ConstantWaveform("s", -1)
        RampWaveform([0, 1, 3], 1, 0)
        GaussianWaveform((100,), 1, 50)

    with pytest.raises(ValueError, match='positive integer'):
        ConstantWaveform(0, -10)
        RampWaveform(-20, 3, 4)

    with pytest.warns(UserWarning):
        wf = BlackmanWaveform(np.pi, 1)

    assert wf.duration == 3
    assert arbitrary.duration == 50
    assert composite.duration == 190


def test_samples():
    assert np.all(constant.samples == -3)
    bm_samples = np.clip(np.blackman(40), 0, np.inf)
    bm_samples *= np.pi / np.sum(bm_samples) / 1e-3
    comp_samples = np.concatenate([bm_samples, np.full(100, -3), arb_samples])
    assert np.all(np.isclose(composite.samples, comp_samples))


def test_integral():
    assert np.isclose(blackman.integral, np.pi)
    assert constant.integral == -0.3
    assert ramp.integral == 24


def test_draw():
    with patch('matplotlib.pyplot.show'):
        composite.draw()
        gaussian.draw()


def test_eq():
    assert constant == ArbitraryWaveform(np.full(100, -3))
    assert constant != -3
    assert constant != ArbitraryWaveform(np.full(50, -3))


def test_composite():
    with pytest.raises(ValueError, match='Needs at least two waveforms'):
        CompositeWaveform()
        CompositeWaveform(composite)
        CompositeWaveform([blackman, arbitrary])
        CompositeWaveform(10)

    with pytest.raises(TypeError, match='not a valid waveform'):
        CompositeWaveform(composite, 'constant')

    assert composite.waveforms == [blackman, constant, arbitrary]

    wf = CompositeWaveform(blackman, arbitrary)
    wf.insert(constant, where=1)
    assert composite == wf

    wf = CompositeWaveform(blackman, constant)
    msg = ('BlackmanWaveform(40 ns, Area: 3.14), ' +
           'ConstantWaveform(100 ns, -3 MHz)')
    assert wf.__str__() == f'Composite({msg})'
    assert wf.__repr__() == f'CompositeWaveform(140 ns, [{msg}])'

    wf.append(arbitrary)
    assert composite == wf


def test_arbitrary():
    wf = ArbitraryWaveform([0, 1])
    assert wf.__str__() == 'Arbitrary'
    assert wf.__repr__() == 'ArbitraryWaveform(2 ns, array([0, 1]))'


def test_blackman():
    with pytest.raises(ValueError, match='Area under the waveform'):
        BlackmanWaveform(100, -100)
        BlackmanWaveform(10, 0)


def test_gaussian():
    with pytest.raises(ValueError, match='smaller than the offset'):
        GaussianWaveform(100, 1, 400, offset=2)

    with pytest.raises(ValueError, match='deviation has to be positive'):
        GaussianWaveform(100, 1, 0)
        GaussianWaveform(100, 1, -1)

    s = "Gaussian(5->10 MHz, sigma=30 ns)"
    assert gaussian.__str__() == s

    r = "GaussianWaveform(70 ns, max=10 MHz, offset=5 MHz, sigma=30 ns)"
    assert gaussian.__repr__() == r
