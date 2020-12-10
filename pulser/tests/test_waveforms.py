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

from pulser.waveforms import (ConstantWaveform, RampWaveform, BlackmanWaveform,
                              CustomWaveform, CompositeWaveform)

np.random.seed(20201105)

constant = ConstantWaveform(100, -3)
ramp = RampWaveform(2e3, 5, 19)
arb_samples = np.random.random(50)
custom = CustomWaveform(arb_samples)
blackman = BlackmanWaveform(40, np.pi)
composite = CompositeWaveform(blackman, constant, custom)


def test_duration():
    with pytest.raises(TypeError, match='needs to be castable to an int'):
        ConstantWaveform("s", -1)
        RampWaveform([0, 1, 3], 1, 0)

    with pytest.raises(ValueError, match='positive integer'):
        ConstantWaveform(0, -10)
        RampWaveform(-20, 3, 4)

    with pytest.warns(UserWarning):
        wf = BlackmanWaveform(np.pi, 1)

    assert wf.duration == 3
    assert custom.duration == 50
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
        blackman.draw()


def test_eq():
    assert constant == CustomWaveform(np.full(100, -3))
    assert constant != -3
    assert constant != CustomWaveform(np.full(50, -3))


def test_composite():
    with pytest.raises(ValueError, match='Needs at least two waveforms'):
        CompositeWaveform()
        CompositeWaveform(composite)
        CompositeWaveform([blackman, custom])
        CompositeWaveform(10)

    with pytest.raises(TypeError, match='not a valid waveform'):
        CompositeWaveform(composite, 'constant')

    assert composite.waveforms == [blackman, constant, custom]

    wf = CompositeWaveform(blackman, custom)
    wf.insert(constant, where=1)
    assert composite == wf

    wf = CompositeWaveform(blackman, constant)
    msg = ('BlackmanWaveform(40 ns, Area: 3.14), ' +
           'ConstantWaveform(100 ns, -3 MHz)')
    assert wf.__str__() == f'Composite({msg})'
    assert wf.__repr__() == f'CompositeWaveform(140 ns, [{msg}])'

    wf.append(custom)
    assert composite == wf


def test_custom():
    wf = CustomWaveform([0, 1])
    assert wf.__str__() == 'Custom'
    assert wf.__repr__() == 'CustomWaveform(2 ns, array([0, 1]))'


def test_blackman():
    with pytest.raises(ValueError, match='Area under the waveform'):
        BlackmanWaveform(100, -100)
        BlackmanWaveform(10, 0)
