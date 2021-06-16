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

import json
import re
from unittest.mock import patch

import numpy as np
import pytest

from pulser.json.coders import PulserEncoder, PulserDecoder
from pulser.parametrized import Variable, ParamObj
from pulser.waveforms import (ConstantWaveform, RampWaveform, BlackmanWaveform,
                              CustomWaveform, CompositeWaveform)

np.random.seed(20201105)

constant = ConstantWaveform(100, -3)
ramp = RampWaveform(2000, 5, 19)
arb_samples = np.random.random(52)
custom = CustomWaveform(arb_samples)
blackman = BlackmanWaveform(40, np.pi)
composite = CompositeWaveform(blackman, constant, custom)


def test_duration():
    with pytest.raises(TypeError, match='needs to be castable to an int'):
        ConstantWaveform("s", -1)
        RampWaveform([0, 1, 3], 1, 0)

    with pytest.raises(ValueError, match='positive duration'):
        ConstantWaveform(15, -10)
        RampWaveform(-20, 3, 4)

    with pytest.warns(UserWarning):
        wf = BlackmanWaveform(np.pi*10, 1)

    assert wf.duration == 31
    assert custom.duration == 52
    assert composite.duration == 192


def test_change_duration():
    with pytest.raises(NotImplementedError):
        custom.change_duration(53)

    new_cte = constant.change_duration(103)
    assert constant.duration == 100
    assert new_cte.duration == 103

    new_blackman = blackman.change_duration(30)
    assert np.isclose(new_blackman.integral, blackman.integral)
    assert new_blackman != blackman

    new_ramp = ramp.change_duration(100)
    assert new_ramp.duration == 100
    assert new_ramp != ramp


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
    assert constant != CustomWaveform(np.full(48, -3))


def test_first_last():
    assert constant.first_value == constant.last_value
    assert ramp.first_value == 5
    assert ramp.last_value == 19
    assert blackman.first_value == 0
    assert blackman.last_value == 0
    assert composite.first_value == 0
    assert composite.last_value == arb_samples[-1]
    assert custom.first_value == arb_samples[0]


def test_hash():
    assert hash(constant) == hash(tuple(np.full(100, -3)))
    assert hash(ramp) == hash(tuple(np.linspace(5, 19, num=2000)))


def test_composite():
    with pytest.raises(ValueError, match='Needs at least two waveforms'):
        CompositeWaveform()
        CompositeWaveform(composite)
        CompositeWaveform([blackman, custom])
        CompositeWaveform(10)

    with pytest.raises(TypeError, match='not a valid waveform'):
        CompositeWaveform(composite, 'constant')

    assert composite.waveforms == [blackman, constant, custom]

    wf = CompositeWaveform(blackman, constant)
    msg = ('BlackmanWaveform(40 ns, Area: 3.14), ' +
           'ConstantWaveform(100 ns, -3 rad/µs)')
    assert wf.__str__() == f'Composite({msg})'
    assert wf.__repr__() == f'CompositeWaveform(140 ns, [{msg}])'


def test_custom():
    data = np.arange(16, dtype=float)
    wf = CustomWaveform(data)
    assert wf.__str__() == 'Custom'
    assert wf.__repr__() == f'CustomWaveform(16 ns, {data!r})'


def test_ramp():
    assert ramp.slope == 7e-3


def test_blackman():
    with pytest.raises(TypeError):
        BlackmanWaveform(100, np.array([1, 2]))
    wf = BlackmanWaveform(100, -2)
    assert np.isclose(wf.integral, -2)
    assert np.all(wf.samples <= 0)
    assert wf == BlackmanWaveform(100, np.array([-2]))

    with pytest.raises(ValueError, match="matching signs"):
        BlackmanWaveform.from_max_val(-10, np.pi)

    wf = BlackmanWaveform.from_max_val(10, 2*np.pi)
    assert np.isclose(wf.integral, 2*np.pi)
    assert np.max(wf.samples) < 10

    wf = BlackmanWaveform.from_max_val(-10, -np.pi)
    assert np.isclose(wf.integral, -np.pi)
    assert np.min(wf.samples) > -10

    var = Variable("var", float)
    wf_var = BlackmanWaveform.from_max_val(-10, var)
    assert isinstance(wf_var, ParamObj)
    var._assign(-np.pi)
    assert wf_var.build() == wf


def test_ops():
    assert -constant == ConstantWaveform(100, 3)
    assert ramp * 2 == RampWaveform(2e3, 10, 38)
    assert --custom == custom
    assert blackman / 2 == BlackmanWaveform(40, np.pi / 2)
    assert composite * 1 == composite
    with pytest.raises(ZeroDivisionError):
        constant / 0


def test_serialization():
    for wf in [constant, ramp, custom, blackman, composite]:
        s = json.dumps(wf, cls=PulserEncoder)
        assert wf == json.loads(s, cls=PulserDecoder)


def test_get_item():

    # Check errors raised

    duration = constant.duration
    with pytest.raises(IndexError,
                       match=re.escape("Index ('index_or_slice' = "
                                       f"{duration}) must be in the range "
                                       f"0~{duration-1}, or "
                                       f"{-duration}~-1 from the end.")):
        constant[duration]
    with pytest.raises(IndexError,
                       match=re.escape("Index ('index_or_slice' = "
                                       f"{-duration-1}) must be in the range "
                                       f"0~{duration-1}, or "
                                       f"{-duration}~-1 from the end.")):
        constant[-duration - 1]

    with pytest.raises(IndexError,
                       match="The step of the slice must be None or 1."):
        constant[0:1:2]
    with pytest.raises(IndexError,
                       match=re.escape("The start of the slice (0) "
                                       "must be less than the stop (0).")):
        constant[0:0]
    with pytest.raises(IndexError,
                       match=re.escape("The range of the slice "
                                       f"(0~{duration+1}) "
                                       "must be included in the range "
                                       "of the waveform.")):
        constant[:duration+1]
    with pytest.raises(IndexError,
                       match=re.escape("The range of the slice (-10~10) "
                                       "must be included in the range "
                                       "of the waveform.")):
        constant[-duration-10:10]

    # Check nominal operations

    for wf in [blackman, composite, constant, custom, ramp]:
        duration = wf.duration
        duration14 = duration // 4
        duration34 = duration * 3 // 4
        samples = wf.samples

        # Check with int index
        for i in range(-duration, duration):
            assert wf[i] == samples[i]

        # Check with slices

        if wf == composite:
            # Exhaustive tests for composite waveforms
            for i in range(0, duration):
                for j in range(i+1, duration):
                    assert (wf[i:j] == samples[i:j]).all()
        else:
            # Limited tests for other waveforms
            assert (wf[0:duration] == samples).all()
            assert (wf[0:-1] == samples[0:-1]).all()
            assert (wf[0:] == samples).all()
            assert (wf[-1:] == samples[-1:]).all()
            assert (wf[:duration] == samples).all()
            assert (wf[:] == samples).all()
            assert (wf[duration14:duration34] ==
                    samples[duration14:duration34]).all()
            assert (wf[-duration34:-duration14] ==
                    samples[-duration34:-duration14]).all()
