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

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pulser import Pulse
from pulser.parametrized import Variable
from pulser.waveforms import BlackmanWaveform, CompositeWaveform


a = Variable("a", float)
b = Variable("b", int, size=2)
b._assign([-1.5, 1.5])
c = Variable("c", str)
t = Variable("t", int)
bwf = BlackmanWaveform(t, a)
pulse = Pulse.ConstantDetuning(bwf, b[0], b[1])
pulse2 = Pulse(bwf, bwf, 1)


def test_var():
    with pytest.raises(TypeError, match="'name' has to be of type 'str'"):
        Variable(1, dtype=int)
    with pytest.raises(TypeError, match="Invalid data type"):
        Variable("x", dtype=list, size=4)
    with pytest.raises(TypeError, match="'size' is not of type 'int'"):
        Variable("x", dtype=float, size=(2, 2))
    with pytest.raises(ValueError, match="size 1 or larger"):
        Variable("x", dtype=str, size=0)
    x = Variable("x", dtype=float)
    assert x.value is None
    assert x._count == 0
    with pytest.raises(FrozenInstanceError):
        x.value = 0.5

    assert a.variables == {"a": a}
    assert len(b) == 2
    assert str(c) == "c"

    with pytest.raises(ValueError, match="to variable of size 2"):
        b._assign([1, 4, 5])
    assert np.all(b.build() == np.array([-1, 1]))
    assert b._count == 1

    with pytest.raises(ValueError, match="string to float"):
        a._assign("something")
    with pytest.raises(ValueError, match="No value"):
        a.build()

    with pytest.raises(TypeError, match="must be of type 'str'"):
        c._assign(3.14)

    d = Variable("d", str, size=2)
    d._assign(["o", "k"])
    assert np.all(d.build() == np.array(["o", "k"]))

    with pytest.raises(TypeError, match="Invalid key type"):
        b[[0, 1]]
    with pytest.raises(TypeError, match="not subscriptable"):
        a[0]
    with pytest.raises(IndexError):
        b[2]


def test_varitem():
    b1 = b[1]
    b01 = b[100::-1]
    assert b01.variables == {"b": b}
    assert str(b1) == "b[1]"
    assert str(b01) == "b[100::-1]"
    assert b1.build() == 1
    assert np.all(b01.build() == np.array([1, -1]))
    with pytest.raises(FrozenInstanceError):
        b1.key = 0


def test_paramobj():
    assert set(bwf.variables.keys()) == {"t", "a"}
    assert set(pulse.variables.keys()) == {"t", "a", "b"}
    assert str(bwf) == "BlackmanWaveform(t, a)"
    assert str(pulse) == f"Pulse.ConstantDetuning({str(bwf)}, b[0], b[1])"
    assert str(pulse2) == f"Pulse({str(bwf)}, {str(bwf)}, 1)"
    with pytest.raises(AttributeError):
        bwf._duration
    time = bwf.duration
    samps = bwf.samples
    cwf = CompositeWaveform(bwf, bwf)
    t._assign(1000)
    a._assign(np.pi)
    assert len(cwf.build().samples) == len(samps.build()) * 2
    assert time.build() == 1000


def test_opsupport():
    a._assign(-2.0)
    x = 5 + a
    x = b - x  # x = [-4, -2]
    x = x / 2
    x = 8 * x  # x = [-16, -8]
    x = -x // 3  # x = [5, 2]
    assert np.all(x.build() == [5.0, 2.0])
    assert (a ** a).build() == 0.25
    assert abs(a).build() == 2.0
    assert (3 % a).build() == -1.0
