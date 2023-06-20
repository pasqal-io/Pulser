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
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from pulser import Pulse
from pulser.json.coders import PulserDecoder, PulserEncoder
from pulser.parametrized import ParamObj, Variable
from pulser.waveforms import BlackmanWaveform, CompositeWaveform


@pytest.fixture
def a():
    return Variable("a", float)


@pytest.fixture
def b():
    b = Variable("b", int, size=2)
    b._assign([-1.5, 1.5])
    return b


@pytest.fixture
def d():
    d = Variable("d", float, size=1)
    d._assign([0.5])
    return d


@pytest.fixture
def t():
    return Variable("t", int)


@pytest.fixture
def bwf(t, a):
    return BlackmanWaveform(t[0], a[0])


def test_var(a, b):
    with pytest.raises(TypeError, match="'name' has to be of type 'str'"):
        Variable(1, dtype=int)
    with pytest.raises(TypeError, match="Invalid data type"):
        Variable("x", dtype=list, size=4)
    with pytest.raises(TypeError, match="'size' is not of type 'int'"):
        Variable("x", dtype=float, size=(2, 2))
    with pytest.raises(ValueError, match="size 1 or larger"):
        Variable("x", dtype=int, size=0)
    x = Variable("x", dtype=float)
    assert x.value is None
    assert x._count == 0
    with pytest.raises(FrozenInstanceError):
        x.value = 0.5

    assert a.variables == {"a": a}
    assert b.size == 2

    with pytest.raises(ValueError, match="to variable of size 2"):
        b._assign([1, 4, 5])
    assert np.all(b.build() == np.array([-1, 1]))
    assert b._count == 1

    with pytest.raises(ValueError, match="string to float"):
        a._assign("something")
    with pytest.raises(ValueError, match="No value"):
        a.build()

    var_ = Variable("var_", int, size=2)
    var_._assign([1, 2])
    assert np.all(var_.build() == np.array([1, 2]))

    with pytest.raises(TypeError, match="Invalid key type"):
        b[[0, 1]]
    with pytest.raises(IndexError):
        b[2]


def test_varitem(a, b, d):
    a0 = a[0]
    b1 = b[1]
    b01 = b[100::-1]
    d0 = d[0]
    assert b01.variables == {"b": b}
    assert str(a0) == "a[0]"
    assert str(b1) == "b[1]"
    assert str(b01) == "b[100::-1]"
    assert str(d0) == "d[0]"
    assert b1.build() == 1
    assert np.all(b01.build() == np.array([1, -1]))
    assert d0.build() == 0.5
    with pytest.raises(FrozenInstanceError):
        b1.key = 0


def test_paramobj(bwf, t, a, b):
    assert set(bwf.variables.keys()) == {"t", "a"}
    pulse = Pulse.ConstantDetuning(bwf, b[0], b[1])
    assert set(pulse.variables.keys()) == {"t", "a", "b"}
    assert str(bwf) == "BlackmanWaveform(t[0], a[0])"
    assert str(pulse) == f"Pulse.ConstantDetuning({str(bwf)}, b[0], b[1])"
    pulse2 = Pulse(bwf, bwf, 1)
    assert str(pulse2) == f"Pulse({str(bwf)}, {str(bwf)}, 1)"
    with pytest.raises(AttributeError):
        bwf._duration
    cwf = CompositeWaveform(bwf, bwf)
    t._assign(1000)
    a._assign(np.pi)
    assert len(cwf.build().samples) == len(bwf.build().samples) * 2
    assert bwf.build().duration == 1000

    param_poly = ParamObj(Polynomial, b)
    with pytest.warns(
        UserWarning, match="Calls to methods of parametrized objects"
    ):
        origin = param_poly(0)
    b._assign((0, 1))
    assert origin.build() == 0.0


def test_opsupport(a, b):
    a._assign(-2.0)
    u = 5 + a
    u = b - u  # u = [-4, -2]
    u = u / 2
    u = 8 * u  # u = [-16, -8]
    u = -u // 3  # u = [5, 2]
    assert np.all(u.build() == [5.0, 2.0])

    v = a**a
    assert v.build() == 0.25
    v = abs(-v * 8)
    assert v.build() == 2.0
    v = 3 % v
    assert v.build() == 1.0
    v = -v
    assert v.build() == -1.0

    x = a + 11
    assert x.build() == 9
    x = x % 6
    assert x.build() == 3
    x = 2 - x
    assert x.build() == -1
    x = 4 / x
    assert x.build() == -4
    x = 9 // x
    assert x.build() == -3
    x = 2**x
    assert x.build() == 0.125
    x = np.log2(x)
    assert x.build() == -3.0

    # Trigonometric functions
    pi = -a * np.pi / 2
    x = np.sin(pi)
    np.testing.assert_almost_equal(x.build(), 0.0)
    x = np.cos(pi)
    np.testing.assert_almost_equal(x.build(), -1.0)
    x = np.tan(pi / 4)
    np.testing.assert_almost_equal(x.build(), 1.0)

    # Other transcendentals
    y = np.exp(b)
    np.testing.assert_almost_equal(y.build(), [1 / np.e, np.e])
    y = np.log(y)
    np.testing.assert_almost_equal(y.build(), b.build())
    y_ = y + 0.4  # y_ = [-0.6, 1.4]
    y = np.round(y_, 1)
    np.testing.assert_array_equal(y.build(), np.round(y_.build(), 1))
    np.testing.assert_array_equal(round(y_).build(), np.round(y_).build())
    np.testing.assert_array_equal(round(y_, 1).build(), y.build())

    y = round(y)
    np.testing.assert_array_equal(y.build(), [-1.0, 1.0])
    y = np.floor(y + 0.1)
    np.testing.assert_array_equal(y.build(), [-1.0, 1.0])
    y = np.ceil(y + 0.1)
    np.testing.assert_array_equal(y.build(), [0.0, 2.0])
    y = np.sqrt((y - 1) ** 2)
    np.testing.assert_array_equal(y.build(), [1.0, 1.0])

    # Test serialization support for operations
    def encode_decode(obj):
        return json.loads(
            json.dumps(obj, cls=PulserEncoder), cls=PulserDecoder
        )

    # Will raise a SerializationError if they fail
    u2 = encode_decode(u)
    assert set(u2.variables) == {"a", "b"}
    u2.variables["a"]._assign(a.value)
    u2.variables["b"]._assign(b.value)
    np.testing.assert_array_equal(u2.build(), u.build())

    v2 = encode_decode(v)
    assert list(v2.variables) == ["a"]
    v2.variables["a"]._assign(a.value)
    assert v2.build() == v.build()

    x2 = encode_decode(x)
    assert list(x2.variables) == ["a"]
    x2.variables["a"]._assign(a.value)
    assert x2.build() == x.build()

    y2 = encode_decode(y)
    assert list(y2.variables) == ["b"]
    y2.variables["b"]._assign(b.value)
    np.testing.assert_array_equal(y2.build(), y.build())
