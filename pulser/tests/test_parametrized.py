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
from pulser.parametrized import Parametrized, ParamObj, Variable
from pulser.waveforms import BlackmanWaveform, CompositeWaveform

import numpy as np
import pytest

a = Variable("a", float)
b = Variable("b", int, size=2)
b._assign([-1.5, 1.5])
c = Variable("c", str)


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

    c._assign(3.14)
    assert c.build() == "3.14"

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
