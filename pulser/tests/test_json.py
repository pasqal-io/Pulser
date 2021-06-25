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
from unittest.mock import patch

import numpy as np
import pytest

from pulser import Sequence, Register
from pulser.devices import Chadoq2
from pulser.json.coders import PulserEncoder, PulserDecoder
from pulser.parametrized.decorators import parametrize
from pulser.waveforms import BlackmanWaveform


def encode(obj):
    return json.dumps(obj, cls=PulserEncoder)


def decode(s):
    return json.loads(s, cls=PulserDecoder)


def encode_decode(obj):
    return decode(encode(obj))


def test_encoder():
    assert np.all(np.arange(10) == encode_decode(np.arange(10)))
    assert set(range(5)) == encode_decode(set(range(5)))
    with pytest.raises(TypeError, match="not JSON serializable"):
        encode(1j)


def test_rare_cases():
    reg = Register.square(4)
    seq = Sequence(reg, Chadoq2)
    var = seq.declare_variable("var")

    wf = BlackmanWaveform(100, var)
    with pytest.warns(
        UserWarning, match="Calls to methods of parametrized " "objects"
    ):
        s = encode(wf.draw())

    with pytest.warns(UserWarning, match="not encode a Sequence"):
        wf_ = Sequence.deserialize(s)

    var._assign(-10)
    with pytest.raises(ValueError, match="No value assigned"):
        wf_.build()

    var_ = wf_._variables["var"]
    var_._assign(-10)
    with patch("matplotlib.pyplot.show"):
        wf_.build()

    rotated_reg = parametrize(Register.rotate)(reg, var)
    with pytest.raises(NotImplementedError):
        encode(rotated_reg)
