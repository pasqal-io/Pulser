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

from pulser import Sequence, Pulse, Register
from pulser.devices import Chadoq2
from pulser._json_coders import PulserEncoder, PulserDecoder


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
    seq = Sequence(Register.square(4), Chadoq2)
    var = seq.declare_variable("var")
    pls = Pulse.ConstantPulse(100, 10, var, 0)
    s = encode(pls.draw())
    with pytest.warns(UserWarning, match="not encode a Sequence"):
        pls_ = Sequence.deserialize(s)

    var._assign(-10)
    with pytest.raises(ValueError, match="No value assigned"):
        pls_.build()

    var_ = pls_._variables["var"]
    var_._assign(-10)
    with patch('matplotlib.pyplot.show'):
        pls_.build()
