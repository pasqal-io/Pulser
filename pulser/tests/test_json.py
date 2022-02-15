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

from pulser import Register, Register3D, Sequence
from pulser.devices import Chadoq2, MockDevice
from pulser.json.coders import PulserDecoder, PulserEncoder
from pulser.json.supported import validate_serialization
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


def test_register_2d():
    reg = Register({"c": (1, 2), "d": (8, 4)})
    seq = Sequence(reg, device=Chadoq2)
    assert reg == encode_decode(seq).register


def test_register_3d():
    reg = Register3D({"a": (1, 2, 3), "b": (8, 5, 6)})
    seq = Sequence(reg, device=MockDevice)
    assert reg == encode_decode(seq).register


def test_rare_cases():
    reg = Register.square(4)
    seq = Sequence(reg, Chadoq2)
    var = seq.declare_variable("var")

    wf = BlackmanWaveform(var * 100 // 10, var)
    with pytest.raises(
        ValueError, match="Serialization of calls to parametrized objects"
    ):
        s = encode(wf.draw())
    s = encode(wf)

    with pytest.raises(ValueError, match="not encode a Sequence"):
        wf_ = Sequence.deserialize(s)

    wf_ = decode(s)
    seq._variables["var"]._assign(-10)
    with pytest.raises(ValueError, match="No value assigned"):
        wf_.build()

    var_ = wf_._variables["var"]
    var_._assign(10)
    assert wf_.build() == BlackmanWaveform(100, 10)
    with pytest.warns(UserWarning, match="Serialization of 'getattr'"):
        draw_func = wf_.draw
    with patch("matplotlib.pyplot.show"):
        with pytest.warns(
            UserWarning, match="Calls to methods of parametrized objects"
        ):
            draw_func().build()

    rotated_reg = parametrize(Register.rotate)(reg, var)
    with pytest.raises(NotImplementedError):
        encode(rotated_reg)


def test_support():
    seq = Sequence(Register.square(2), Chadoq2)
    var = seq.declare_variable("var")

    obj_dict = BlackmanWaveform.from_max_val(1, var)._to_dict()
    del obj_dict["__module__"]
    with pytest.raises(TypeError, match="Invalid 'obj_dict'."):
        validate_serialization(obj_dict)

    obj_dict["__module__"] = "pulser.fake"
    with pytest.raises(
        SystemError, match="No serialization support for module 'pulser.fake'."
    ):
        validate_serialization(obj_dict)

    wf_obj_dict = obj_dict["__args__"][0]
    wf_obj_dict["__submodule__"] = "RampWaveform"
    with pytest.raises(
        SystemError,
        match="No serialization support for attributes of "
        "'pulser.waveforms.RampWaveform'",
    ):
        validate_serialization(wf_obj_dict)

    del wf_obj_dict["__submodule__"]
    with pytest.raises(
        SystemError,
        match="No serialization support for 'pulser.waveforms.from_max_val'",
    ):
        validate_serialization(wf_obj_dict)
