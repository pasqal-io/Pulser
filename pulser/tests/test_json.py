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
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import (
    SquareLatticeLayout,
    TriangularLatticeLayout,
)
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


def test_layout():
    custom_layout = RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]])
    new_custom_layout = encode_decode(custom_layout)
    assert new_custom_layout == custom_layout
    assert type(new_custom_layout) is RegisterLayout

    tri_layout = TriangularLatticeLayout(100, 10)
    new_tri_layout = encode_decode(tri_layout)
    assert new_tri_layout == tri_layout
    assert isinstance(new_tri_layout, TriangularLatticeLayout)

    square_layout = SquareLatticeLayout(8, 10, 6)
    new_square_layout = encode_decode(square_layout)
    assert new_square_layout == square_layout
    assert isinstance(new_square_layout, SquareLatticeLayout)


def test_register_from_layout():
    layout = RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]])
    reg = layout.define_register(1, 0)
    assert reg == Register({"q0": [0, 1], "q1": [0, 0]})
    seq = Sequence(reg, device=MockDevice)
    new_reg = encode_decode(seq).register
    assert reg == new_reg
    assert new_reg._layout_info.layout == layout
    assert new_reg._layout_info.trap_ids == (1, 0)


def test_mappable_register():
    layout = RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]])
    mapp_reg = layout.make_mappable_register(2)
    new_mapp_reg = encode_decode(mapp_reg)
    assert new_mapp_reg._layout == layout
    assert new_mapp_reg.qubit_ids == ("q0", "q1")


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
