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

import numpy as np
import pytest

from pulser import Register, Register3D, Sequence
from pulser.devices import Chadoq2, MockDevice
from pulser.json.coders import PulserDecoder, PulserEncoder
from pulser.json.exceptions import SerializationError
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


def test_device(mod_device):
    assert encode_decode(Chadoq2) == Chadoq2
    with pytest.raises(SerializationError):
        encode_decode(mod_device)


def test_virtual_device(mod_device):
    assert encode_decode(MockDevice) == MockDevice
    virtual_mod = mod_device.to_virtual()
    assert encode_decode(virtual_mod) == virtual_mod


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
    assert type(new_tri_layout) is TriangularLatticeLayout

    square_layout = SquareLatticeLayout(8, 10, 6)
    new_square_layout = encode_decode(square_layout)
    assert new_square_layout == square_layout
    assert type(new_square_layout) is SquareLatticeLayout


def test_register_from_layout():
    layout = RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]])
    reg = layout.define_register(1, 0)
    assert reg == Register({"q0": [0, 1], "q1": [0, 0]})
    seq = Sequence(reg, device=MockDevice)
    new_reg = encode_decode(seq).register
    assert reg == new_reg
    assert new_reg.layout == layout
    assert new_reg._layout_info.trap_ids == (1, 0)


@pytest.mark.parametrize(
    "reg",
    [
        Register(dict(enumerate([(2, 3), (5, 1), (10, 0)]))),
        Register3D({3: (2, 3, 4), 4: (3, 4, 5), 2: (4, 5, 7)}),
    ],
)
def test_register_numbered_keys(reg):
    j = json.dumps(reg, cls=PulserEncoder)
    decoded_reg = json.loads(j, cls=PulserDecoder)
    assert reg == decoded_reg
    assert all([type(i) == int for i in decoded_reg.qubit_ids])


def test_mappable_register():
    layout = RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]])
    mapp_reg = layout.make_mappable_register(2)
    new_mapp_reg = encode_decode(mapp_reg)
    assert new_mapp_reg.layout == layout
    assert new_mapp_reg.qubit_ids == ("q0", "q1")

    seq = Sequence(mapp_reg, MockDevice)
    assert seq.is_register_mappable()
    mapped_seq = seq.build(qubits={"q0": 2, "q1": 1})
    assert not mapped_seq.is_register_mappable()
    new_mapped_seq = Sequence.deserialize(mapped_seq.serialize())
    assert not new_mapped_seq.is_register_mappable()


def test_rare_cases(patch_plt_show):
    reg = Register.square(4)
    seq = Sequence(reg, Chadoq2)
    var = seq.declare_variable("var")

    wf = BlackmanWaveform(var * 100 // 10, var)
    with pytest.warns(
        UserWarning, match="Calls to methods of parametrized objects"
    ), pytest.raises(
        ValueError, match="Serialization of calls to parametrized objects"
    ):
        s = encode(wf())
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
        SerializationError,
        match="No serialization support for module 'pulser.fake'.",
    ):
        validate_serialization(obj_dict)

    wf_obj_dict = obj_dict["__args__"][0]
    wf_obj_dict["__submodule__"] = "RampWaveform"
    with pytest.raises(
        SerializationError,
        match="No serialization support for attributes of "
        "'pulser.waveforms.RampWaveform'",
    ):
        validate_serialization(wf_obj_dict)

    del wf_obj_dict["__submodule__"]
    with pytest.raises(
        SerializationError,
        match="No serialization support for 'pulser.waveforms.from_max_val'",
    ):
        validate_serialization(wf_obj_dict)


def test_sequence_module():
    # Check that the sequence module is backwards compatible after refactoring
    seq = Sequence(Register.square(2), Chadoq2)

    obj_dict = json.loads(seq.serialize())
    assert obj_dict["__module__"] == "pulser.sequence"

    # Defensively check that the standard format runs
    Sequence.deserialize(seq.serialize())

    # Use module being used in v0.7.0-0.7.2.0
    obj_dict["__module__"] == "pulser.sequence.sequence"

    # Check that it also works
    s = json.dumps(obj_dict)
    Sequence.deserialize(s)
