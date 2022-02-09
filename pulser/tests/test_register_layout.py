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
from hashlib import sha256

import numpy as np
import pytest

from pulser.register.register_layout import RegisterLayout
from pulser.register import Register3D, Register

layout = RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]])
layout3d = RegisterLayout([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]])


def test_creation():
    with pytest.raises(
        ValueError, match="must be an array or list of coordinates"
    ):
        RegisterLayout([[0, 0, 0], [1, 1], [1, 0], [0, 1]])

    with pytest.raises(ValueError, match="size 2 or 3"):
        RegisterLayout([[0], [1], [2]])

    assert np.all(layout.coords == [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.all(
        layout3d.coords == [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]
    )
    assert layout.number_of_traps == 4
    assert layout.max_atom_num == 2
    assert layout.dimensionality == 2
    for i, coord in enumerate(layout.coords):
        assert np.all(layout.traps_dict[i] == coord)


def test_register_definition():
    with pytest.raises(ValueError, match="must be a unique integer"):
        layout.define_register(0, 1, 1)

    with pytest.raises(ValueError, match="correspond to the ID of a trap"):
        layout.define_register(0, 4, 3)

    with pytest.raises(ValueError, match="must be a sequence of unique IDs"):
        layout.define_register(0, 1, qubit_ids=["a", "b", "b"])

    with pytest.raises(ValueError, match="must have the same size"):
        layout.define_register(0, 1, qubit_ids=["a", "b", "c"])

    with pytest.raises(
        ValueError, match="greater than the maximum number of qubits"
    ):
        layout.define_register(0, 1, 3)

    assert layout.define_register(0, 1) == Register.from_coordinates(
        [[0, 0], [0, 1]], prefix="q", center=False
    )

    assert layout3d.define_register(0, 1) == Register3D(
        {"q0": [0, 0, 0], "q1": [0, 1, 0]}
    )

    reg2d = layout.define_register(0, 2)
    assert reg2d._layout_info == (layout, (0, 2))
    with pytest.raises(ValueError, match="dimensionality is not the same"):
        reg2d._set_layout(layout3d, 0, 2)
    with pytest.raises(
        ValueError, match="must be equal to the number of atoms"
    ):
        reg2d._set_layout(layout, 0)
    with pytest.raises(
        ValueError, match="don't match this register's coordinates"
    ):
        reg2d._set_layout(layout, 0, 1)


def test_draw():
    with patch("matplotlib.pyplot.show"):
        layout.draw()

    with patch("matplotlib.pyplot.show"):
        layout3d.draw()

    with patch("matplotlib.pyplot.show"):
        layout3d.draw(projection=False)


def test_repr():
    hash_ = sha256(bytes(2))
    hash_.update(layout.coords.tobytes())
    assert repr(layout) == f"RegisterLayout_{hash_.hexdigest()}"


def test_eq():
    assert RegisterLayout([[0, 0], [1, 0]]) != Register.from_coordinates(
        [[0, 0], [1, 0]]
    )
    assert layout != layout3d
    layout1 = RegisterLayout([[0, 0], [1, 0]])
    layout2 = RegisterLayout([[1, 0], [0, 0]])
    assert layout1 == layout2
    assert hash(layout1) == hash(layout2)
