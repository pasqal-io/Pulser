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

import re
from hashlib import sha256
from unittest.mock import patch

import numpy as np
import pytest

from pulser.register import Register, Register3D
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import (
    SquareLatticeLayout,
    TriangularLatticeLayout,
)


@pytest.fixture
def layout():
    return RegisterLayout([[0, 0], [1, 1], [1, 0], [0, 1]], slug="2DLayout")


@pytest.fixture
def layout3d():
    return RegisterLayout([[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]])


def test_creation(layout, layout3d):
    with pytest.raises(
        ValueError, match="must be an array or list of coordinates"
    ):
        RegisterLayout([[0, 0, 0], [1, 1], [1, 0], [0, 1]])

    with pytest.raises(
        ValueError, match="must be an array or list of coordinates"
    ):
        RegisterLayout([0, 1, 2])

    with pytest.raises(ValueError, match="size 2 or 3"):
        RegisterLayout([[0], [1], [2]])

    with pytest.raises(
        ValueError,
        match="All trap coordinates of a register layout must be unique.",
    ):
        RegisterLayout([[0, 1], [0.0, 1.0]])

    assert np.all(layout.coords == [[0, 0], [0, 1], [1, 0], [1, 1]])
    assert np.all(
        layout3d.coords == [[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]
    )
    assert layout.number_of_traps == 4
    assert layout.dimensionality == 2
    for i, coord in enumerate(layout.coords):
        assert np.all(layout.traps_dict[i] == coord)


def test_slug(layout, layout3d):
    assert layout.slug == "2DLayout"
    assert layout3d.slug is None
    assert str(layout) == "2DLayout"
    assert str(layout3d) == repr(layout3d)


def test_register_definition(layout, layout3d):
    with pytest.raises(ValueError, match="must be a unique integer"):
        layout.define_register(0, 1, 1)

    with pytest.raises(ValueError, match="correspond to the ID of a trap"):
        layout.define_register(0, 4, 3)

    with pytest.raises(ValueError, match="must be a sequence of unique IDs"):
        layout.define_register(0, 1, qubit_ids=["a", "b", "b"])

    with pytest.raises(ValueError, match="must have the same size"):
        layout.define_register(0, 1, qubit_ids=["a", "b", "c"])

    assert layout.define_register(0, 1) == Register.from_coordinates(
        [[0, 0], [0, 1]], prefix="q", center=False
    )

    assert layout3d.define_register(0, 1) == Register3D(
        {"q0": [0, 0, 0], "q1": [0, 1, 0]}
    )

    reg2d = layout.define_register(0, 2)
    assert reg2d._layout_info == (layout, (0, 2))
    with pytest.raises(ValueError, match="dimensionality is not the same"):
        reg2d._validate_layout(layout3d, (0, 2))
    with pytest.raises(
        ValueError, match="Every 'trap_id' must be a unique integer"
    ):
        reg2d._validate_layout(layout, (0, 2, 2))
    with pytest.raises(
        ValueError, match="must be equal to the number of atoms"
    ):
        reg2d._validate_layout(layout, (0,))
    with pytest.raises(
        ValueError, match="don't match this register's coordinates"
    ):
        reg2d._validate_layout(layout, (0, 1))

    with pytest.raises(TypeError, match="cannot be rotated"):
        reg2d.rotate(30)


def test_draw(layout, layout3d, patch_plt_show):
    layout.draw()

    with patch("matplotlib.pyplot.savefig"):
        layout.draw(fig_name="my_registerlayout.pdf")

    layout3d.draw()

    layout3d.draw(projection=False)

    with patch("matplotlib.pyplot.savefig"):
        layout3d.draw(fig_name="my_registerlayout.pdf")


def test_repr(layout):
    hash_ = sha256(bytes(2))
    hash_.update(layout.coords.tobytes())
    assert repr(layout) == f"RegisterLayout_{hash_.hexdigest()}"


def test_static_hash(layout):
    int_hash = int.from_bytes(layout._safe_hash(), byteorder="big")
    assert layout.static_hash() == f"{int_hash:x}"
    assert repr(layout) == f"RegisterLayout_{layout.static_hash()}"


def test_eq(layout, layout3d):
    assert RegisterLayout([[0, 0], [1, 0]]) != Register.from_coordinates(
        [[0, 0], [1, 0]]
    )
    assert layout != layout3d
    layout1 = RegisterLayout([[0, 0], [1, 0]])
    layout2 = RegisterLayout([[1, 0], [0, 0]])
    assert layout1 == layout2
    assert hash(layout1) == hash(layout2)


def test_traps_from_coordinates(layout):
    assert layout._coords_to_traps == {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3,
    }
    assert layout.get_traps_from_coordinates(
        (0.9999995, 0.0000004), (0, 1), (1, 1)
    ) == [2, 1, 3]
    with pytest.raises(ValueError, match="not a part of the RegisterLayout"):
        layout.get_traps_from_coordinates((0.9999994, 1))


def test_square_lattice_layout():
    square = SquareLatticeLayout(9, 7, 5)
    assert str(square) == "SquareLatticeLayout(9x7, 5.0µm)"
    assert square.square_register(3) == Register.square(
        3, spacing=5, prefix="q"
    )
    # An even number of atoms on the side won't align the center with an atom
    assert square.square_register(4) != Register.square(
        4, spacing=5, prefix="q"
    )
    with pytest.raises(ValueError, match="'8x8' array doesn't fit"):
        square.square_register(8)

    assert square.rectangular_register(3, 7, prefix="r") == Register.rectangle(
        3, 7, spacing=5, prefix="r"
    )
    with pytest.raises(ValueError, match="'10x3' array doesn't fit"):
        square.rectangular_register(10, 3)


def test_triangular_lattice_layout():
    tri = TriangularLatticeLayout(50, 5)
    assert str(tri) == "TriangularLatticeLayout(50, 5.0µm)"

    assert tri.hexagonal_register(19) == Register.hexagon(
        2, spacing=5, prefix="q"
    )
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The desired register has more atoms (51) than there"
            " are traps in this TriangularLatticeLayout (50)"
        ),
    ):
        tri.hexagonal_register(51)

    with pytest.raises(
        ValueError, match="has more atoms than there are traps"
    ):
        tri.rectangular_register(7, 8)

    # Case where the register doesn't fit
    with pytest.raises(ValueError, match="not a part of the RegisterLayout"):
        tri.rectangular_register(8, 3)

    # But this fits fine, though off-centered with the Register default
    assert tri.rectangular_register(5, 5) != Register.triangular_lattice(
        5, 5, spacing=5, prefix="q"
    )


def test_mappable_register_creation():
    tri = TriangularLatticeLayout(50, 5)
    with pytest.raises(ValueError, match="greater than the number of traps"):
        tri.make_mappable_register(51)

    mapp_reg = tri.make_mappable_register(5)
    assert mapp_reg.qubit_ids == ("q0", "q1", "q2", "q3", "q4")

    assert mapp_reg.find_indices(
        {"q2", "q4", "q1"}, ["q4", "q2", "q1", "q2"]
    ) == [2, 1, 0, 1]

    with pytest.raises(
        ValueError, match="must be selected among pre-declared qubit IDs"
    ):
        mapp_reg.find_indices(
            {"q2", "q4", "q1", "q5"}, ["q4", "q2", "q1", "q2"]
        )

    with pytest.raises(
        ValueError, match="must be selected among the chosen IDs"
    ):
        mapp_reg.find_indices(
            {"q2", "q4", "q1"}, ["q4", "q2", "q1", "q2", "q3"]
        )

    with pytest.raises(
        ValueError, match="labeled with pre-declared qubit IDs"
    ):
        mapp_reg.build_register({"q0": 0, "q5": 2})

    qubit_map = {"q0": 10, "q1": 49}
    reg = mapp_reg.build_register(qubit_map)
    assert reg == Register(
        {"q0": tri.traps_dict[10], "q1": tri.traps_dict[49]}
    )
    names = ["q1", "q0", "q0"]
    assert mapp_reg.find_indices(qubit_map.keys(), names) == reg.find_indices(
        names
    )
