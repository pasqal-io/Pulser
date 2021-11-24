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

import numpy as np
import pytest

from pulser import Register, Register3D
from pulser.devices import Chadoq2


def test_creation():
    empty_dict = {}
    with pytest.raises(ValueError, match="Cannot create a Register with"):
        Register(empty_dict)

    coords = [(0, 0), (1, 0)]
    ids = ["q0", "q1"]
    qubits = dict(zip(ids, coords))
    with pytest.raises(TypeError):
        Register(coords)
        Register(ids)

    with pytest.raises(ValueError, match="vectors of size 2"):
        Register.from_coordinates([(0, 1, 0, 1)])

    with pytest.raises(
        NotImplementedError, match="a prefix and a set of labels"
    ):
        Register.from_coordinates(coords, prefix="a", labels=["a", "b"])

    with pytest.raises(ValueError, match="vectors of size 3"):
        Register3D.from_coordinates([((1, 0),), ((-1, 0),)])

    reg1 = Register(qubits)
    reg2 = Register.from_coordinates(coords, center=False, prefix="q")
    assert np.all(np.array(reg1._coords) == np.array(reg2._coords))
    assert reg1._ids == reg2._ids

    reg2b = Register.from_coordinates(coords, center=False, labels=["a", "b"])
    assert reg2b._ids == ["a", "b"]

    with pytest.raises(ValueError, match="Label length"):
        Register.from_coordinates(coords, center=False, labels=["a", "b", "c"])

    reg3 = Register.from_coordinates(np.array(coords), prefix="foo")
    coords_ = np.array([(-0.5, 0), (0.5, 0)])
    assert reg3._ids == ["foo0", "foo1"]
    assert np.all(reg3._coords == coords_)
    assert not np.all(coords_ == coords)

    reg4 = Register.rectangle(1, 2, spacing=1)
    assert np.all(reg4._coords == coords_)

    reg5 = Register.square(2, spacing=2)
    coords_ = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=float)
    assert np.all(np.array(reg5._coords) == coords_)

    reg6 = Register.triangular_lattice(2, 2, spacing=4)
    coords_ = np.array(
        [
            (-3, -np.sqrt(3)),
            (1, -np.sqrt(3)),
            (-1, np.sqrt(3)),
            (3, np.sqrt(3)),
        ]
    )
    assert np.all(np.array(reg6._coords) == coords_)


def test_rectangle():
    # Check rows
    with pytest.raises(ValueError, match="The number of rows"):
        Register.rectangle(0, 2)

    # Check columns
    with pytest.raises(ValueError, match="The number of columns"):
        Register.rectangle(2, 0)

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register.rectangle(2, 2, 0.0)


def test_square():
    # Check side
    with pytest.raises(ValueError, match="The number of atoms per side"):
        Register.square(0)

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register.square(2, 0.0)


def test_triangular_lattice():
    # Check rows
    with pytest.raises(ValueError, match="The number of rows"):
        Register.triangular_lattice(0, 2)

    # Check columns
    with pytest.raises(ValueError, match="The number of atoms per row"):
        Register.triangular_lattice(2, 0)

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register.triangular_lattice(2, 2, 0.0)


def test_hexagon():
    # Check number of layers
    with pytest.raises(ValueError, match="The number of layers"):
        Register.hexagon(0)

    # Check spacing
    with pytest.raises(ValueError, match="Spacing "):
        Register.hexagon(1, spacing=-1.0)

    # Check small hexagon (1 layer)
    reg = Register.hexagon(1, spacing=1.0)
    assert len(reg.qubits) == 7
    atoms = list(reg.qubits.values())
    crest_y = np.sqrt(3) / 2
    assert np.all(np.isclose(atoms[0], [0.0, 0.0]))
    assert np.all(np.isclose(atoms[1], [-0.5, crest_y]))
    assert np.all(np.isclose(atoms[2], [0.5, crest_y]))
    assert np.all(np.isclose(atoms[3], [1.0, 0.0]))
    assert np.all(np.isclose(atoms[4], [0.5, -crest_y]))
    assert np.all(np.isclose(atoms[5], [-0.5, -crest_y]))
    assert np.all(np.isclose(atoms[6], [-1.0, 0.0]))

    # Check a few atoms for a bigger hexagon (2 layers)
    reg = Register.hexagon(2, spacing=1.0)
    assert len(reg.qubits) == 19
    atoms = list(reg.qubits.values())
    crest_y = np.sqrt(3) / 2.0
    assert np.all(np.isclose(atoms[7], [-1.5, crest_y]))
    assert np.all(np.isclose(atoms[8], [-1.0, 2.0 * crest_y]))
    assert np.all(np.isclose(atoms[9], [-0.0, 2.0 * crest_y]))
    assert np.all(np.isclose(atoms[13], [1.5, -crest_y]))
    assert np.all(np.isclose(atoms[14], [1.0, -2.0 * crest_y]))
    assert np.all(np.isclose(atoms[15], [0.0, -2.0 * crest_y]))


def test_max_connectivity():
    device = Chadoq2
    max_atom_num = device.max_atom_num
    spacing = device.min_atom_distance
    crest_y = np.sqrt(3) / 2.0

    # Check device type
    with pytest.raises(TypeError):
        reg = Register.max_connectivity(2, None)

    # Check min number of atoms
    with pytest.raises(
        ValueError, match=r"The number of qubits(.+)greater than"
    ):
        reg = Register.max_connectivity(0, device)

    # Check max number of atoms
    with pytest.raises(ValueError, match=r"The number of qubits(.+)less than"):
        reg = Register.max_connectivity(max_atom_num + 1, device)

    # Check spacing
    reg = Register.max_connectivity(max_atom_num, device, spacing=spacing)
    with pytest.raises(ValueError, match="Spacing "):
        reg = Register.max_connectivity(
            max_atom_num, device, spacing=spacing - 1.0
        )

    # Check 1 atom
    reg = Register.max_connectivity(1, device)
    assert len(reg.qubits) == 1
    atoms = list(reg.qubits.values())
    assert np.all(np.isclose(atoms[0], [0.0, 0.0]))

    # Check for less than 7 atoms:
    for i in range(1, 7):
        hex_coords = np.array(
            [
                (0.0, 0.0),
                (-0.5, crest_y),
                (0.5, crest_y),
                (1.0, 0.0),
                (0.5, -crest_y),
                (-0.5, -crest_y),
            ]
        )
        reg = Register.max_connectivity(i, device)
        reg2 = Register.from_coordinates(
            spacing * hex_coords[:i], center=False
        )
        assert len(reg.qubits) == i
        atoms = list(reg.qubits.values())
        atoms2 = list(reg2.qubits.values())
        for k in range(i):
            assert np.all(np.isclose(atoms[k], atoms2[k]))

    # Check full layers on a small hexagon (1 layer)
    reg = Register.max_connectivity(7, device)
    assert len(reg.qubits) == 7
    atoms = list(reg.qubits.values())
    assert np.all(np.isclose(atoms[0], [0.0, 0.0]))
    assert np.all(np.isclose(atoms[1], [-0.5 * spacing, crest_y * spacing]))
    assert np.all(np.isclose(atoms[2], [0.5 * spacing, crest_y * spacing]))
    assert np.all(np.isclose(atoms[3], [1.0 * spacing, 0.0]))
    assert np.all(np.isclose(atoms[4], [0.5 * spacing, -crest_y * spacing]))
    assert np.all(np.isclose(atoms[5], [-0.5 * spacing, -crest_y * spacing]))
    assert np.all(np.isclose(atoms[6], [-1.0 * spacing, 0.0]))

    # Check full layers for a bigger hexagon (2 layers)
    reg = Register.max_connectivity(19, device)
    assert len(reg.qubits) == 19
    atoms = list(reg.qubits.values())
    assert np.all(np.isclose(atoms[7], [-1.5 * spacing, crest_y * spacing]))
    assert np.all(
        np.isclose(atoms[8], [-1.0 * spacing, 2.0 * crest_y * spacing])
    )
    assert np.all(np.isclose(atoms[13], [1.5 * spacing, -crest_y * spacing]))
    assert np.all(
        np.isclose(atoms[14], [1.0 * spacing, -2.0 * crest_y * spacing])
    )

    # Check extra atoms (2 full layers + 7 extra atoms)
    # for C3 symmetry, C6 symmetry and offset for next atoms
    reg = Register.max_connectivity(26, device)
    assert len(reg.qubits) == 26
    atoms = list(reg.qubits.values())
    assert np.all(np.isclose(atoms[19], [-2.5 * spacing, crest_y * spacing]))
    assert np.all(
        np.isclose(atoms[20], [-2.0 * spacing, 2.0 * crest_y * spacing])
    )
    assert np.all(
        np.isclose(atoms[21], [-0.5 * spacing, 3.0 * crest_y * spacing])
    )
    assert np.all(
        np.isclose(atoms[22], [2.0 * spacing, 2.0 * crest_y * spacing])
    )
    assert np.all(np.isclose(atoms[23], [2.5 * spacing, -crest_y * spacing]))
    assert np.all(
        np.isclose(atoms[24], [0.5 * spacing, -3.0 * crest_y * spacing])
    )
    assert np.all(
        np.isclose(atoms[25], [-2.0 * spacing, -2.0 * crest_y * spacing])
    )


def test_rotation():
    reg = Register.square(2, spacing=np.sqrt(2))
    reg.rotate(45)
    coords_ = np.array([(0, -1), (1, 0), (-1, 0), (0, 1)], dtype=float)
    assert np.all(np.isclose(reg._coords, coords_))


def test_drawing():
    with pytest.raises(ValueError, match="Blockade radius"):
        reg = Register.from_coordinates([(1, 0), (0, 1)])
        reg.draw(blockade_radius=0.0, draw_half_radius=True)

    reg = Register.from_coordinates([(1, 0), (0, 1)])
    with patch("matplotlib.pyplot.show"):
        reg.draw(blockade_radius=0.1, draw_graph=True)

    reg = Register.triangular_lattice(3, 8)
    with patch("matplotlib.pyplot.show"):
        reg.draw()

    with patch("matplotlib.pyplot.show"):
        with patch("matplotlib.pyplot.savefig"):
            reg.draw(fig_name="my_register.pdf")

    reg = Register.rectangle(1, 8)
    with patch("matplotlib.pyplot.show"):
        reg.draw(blockade_radius=5, draw_half_radius=True, draw_graph=True)

    with pytest.raises(ValueError, match="'blockade_radius' to draw."):
        reg.draw(draw_half_radius=True)

    reg = Register.square(1)
    with pytest.raises(NotImplementedError, match="Needs more than one atom"):
        reg.draw(blockade_radius=5, draw_half_radius=True)


def test_orthorombic():
    # Check rows
    with pytest.raises(ValueError, match="The number of rows"):
        Register3D.cuboid(0, 2, 2)

    # Check columns
    with pytest.raises(ValueError, match="The number of columns"):
        Register3D.cuboid(2, 0, 2)

    # Check layers
    with pytest.raises(ValueError, match="The number of layers"):
        Register3D.cuboid(2, 2, 0)

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register3D.cuboid(2, 2, 2, 0.0)


def test_cubic():
    # Check side
    with pytest.raises(ValueError, match="The number of atoms per side"):
        Register3D.cubic(0)

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register3D.cubic(2, 0.0)


def test_drawing3D():
    with pytest.raises(ValueError, match="Blockade radius"):
        reg = Register3D.from_coordinates([(1, 0, 0), (0, 0, 1)])
        reg.draw(blockade_radius=0.0)

    reg = Register3D.cubic(3, 8)
    with patch("matplotlib.pyplot.show"):
        with patch("matplotlib.pyplot.savefig"):
            reg.draw(fig_name="my_register.pdf")

    reg = Register3D.cuboid(1, 8, 2)
    with patch("matplotlib.pyplot.show"):
        reg.draw(blockade_radius=5, draw_half_radius=True, draw_graph=True)

    with pytest.raises(ValueError, match="'blockade_radius' to draw."):
        reg.draw(draw_half_radius=True)

    reg = Register3D.cuboid(2, 2, 2)
    with patch("matplotlib.pyplot.show"):
        reg.draw(
            blockade_radius=5,
            draw_half_radius=True,
            draw_graph=True,
            projection=False,
            with_labels=True,
        )
    with patch("matplotlib.pyplot.show"):
        reg.draw(
            blockade_radius=5,
            draw_half_radius=True,
            draw_graph=False,
            projection=True,
            with_labels=True,
        )

    reg = Register3D.cubic(1)
    with pytest.raises(NotImplementedError, match="Needs more than one atom"):
        reg.draw(blockade_radius=5, draw_half_radius=True)


def test_to_2D():
    reg = Register3D.cuboid(2, 2, 2)
    with pytest.raises(ValueError, match="Atoms are not coplanar"):
        reg.to_2D()
    reg.to_2D(tol_width=6)

    reg = Register3D.cuboid(2, 2, 1)
    reg.to_2D()
