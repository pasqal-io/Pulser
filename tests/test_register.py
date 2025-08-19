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
from __future__ import annotations

import dataclasses
from unittest.mock import patch

import numpy as np
import pytest

from pulser import Register, Register3D
from pulser.devices import AnalogDevice, DigitalAnalogDevice, MockDevice
from pulser.register import RegisterLayout


def test_creation():
    empty_dict = {}
    with pytest.raises(ValueError, match="Cannot create a Register with"):
        Register(empty_dict)

    coords = [(0, 0), (1, 0)]
    ids = ("q0", "q1")
    qubits = dict(zip(ids, coords))
    with pytest.raises(TypeError):
        Register(coords)
        Register(ids)

    with pytest.raises(ValueError, match="vectors of size 2"):
        Register.from_coordinates([(0, 1, 0, 1)], prefix="q")

    with pytest.raises(
        NotImplementedError, match="a prefix and a set of labels"
    ):
        Register.from_coordinates(coords, prefix="a", labels=["a", "b"])

    with pytest.raises(ValueError, match="vectors of size 3"):
        Register3D.from_coordinates([((1, 0),), ((-1, 0),)], prefix="q")

    reg1 = Register(qubits)
    reg2 = Register.from_coordinates(coords, center=False, prefix="q")
    assert np.all(np.array(reg1._coords) == np.array(reg2._coords))
    assert reg2.sorted_coords.dtype == np.float64
    assert reg1._ids == reg2._ids

    reg2b = Register.from_coordinates(coords, center=False, labels=["a", "b"])
    assert reg2b._ids == ("a", "b")

    with pytest.raises(ValueError, match="Label length"):
        Register.from_coordinates(coords, center=False, labels=["a", "b", "c"])

    reg3 = Register.from_coordinates(
        np.array(coords, dtype=np.float32), prefix="foo"
    )
    coords_ = np.array([(-0.5, 0), (0.5, 0)])
    assert reg3._ids == ("foo0", "foo1")
    # Data is always stored as float -> float64 for numpy, float32 for torch
    assert (reg3._coords[0].dtype, reg3._coords[1].dtype) == (
        np.float64,
        np.float64,
    )
    assert reg3.sorted_coords.dtype == np.float64
    assert np.all(reg3._coords == coords_)
    assert not np.all(coords_ == coords)

    reg4 = Register.rectangle(1, 2, spacing=1, prefix="q")
    assert np.all(reg4._coords == coords_)

    reg5 = Register.square(2, spacing=2, prefix="q")
    coords_ = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=float)
    assert np.all(np.array(reg5._coords) == coords_)

    reg6 = Register.triangular_lattice(2, 2, spacing=4, prefix="q")
    coords_ = np.array(
        [
            (-3, -np.sqrt(3)),
            (1, -np.sqrt(3)),
            (-1, np.sqrt(3)),
            (3, np.sqrt(3)),
        ]
    )
    assert np.all(np.array(reg6._coords) == coords_)

    with pytest.raises(
        ValueError, match="must only be 'layout' and 'trap_ids'"
    ):
        Register(qubits, spacing=10, layout="square", trap_ids=(0, 1, 3))

    with pytest.warns(
        DeprecationWarning,
        match="Usage of `int`s or any non-`str`types as `QubitId`s",
    ):
        Register.from_coordinates([(0, 0)])


def test_repr():
    assert (
        repr(Register(dict(q0=(1.0, 0.0), q1=(-1, 5))))
        == "Register({'q0': array([1., 0.]), 'q1': array([-1.,  5.])})"
    )
    assert (
        repr(Register3D(dict(q0=(1, 2, 3))))
        == "Register3D({'q0': array([1., 2., 3.])})"
    )


def test_rectangular_lattice():
    # Check rows
    with pytest.raises(ValueError, match="The number of rows"):
        Register.rectangular_lattice(0, 2, 3, 4)

    # Check columns
    with pytest.raises(ValueError, match="The number of columns"):
        Register.rectangular_lattice(2, 0, 3, 4)

    # Check row spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register.rectangular_lattice(2, 2, 0.0, 5)

    # Check col spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register.rectangular_lattice(2, 2, 3, 0.0)


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
    reg = Register.hexagon(1, spacing=1.0, prefix="q")
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
    reg = Register.hexagon(2, spacing=1.0, prefix="q")
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
    device = DigitalAnalogDevice
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
    reg = Register.max_connectivity(
        max_atom_num, device, spacing=spacing, prefix="q"
    )
    with pytest.raises(ValueError, match="Spacing "):
        Register.max_connectivity(max_atom_num, device, spacing=spacing - 1.0)
    reg = Register.max_connectivity(
        max_atom_num, MockDevice, spacing=spacing, prefix="q"
    )
    with pytest.raises(
        NotImplementedError,
        match="Maximum connectivity layouts are not well defined for a "
        "device with 'min_atom_distance=0.0'.",
    ):
        Register.max_connectivity(1e9, MockDevice)

    # Check 1 atom
    reg = Register.max_connectivity(1, device, prefix="q")
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
        reg = Register.max_connectivity(i, device, prefix="q")
        device.validate_register(reg)
        reg2 = Register.from_coordinates(
            spacing * hex_coords[:i], center=False, prefix="q"
        )
        assert len(reg.qubits) == i
        atoms = list(reg.qubits.values())
        atoms2 = list(reg2.qubits.values())
        for k in range(i):
            assert np.all(np.isclose(atoms[k], atoms2[k]))

    # Check full layers on a small hexagon (1 layer)
    reg = Register.max_connectivity(7, device, prefix="q")
    device.validate_register(reg)
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
    reg = Register.max_connectivity(19, device, prefix="q")
    device.validate_register(reg)
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
    reg = Register.max_connectivity(26, device, prefix="q")
    device.validate_register(reg)
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
    reg = Register.square(2, spacing=np.sqrt(2), prefix="q")
    rot_reg = reg.rotated(45)
    new_coords_ = np.array([(0, -1), (1, 0), (-1, 0), (0, 1)], dtype=float)
    np.testing.assert_allclose(
        rot_reg._coords_arr.as_array(), new_coords_, atol=1e-15
    )
    assert rot_reg != reg


draw_params = [
    dict(),
    dict(qubit_colors=dict()),
    dict(qubit_colors={1: "darkred"}),
]


@pytest.mark.parametrize("draw_params", draw_params)
def test_drawing(draw_params, patch_plt_show):
    with pytest.raises(ValueError, match="Blockade radius"):
        reg = Register.from_coordinates([(1, 0), (0, 1)], prefix="q")
        reg.draw(blockade_radius=0.0, draw_half_radius=True, **draw_params)

    reg = Register.from_coordinates([(1, 0), (0, 1)], prefix="q")
    reg.draw(blockade_radius=0.1, draw_graph=True, **draw_params)
    with pytest.raises(ValueError, match="The register must have"):
        reg.draw(draw_empty_sites=True)

    reg = Register.triangular_lattice(3, 8, prefix="q")
    reg.draw(**draw_params)

    with patch("matplotlib.pyplot.savefig"):
        reg.draw(fig_name="my_register.pdf")

    reg = Register.rectangle(1, 8, prefix="q")
    reg.draw(
        blockade_radius=5,
        draw_half_radius=True,
        draw_graph=True,
        **draw_params,
    )

    with pytest.raises(ValueError, match="'blockade_radius' to draw."):
        reg.draw(draw_half_radius=True, **draw_params)

    reg = Register.square(1, prefix="q")
    with pytest.raises(NotImplementedError, match="Needs more than one atom"):
        reg.draw(blockade_radius=5, draw_half_radius=True, **draw_params)

    reg = RegisterLayout(
        [[0, 0], [1, 1], [1, 0], [0, 1]], slug="2DLayout"
    ).define_register(0)
    reg.draw(draw_empty_sites=True)


def test_orthorombic():
    # Check rows
    with pytest.raises(ValueError, match="The number of rows"):
        Register3D.cuboid(0, 2, 2, prefix="q")

    # Check columns
    with pytest.raises(ValueError, match="The number of columns"):
        Register3D.cuboid(2, 0, 2, prefix="q")

    # Check layers
    with pytest.raises(ValueError, match="The number of layers"):
        Register3D.cuboid(2, 2, 0, prefix="q")

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register3D.cuboid(2, 2, 2, 0.0, prefix="q")


def test_cubic():
    # Check side
    with pytest.raises(ValueError, match="The number of atoms per side"):
        Register3D.cubic(0, prefix="q")

    # Check spacing
    with pytest.raises(ValueError, match="Spacing"):
        Register3D.cubic(2, 0.0, prefix="q")


@pytest.mark.parametrize("draw_params", draw_params)
def test_drawing3D(draw_params, patch_plt_show):
    with pytest.raises(ValueError, match="Blockade radius"):
        reg = Register3D.from_coordinates([(1, 0, 0), (0, 0, 1)], prefix="q")
        reg.draw(blockade_radius=0.0, **draw_params)

    reg = Register3D.cubic(3, 8, prefix="q")

    with patch("matplotlib.pyplot.savefig"):
        reg.draw(fig_name="my_register.pdf", **draw_params)

    reg = Register3D.cuboid(1, 8, 2, prefix="q")
    reg.draw(
        blockade_radius=5,
        draw_half_radius=True,
        draw_graph=True,
        **draw_params,
    )

    with pytest.raises(ValueError, match="'blockade_radius' to draw."):
        reg.draw(draw_half_radius=True, **draw_params)

    reg = Register3D.cuboid(2, 2, 2, prefix="q")
    reg.draw(
        blockade_radius=5,
        draw_half_radius=True,
        draw_graph=True,
        projection=False,
        with_labels=True,
        **draw_params,
    )
    reg.draw(
        blockade_radius=5,
        draw_half_radius=True,
        draw_graph=False,
        projection=True,
        with_labels=True,
        **draw_params,
    )

    reg = Register3D.cubic(1, prefix="q")
    with pytest.raises(NotImplementedError, match="Needs more than one atom"):
        reg.draw(blockade_radius=5, draw_half_radius=True, **draw_params)


def test_to_2D():
    reg = Register3D.cuboid(2, 2, 2, prefix="q")
    with pytest.raises(ValueError, match="Atoms are not coplanar"):
        reg.to_2D()
    reg.to_2D(tol_width=6)

    reg = Register3D.cuboid(2, 2, 1, prefix="q")
    reg.to_2D()


def test_find_indices():
    reg = Register(dict(a=(0, 0), c=(5, 0), b=(0, 5)))
    assert reg.find_indices(["c", "b", "a"]) == [1, 2, 0]

    with pytest.raises(
        ValueError,
        match="IDs list must be selected among the IDs of the register's "
        "qubits",
    ):
        reg.find_indices(["c", "e", "d"])


def assert_eq(left, right):
    assert left == right
    assert right == left


def assert_ineq(left, right):
    assert left != right
    assert right != left


def test_equality_function():
    reg1 = Register({"c": (1, 2), "d": (8, 4)})
    assert_eq(reg1, reg1)
    assert_eq(reg1, Register({"c": (1, 2), "d": (8, 4)}))
    assert_ineq(reg1, Register({"d": (8, 4), "c": (1, 2)}))
    assert_ineq(reg1, Register({"c": (8, 4), "d": (1, 2)}))
    assert_ineq(reg1, Register({"c": (1, 2), "d": (8, 4), "e": (8, 4)}))
    assert_ineq(reg1, 10)

    reg2 = Register3D({"a": (1, 2, 3), "b": (8, 5, 6)})
    assert_eq(reg2, reg2)
    assert_eq(reg2, Register3D({"a": (1, 2, 3), "b": (8, 5, 6)}))
    assert_eq(reg2, Register3D({"a": (1, 2, 3), "b": (8, 5, 6)}))
    assert_ineq(reg2, Register3D({"b": (8, 5, 6), "a": (1, 2, 3)}))
    assert_ineq(reg2, Register3D({"b": (1, 2, 3), "a": (8, 5, 6)}))
    assert_ineq(
        reg2, Register3D({"a": (1, 2, 3), "b": (8, 5, 6), "e": (8, 5, 6)})
    )
    assert_ineq(reg2, 10)

    assert_ineq(reg1, reg2)


def test_coords_hash():
    reg1 = Register.square(2, prefix="foo")
    reg2 = Register.rectangle(2, 2, prefix="bar")
    assert reg1 != reg2  # Ids are different
    coords1 = list(c.as_array() for c in reg1.qubits.values())
    coords2 = list(c.as_array() for c in reg2.qubits.values())
    np.testing.assert_equal(coords1, coords2)  # But coords are the same
    assert reg1.coords_hex_hash() == reg2.coords_hex_hash()

    # Same coords but in inverse order
    reg3 = Register.from_coordinates(coords1[::-1], prefix="q")
    assert reg1.coords_hex_hash() == reg3.coords_hex_hash()

    # Modify a coordinate below precision
    coords1[0][0] += 1e-10
    reg4 = Register.from_coordinates(coords1, prefix="q")
    assert reg1.coords_hex_hash() == reg4.coords_hex_hash()

    # Modify a coordinate above precision
    coords1[0][1] += 1e-6
    reg5 = Register.from_coordinates(coords1, prefix="q")
    assert reg1.coords_hex_hash() != reg5.coords_hex_hash()


def _assert_reg_requires_grad(
    reg: Register | Register3D, invert: bool = False
) -> None:
    for coords in reg.qubits.values():
        if invert:
            assert not coords.requires_grad
        else:
            assert coords.is_tensor and coords.requires_grad


@pytest.mark.parametrize(
    "register_type, coords",
    [
        (Register, [[1, -4], [0, 0]]),
        (Register3D, [[1, -4, 5], [0, 0, 0]]),
    ],
)
def test_custom_register_torch(register_type, coords, patch_plt_show):
    torch = pytest.importorskip("torch")

    diff_qubit = torch.tensor(coords[0], requires_grad=True, dtype=float)

    reg1 = register_type({"q0": diff_qubit, "q1": coords[1]})
    reg2 = register_type.from_coordinates(
        [diff_qubit, coords[1]], center=False, prefix="q"
    )
    assert reg1 == reg2

    # Also check that centering keeps the grad
    reg3 = register_type.from_coordinates(
        [diff_qubit, coords[1]], center=True, prefix="q"
    )
    assert torch.all(reg3.qubits["q0"].as_tensor() == diff_qubit / 2)

    for r in [reg1, reg2, reg3]:
        _assert_reg_requires_grad(r)
        if r.dimensionality == 2:
            # Check after rotation
            _assert_reg_requires_grad(r.rotated(30))
        else:
            # Check after conversion to 2D
            _assert_reg_requires_grad(r.to_2D(0.1))

        # Check that drawing still works too
        r.draw()

    # check that generating with long type works
    reg4 = register_type.from_coordinates(
        [torch.tensor(coord, dtype=torch.long) for coord in coords], prefix="q"
    )
    assert reg4 == reg3


@pytest.mark.parametrize(
    "reg_classmethod, param_name, extra_params",
    [
        (Register.square, "spacing", {"side": 2}),
        (Register.rectangle, "spacing", {"rows": 1, "columns": 3}),
        (
            Register.rectangular_lattice,
            "row_spacing",
            {"rows": 1, "columns": 3},
        ),
        (
            Register.rectangular_lattice,
            "col_spacing",
            {"rows": 1, "columns": 3},
        ),
        (
            Register.triangular_lattice,
            "spacing",
            {"rows": 3, "atoms_per_row": 5},
        ),
        (Register.hexagon, "spacing", {"layers": 5}),
        (
            Register.max_connectivity,
            "spacing",
            {"n_qubits": 20, "device": DigitalAnalogDevice},
        ),
        (Register3D.cubic, "spacing", {"side": 3}),
        (Register3D.cuboid, "spacing", {"rows": 4, "columns": 2, "layers": 5}),
    ],
)
@pytest.mark.parametrize("requires_grad", [True, False])
def test_register_recipes_torch(
    reg_classmethod, param_name, extra_params, requires_grad
):
    torch = pytest.importorskip("torch")
    kwargs = {
        param_name: torch.tensor(6.0, requires_grad=requires_grad),
        "prefix": "q",
        **extra_params,
    }
    reg = reg_classmethod(**kwargs)
    _assert_reg_requires_grad(reg, invert=not requires_grad)


@pytest.mark.parametrize("optimal_filling", [None, 0.4, 0.1])
@pytest.mark.parametrize(
    "reg, max_atom_num",
    [
        (Register.square(4, spacing=5, prefix="test"), 20),
        (
            Register.from_coordinates(
                [
                    np.array([-2.501571, -0.003283]),
                    np.array([2.50157, 0.003283]),
                    np.array([-2.501571, 5.0]),
                    np.array([2.50157, 5.1]),
                ],
                prefix="q",
            ),
            8,
        ),
        (
            Register.from_coordinates(
                [
                    np.array([-2.501571, -0.003283], dtype=np.float32),
                    np.array([2.50157, 0.003283], dtype=np.float32),
                    np.array([-2.501571, 5.0], dtype=np.float32),
                    np.array([2.50157, 5.1], dtype=np.float32),
                ],
                prefix="q",
            ),
            8,
        ),
        (
            Register.from_coordinates(
                [
                    np.array([-2.501571, -0.003283], dtype=np.float32),
                    np.array([2.50157, 0.003283], dtype=np.float32),
                    np.array([-2.501571, 5.0], dtype=np.float64),
                    np.array([2.50157, 5.1], dtype=np.float64),
                ],
                prefix="q",
            ),
            8,
        ),
    ],
)
def test_automatic_layout(optimal_filling, reg, max_atom_num):
    max_layout_filling = 0.5
    min_traps = int(np.ceil(len(reg.qubits) / max_layout_filling))
    optimal_traps = int(
        np.ceil(len(reg.qubits) / (optimal_filling or max_layout_filling))
    )
    device = dataclasses.replace(
        AnalogDevice,
        max_atom_num=max_atom_num,
        max_layout_filling=max_layout_filling,
        optimal_layout_filling=optimal_filling,
        pre_calibrated_layouts=(),
    )
    device.validate_register(reg)

    # On its own, it works
    new_reg = reg.with_automatic_layout(device, layout_slug="foo")
    assert new_reg.qubit_ids == reg.qubit_ids  # Same IDs in the same order
    assert new_reg == reg  # The register itself is identical
    assert isinstance(new_reg.layout, RegisterLayout)
    assert str(new_reg.layout) == "foo"
    trap_num = new_reg.layout.number_of_traps
    assert min_traps <= trap_num <= optimal_traps
    # To test the device limits on trap number are enforced
    if not optimal_filling:
        assert trap_num == min_traps
        bound_below_dev = dataclasses.replace(
            device, min_layout_traps=trap_num + 1
        )
        assert (
            reg.with_automatic_layout(bound_below_dev).layout.number_of_traps
            == bound_below_dev.min_layout_traps
        )
    else:
        assert trap_num > min_traps
        bound_above_dev = dataclasses.replace(
            device,
            max_layout_traps=trap_num - 1,
            # So that we can still fit 20 atoms
            max_layout_filling=device.max_layout_filling + 0.4,
        )
        assert (
            reg.with_automatic_layout(bound_above_dev).layout.number_of_traps
            == bound_above_dev.max_layout_traps
        )
        # If we set min_layout_filling to the optimal filling, we should end
        # up with the optimal number of traps (because there can't be more)
        bound_above_from_min_filling = dataclasses.replace(
            device, min_layout_filling=optimal_filling
        )
        assert bound_above_from_min_filling.min_layout_filling > 0.0
        assert (
            reg.with_automatic_layout(
                bound_above_from_min_filling
            ).layout.number_of_traps
            == optimal_traps
        )

        # However, if the maximum number of traps allowed by min_layout_filling
        # matches the minimum number of traps allowed the constraint is not
        # imposed and we end up back to the original trap number
        not_bound_above_from_min_filling = dataclasses.replace(
            bound_above_from_min_filling, min_layout_traps=optimal_traps
        )
        assert (
            reg.with_automatic_layout(
                not_bound_above_from_min_filling
            ).layout.number_of_traps
            == trap_num
        )

    with pytest.raises(TypeError, match="must be of type Device"):
        reg.with_automatic_layout(MockDevice)

    # Minimum number of traps is too high
    with pytest.raises(RuntimeError, match="Failed to find a site"):
        reg.with_automatic_layout(
            dataclasses.replace(device, min_layout_traps=200)
        )

    # The Register is larger than max_traps
    big_reg = Register.square(8, spacing=5, prefix="q")
    min_traps = np.ceil(len(big_reg.qubit_ids) / max_layout_filling)
    with pytest.raises(
        RuntimeError, match="Failed to find a site for 2 traps"
    ):
        big_reg.with_automatic_layout(
            dataclasses.replace(device, max_layout_traps=int(min_traps - 2))
        )
    # Without max_traps, it would still work
    assert (
        big_reg.with_automatic_layout(device).layout.number_of_traps
        >= min_traps
    )


def test_automatic_layout_diff():
    torch = pytest.importorskip("torch")
    with pytest.raises(
        NotImplementedError,
        match="does not support registers with differentiable coordinates",
    ):
        Register.square(
            2, spacing=torch.tensor(10.0, requires_grad=True), prefix="q"
        ).with_automatic_layout(AnalogDevice)
