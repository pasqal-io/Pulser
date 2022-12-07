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
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

import pulser
from pulser.channels import Microwave, Rydberg
from pulser.devices import Chadoq2, Device, VirtualDevice
from pulser.register import Register, Register3D
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import TriangularLatticeLayout


@pytest.fixture
def test_params():
    return dict(
        name="Test",
        dimensions=2,
        rydberg_level=70,
        _channels=(),
        min_atom_distance=1,
        max_atom_num=None,
        max_radial_distance=None,
    )


@pytest.mark.parametrize(
    "param, value, msg",
    [
        ("name", 1, None),
        ("supports_slm_mask", 0, None),
        ("reusable_channels", "true", None),
        ("max_atom_num", 1e9, None),
        ("max_radial_distance", 100.4, None),
        ("rydberg_level", 70.0, "Rydberg level has to be an int."),
        (
            "_channels",
            ((1, Rydberg.Global(None, None)),),
            "All channel IDs must be of type 'str', not 'int'",
        ),
        (
            "_channels",
            (("ch1", "Rydberg.Global(None, None)"),),
            "All channels must be of type 'Channel', not 'str'",
        ),
        (
            "_channels",
            (("mw_ch", Microwave.Global(None, None)),),
            "When the device has a 'Microwave' channel, "
            "'interaction_coeff_xy' must be a 'float',"
            " not '<class 'NoneType'>'.",
        ),
    ],
)
def test_post_init_type_checks(test_params, param, value, msg):
    test_params[param] = value
    error_msg = msg or f"{param} must be of type"
    with pytest.raises(TypeError, match=error_msg):
        VirtualDevice(**test_params)


@pytest.mark.parametrize(
    "param, value, msg",
    [
        (
            "dimensions",
            1,
            re.escape("'dimensions' must be one of (2, 3), not 1."),
        ),
        ("rydberg_level", 49, "Rydberg level should be between 50 and 100."),
        ("rydberg_level", 101, "Rydberg level should be between 50 and 100."),
        (
            "min_atom_distance",
            -0.001,
            "'min_atom_distance' must be greater than or equal to zero",
        ),
        ("max_atom_num", 0, None),
        ("max_radial_distance", 0, None),
    ],
)
def test_post_init_value_errors(test_params, param, value, msg):
    test_params[param] = value
    error_msg = msg or f"When defined, '{param}' must be greater than zero"
    with pytest.raises(ValueError, match=error_msg):
        VirtualDevice(**test_params)


potential_params = ("max_atom_num", "max_radial_distance")


@pytest.mark.parametrize("none_param", potential_params)
def test_optional_parameters(test_params, none_param):
    test_params.update({p: 10 for p in potential_params})
    test_params[none_param] = None
    with pytest.raises(
        TypeError,
        match=f"'{none_param}' can't be None in a 'Device' instance.",
    ):
        Device(**test_params)
    VirtualDevice(**test_params)  # Valid as None on a VirtualDevice


def test_tuple_conversion(test_params):
    test_params["_channels"] = (
        ["rydberg_global", Rydberg.Global(None, None)],
    )
    dev = VirtualDevice(**test_params)
    assert dev._channels == (("rydberg_global", Rydberg.Global(None, None)),)


def test_valid_devices():
    for dev in pulser.devices._valid_devices:
        assert dev.dimensions in (2, 3)
        assert dev.rydberg_level > 49
        assert dev.rydberg_level < 101
        assert dev.max_atom_num > 10
        assert dev.max_radial_distance > 10
        assert dev.min_atom_distance > 0
        assert dev.interaction_coeff > 0
        assert isinstance(dev.channels, dict)
        with pytest.raises(FrozenInstanceError):
            dev.name = "something else"
    assert Chadoq2 in pulser.devices._valid_devices
    assert Chadoq2.supported_bases == {"digital", "ground-rydberg"}
    with patch("sys.stdout"):
        Chadoq2.print_specs()
    assert Chadoq2.__repr__() == "Chadoq2"


def test_change_rydberg_level():
    dev = pulser.devices.MockDevice
    dev.change_rydberg_level(60)
    assert dev.rydberg_level == 60
    assert np.isclose(dev.interaction_coeff, 865723.02)
    with pytest.raises(TypeError, match="Rydberg level has to be an int."):
        dev.change_rydberg_level(70.5)
    with pytest.raises(
        ValueError, match="Rydberg level should be between 50 and 100."
    ):
        dev.change_rydberg_level(110)
    dev.change_rydberg_level(70)

    with pytest.warns(DeprecationWarning):
        assert pulser.__version__ < "0.9"
        og_ryd_level = Chadoq2.rydberg_level
        Chadoq2.change_rydberg_level(60)
        assert Chadoq2.rydberg_level == 60
        Chadoq2.change_rydberg_level(og_ryd_level)


def test_rydberg_blockade():
    dev = pulser.devices.MockDevice
    assert np.isclose(dev.rydberg_blockade_radius(3 * np.pi), 9.119201)
    assert np.isclose(dev.rabi_from_blockade(9), 10.198984)
    rand_omega = np.random.rand() * 2 * np.pi
    assert np.isclose(
        rand_omega,
        dev.rabi_from_blockade(dev.rydberg_blockade_radius(rand_omega)),
    )


def test_validate_register():
    with pytest.raises(ValueError, match="The number of atoms"):
        Chadoq2.validate_register(Register.square(50))

    coords = [(100, 0), (-100, 0)]
    with pytest.raises(TypeError):
        Chadoq2.validate_register(coords)
    with pytest.raises(ValueError, match="at most 50 μm away from the center"):
        Chadoq2.validate_register(Register.from_coordinates(coords))

    with pytest.raises(ValueError, match="at most 2D vectors"):
        coords = [(-10, 4, 0), (0, 0, 0)]
        Chadoq2.validate_register(Register3D(dict(enumerate(coords))))

    with pytest.raises(ValueError, match="The minimal distance between atoms"):
        Chadoq2.validate_register(
            Register.triangular_lattice(3, 4, spacing=3.9)
        )

    with pytest.raises(
        ValueError, match="associated with an incompatible register layout"
    ):
        tri_layout = TriangularLatticeLayout(201, 5)
        Chadoq2.validate_register(tri_layout.hexagonal_register(10))

    Chadoq2.validate_register(Register.rectangle(5, 10, spacing=5))


def test_validate_layout():
    with pytest.raises(ValueError, match="The number of traps"):
        Chadoq2.validate_layout(RegisterLayout(Register.square(20)._coords))

    coords = [(100, 0), (-100, 0)]
    with pytest.raises(TypeError):
        Chadoq2.validate_layout(Register.from_coordinates(coords))
    with pytest.raises(ValueError, match="at most 50 μm away from the center"):
        Chadoq2.validate_layout(RegisterLayout(coords))

    with pytest.raises(ValueError, match="at most 2 dimensions"):
        coords = [(-10, 4, 0), (0, 0, 0)]
        Chadoq2.validate_layout(RegisterLayout(coords))

    with pytest.raises(ValueError, match="The minimal distance between traps"):
        Chadoq2.validate_layout(
            TriangularLatticeLayout(12, Chadoq2.min_atom_distance - 1e-6)
        )

    valid_layout = RegisterLayout(
        Register.square(int(np.sqrt(Chadoq2.max_atom_num * 2)))._coords
    )
    Chadoq2.validate_layout(valid_layout)

    valid_tri_layout = TriangularLatticeLayout(
        Chadoq2.max_atom_num * 2, Chadoq2.min_atom_distance
    )
    Chadoq2.validate_layout(valid_tri_layout)


def test_calibrated_layouts():
    with pytest.raises(ValueError, match="The number of traps"):
        Device(
            name="TestDevice",
            dimensions=2,
            rydberg_level=70,
            max_atom_num=100,
            max_radial_distance=50,
            min_atom_distance=4,
            _channels=(),
            pre_calibrated_layouts=(TriangularLatticeLayout(201, 5),),
        )

    TestDevice = Device(
        name="TestDevice",
        dimensions=2,
        rydberg_level=70,
        max_atom_num=100,
        max_radial_distance=50,
        min_atom_distance=4,
        _channels=(),
        pre_calibrated_layouts=(
            TriangularLatticeLayout(100, 6.8),  # Rounds down with int()
            TriangularLatticeLayout(200, 5),
        ),
    )
    assert TestDevice.calibrated_register_layouts.keys() == {
        "TriangularLatticeLayout(100, 6µm)",
        "TriangularLatticeLayout(200, 5µm)",
    }


def test_device_with_virtual_channel():
    with pytest.raises(
        ValueError,
        match="A 'Device' instance cannot contain virtual channels.",
    ):
        Device(
            name="TestDevice",
            dimensions=2,
            rydberg_level=70,
            max_atom_num=100,
            max_radial_distance=50,
            min_atom_distance=4,
            _channels=(("rydberg_global", Rydberg.Global(None, 10)),),
        )


def test_convert_to_virtual():
    params = dict(
        name="Test",
        dimensions=2,
        rydberg_level=80,
        min_atom_distance=1,
        max_atom_num=20,
        max_radial_distance=40,
        _channels=(("rydberg_global", Rydberg.Global(0, 10)),),
    )
    assert Device(
        pre_calibrated_layouts=(TriangularLatticeLayout(40, 2),), **params
    ).to_virtual() == VirtualDevice(
        supports_slm_mask=False, reusable_channels=False, **params
    )
