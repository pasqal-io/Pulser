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
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

import numpy as np
import pytest

import pulser
from pulser.channels import Microwave, Raman, Rydberg
from pulser.channels.dmm import DMM
from pulser.devices import Chadoq2, Device, VirtualDevice
from pulser.register import Register, Register3D
from pulser.register.register_layout import RegisterLayout
from pulser.register.special_layouts import (
    SquareLatticeLayout,
    TriangularLatticeLayout,
)


@pytest.fixture
def test_params():
    return dict(
        name="Test",
        dimensions=2,
        rydberg_level=70,
        channel_ids=None,
        channel_objects=(),
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
            "channel_ids",
            {"fake_channel"},
            "When defined, 'channel_ids' must be a tuple or a list "
            "of strings.",
        ),
        (
            "channel_ids",
            ("ch1", 2),
            "When defined, 'channel_ids' must be a tuple or a list "
            "of strings.",
        ),
        (
            "channel_objects",
            ("Rydberg.Global(None, None)",),
            "All channels must be of type 'Channel', not 'str'",
        ),
        (
            "channel_objects",
            (Microwave.Global(None, None),),
            "When the device has a 'Microwave' channel, "
            "'interaction_coeff_xy' must be a 'float',"
            " not '<class 'NoneType'>'.",
        ),
        (
            "dmm_objects",
            ("DMM(bottom_detuning=-1)",),
            "All DMM channels must be of type 'DMM', not 'str'",
        ),
        ("max_sequence_duration", 1.02, None),
        ("max_runs", 1e8, None),
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
        (
            "max_layout_filling",
            0.0,
            "maximum layout filling fraction must be greater than 0. and"
            " less than or equal to 1.",
        ),
        (
            "channel_ids",
            ("rydberg_global", "rydberg_global"),
            "When defined, 'channel_ids' can't have repeated elements.",
        ),
        (
            "channel_ids",
            ("rydberg_global",),
            "When defined, the number of channel IDs must"
            " match the number of channel objects.",
        ),
        ("max_sequence_duration", 0, None),
        ("max_runs", 0, None),
    ],
)
def test_post_init_value_errors(test_params, param, value, msg):
    test_params[param] = value
    error_msg = msg or f"When defined, '{param}' must be greater than zero"
    with pytest.raises(ValueError, match=error_msg):
        VirtualDevice(**test_params)


def test_post_init_slm_dmm_compatibility(test_params):
    test_params["supports_slm_mask"] = True
    test_params["dmm_objects"] = ()
    with pytest.raises(
        ValueError,
        match="One DMM object should be defined to support SLM mask.",
    ):
        VirtualDevice(**test_params)


potential_params = ["max_atom_num", "max_radial_distance"]
always_none_allowed = ["max_sequence_duration", "max_runs"]


@pytest.mark.parametrize("none_param", potential_params + always_none_allowed)
def test_optional_parameters(test_params, none_param):
    test_params.update({p: 10 for p in potential_params})
    test_params[none_param] = None
    if none_param not in always_none_allowed:
        with pytest.raises(
            TypeError,
            match=f"'{none_param}' can't be None in a 'Device' instance.",
        ):
            Device(**test_params)
    else:
        Device(**test_params)
    VirtualDevice(**test_params)  # Valid as None on a VirtualDevice


def test_default_channel_ids(test_params):
    # Needed because of the Microwave global channel
    test_params["interaction_coeff_xy"] = 10000.0
    test_params["channel_objects"] = (
        Rydberg.Local(None, None),
        Raman.Local(None, None),
        Rydberg.Local(None, None),
        Raman.Global(None, None),
        Microwave.Global(None, None),
    )
    dev = VirtualDevice(**test_params)
    assert dev.channel_ids == (
        "rydberg_local",
        "raman_local",
        "rydberg_local_2",
        "raman_global",
        "mw_global",
    )


def test_tuple_conversion(test_params):
    test_params["channel_objects"] = [Rydberg.Global(None, None)]
    test_params["channel_ids"] = ["custom_channel"]
    dev = VirtualDevice(**test_params)
    assert dev.channel_objects == (Rydberg.Global(None, None),)
    assert dev.channel_ids == ("custom_channel",)


def test_valid_devices():
    for dev in pulser.devices._valid_devices:
        assert dev.dimensions in (2, 3)
        assert dev.rydberg_level > 49
        assert dev.rydberg_level < 101
        assert dev.max_atom_num > 10
        assert dev.max_radial_distance > 10
        assert dev.min_atom_distance > 0
        assert dev.interaction_coeff > 0
        assert 0 < dev.max_layout_filling <= 1
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
        tri_layout = TriangularLatticeLayout(200, 20)
        Chadoq2.validate_register(tri_layout.hexagonal_register(10))

    Chadoq2.validate_register(Register.rectangle(5, 10, spacing=5))


def test_validate_layout():
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


@pytest.mark.parametrize(
    "register",
    [
        TriangularLatticeLayout(100, 5).hexagonal_register(80),
        TriangularLatticeLayout(100, 5).make_mappable_register(51),
    ],
)
def test_layout_filling(register):
    assert Chadoq2.max_layout_filling == 0.5
    assert register.layout.number_of_traps == 100
    with pytest.raises(
        ValueError,
        match=re.escape(
            "the given register has too many qubits "
            f"({len(register.qubit_ids)}). "
            "On this device, this layout can hold at most 50 qubits."
        ),
    ):
        Chadoq2.validate_layout_filling(register)


def test_layout_filling_fail():
    with pytest.raises(
        TypeError,
        match="'validate_layout_filling' can only be called for"
        " registers with a register layout.",
    ):
        Chadoq2.validate_layout_filling(Register.square(5))


def test_calibrated_layouts():
    with pytest.raises(ValueError, match="The minimal distance between traps"):
        Device(
            name="TestDevice",
            dimensions=2,
            rydberg_level=70,
            max_atom_num=100,
            max_radial_distance=50,
            min_atom_distance=4,
            channel_objects=(),
            pre_calibrated_layouts=(TriangularLatticeLayout(201, 3),),
        )

    layout100 = TriangularLatticeLayout(100, 6.8)
    layout200 = TriangularLatticeLayout(200, 5)
    TestDevice = Device(
        name="TestDevice",
        dimensions=2,
        rydberg_level=70,
        max_atom_num=100,
        max_radial_distance=50,
        min_atom_distance=4,
        channel_objects=(),
        pre_calibrated_layouts=(layout100, layout200),
    )
    assert TestDevice.calibrated_register_layouts.keys() == {
        "TriangularLatticeLayout(100, 6.8µm)",
        "TriangularLatticeLayout(200, 5.0µm)",
    }
    with pytest.raises(
        TypeError,
        match="The register to check must be of type ",
    ):
        TestDevice.register_is_from_calibrated_layout(layout100)
    assert TestDevice.is_calibrated_layout(layout100)
    register = layout200.define_register(*range(10))
    assert TestDevice.register_is_from_calibrated_layout(register)
    # Checking a register not built from a layout returns False
    assert not TestDevice.register_is_from_calibrated_layout(
        Register.triangular_lattice(4, 25, 6.8)
    )
    # Checking Layouts that don't match calibrated layouts returns False
    square_layout = SquareLatticeLayout(10, 10, 6.8)
    layout125 = TriangularLatticeLayout(125, 6.8)
    compact_layout = TriangularLatticeLayout(100, 3)
    for bad_layout in (square_layout, layout125, compact_layout):
        assert not TestDevice.is_calibrated_layout(bad_layout)
        register = bad_layout.define_register(*range(10))
        assert not TestDevice.register_is_from_calibrated_layout(register)


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
            channel_objects=(Rydberg.Global(None, 10),),
        )


def test_convert_to_virtual():
    params = dict(
        name="Test",
        dimensions=2,
        rydberg_level=80,
        min_atom_distance=1,
        max_atom_num=20,
        max_radial_distance=40,
        channel_objects=(Rydberg.Global(0, 10),),
    )
    assert Device(
        pre_calibrated_layouts=(TriangularLatticeLayout(40, 2),), **params
    ).to_virtual() == VirtualDevice(
        supports_slm_mask=False,
        reusable_channels=False,
        dmm_objects=(),
        **params,
    )


def test_device_params():
    all_params = Chadoq2._params()
    init_params = Chadoq2._params(init_only=True)
    assert set(all_params) - set(init_params) == {"reusable_channels"}

    virtual_chadoq2 = Chadoq2.to_virtual()
    all_virtual_params = virtual_chadoq2._params()
    init_virtual_params = virtual_chadoq2._params(init_only=True)
    assert all_virtual_params == init_virtual_params
    assert set(all_params) - set(all_virtual_params) == {
        "pre_calibrated_layouts"
    }


def test_dmm_channels():
    with pytest.raises(
        ValueError,
        match="A 'Device' instance cannot contain virtual channels."
        " For channel 'dmm_0', please define: 'bottom_detuning'",
    ):
        replace(Chadoq2, dmm_objects=(DMM(),))
    dmm = DMM(
        bottom_detuning=-1,
        total_bottom_detuning=-100,
        clock_period=1,
        min_duration=1,
        max_duration=1e6,
        mod_bandwidth=20,
    )
    device = replace(Chadoq2, dmm_objects=(dmm,))
    assert len(device.dmm_channels) == 1
    assert device.dmm_channels["dmm_0"] == dmm
    with pytest.raises(
        ValueError,
        match=(
            "When defined, the names of channel IDs must be different"
            " than the names of DMM channels 'dmm_0', 'dmm_1', ... ."
        ),
    ):
        device = replace(
            Chadoq2,
            dmm_objects=(dmm,),
            channel_objects=(Rydberg.Global(None, None),),
            channel_ids=("dmm_0",),
        )
