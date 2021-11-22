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

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

import pulser
from pulser.devices import Chadoq2
from pulser.register import Register, Register3D


def test_init():
    for dev in pulser.devices._valid_devices:
        assert dev.dimensions in (2, 3)
        assert dev.rydberg_level > 49
        assert dev.rydberg_level < 101
        assert dev.max_atom_num > 10
        assert dev.max_radial_distance > 10
        assert dev.min_atom_distance > 0
        assert dev.interaction_coeff > 0
        assert dev.interaction_coeff_xy > 0
        assert isinstance(dev.channels, dict)
        with pytest.raises(FrozenInstanceError):
            dev.name = "something else"
    assert Chadoq2 in pulser.devices._valid_devices
    assert Chadoq2.supported_bases == {"digital", "ground-rydberg"}
    with patch("sys.stdout"):
        Chadoq2.print_specs()
    assert Chadoq2.__repr__() == "Chadoq2"


def test_mock():
    dev = pulser.devices.MockDevice
    assert dev.dimensions == 3
    assert dev.rydberg_level > 49
    assert dev.rydberg_level < 101
    assert dev.max_atom_num > 1000
    assert dev.min_atom_distance <= 1
    assert dev.interaction_coeff > 0
    assert dev.interaction_coeff_xy == 3700
    names = ["Rydberg", "Raman", "Microwave"]
    basis = ["ground-rydberg", "digital", "XY"]
    for ch in dev.channels.values():
        assert ch.name in names
        assert ch.basis == basis[names.index(ch.name)]
        assert ch.addressing in ["Local", "Global"]
        assert ch.max_abs_detuning >= 1000
        assert ch.max_amp >= 200
        if ch.addressing == "Local":
            assert ch.retarget_time == 0
            assert ch.max_targets > 1
            assert ch.max_targets == int(ch.max_targets)


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

    Chadoq2.validate_register(Register.rectangle(5, 10, spacing=5))
