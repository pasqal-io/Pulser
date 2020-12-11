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

import pytest

import pulser
from pulser.devices import Chadoq2
from pulser.register import Register


def test_init():
    for dev in pulser.devices._valid_devices:
        assert dev.dimensions in (2, 3)
        assert dev.max_atom_num > 10
        assert dev.max_radial_distance > 10
        assert dev.min_atom_distance > 0
        assert isinstance(dev.channels, dict)
        with pytest.raises(FrozenInstanceError):
            dev.name = "something else"
        for i, (id, ch) in enumerate(dev.channels.items()):
            assert id == dev._channels[i][0]
            assert isinstance(id, str)
            assert ch == dev._channels[i][1]
            assert isinstance(ch, pulser.channels.Channel)
    assert Chadoq2 in pulser.devices._valid_devices
    assert Chadoq2.supported_bases == {'digital', 'ground-rydberg'}
    with patch('sys.stdout'):
        Chadoq2.specs()
    assert Chadoq2.__repr__() == 'Chadoq2'


def test_validate_register():
    with pytest.raises(ValueError, match='Too many atoms'):
        Chadoq2._validate_register(Register.square(50))

    coords = [(100, 0), (-100, 0)]
    with pytest.raises(TypeError):
        Chadoq2._validate_register(coords)
    with pytest.raises(ValueError, match='at most 50 um away from the center'):
        Chadoq2._validate_register(Register.from_coordinates(coords))

    with pytest.raises(ValueError, match='must be 2D vectors'):
        coords += [(-10, 4, 0)]
        Chadoq2._validate_register(Register(dict(enumerate(coords))))

    with pytest.raises(ValueError, match="don't respect the minimal distance"):
        Chadoq2._validate_register(Register.triangular_lattice(
                                                            3, 4, spacing=3.9))

    Chadoq2._validate_register(Register.rectangle(5, 10, spacing=5))
