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

from pulser import Register


def test_creation():
    coords = [(0, 0), (1, 0)]
    ids = ['q0', 'q1']
    qubits = dict(zip(ids, coords))
    with pytest.raises(TypeError):
        Register(coords)
        Register(ids)

    with pytest.raises(ValueError, match="vectors of size 2 or 3"):
        Register.from_coordinates([(0, 1, 0, 1)])

    with pytest.raises(ValueError, match="vectors of size 2 or 3"):
        Register.from_coordinates([((1, 0),), ((-1, 0),)])

    reg1 = Register(qubits)
    reg2 = Register.from_coordinates(coords, center=False, prefix='q')
    assert np.all(np.array(reg1._coords) == np.array(reg2._coords))
    assert reg1._ids == reg2._ids

    reg3 = Register.from_coordinates(np.array(coords), prefix='foo')
    coords_ = np.array([(-0.5, 0), (0.5, 0)])
    assert reg3._ids == ['foo0', 'foo1']
    assert np.all(reg3._coords == coords_)
    assert not np.all(coords_ == coords)

    reg4 = Register.rectangle(1, 2, spacing=1)
    assert np.all(reg4._coords == coords_)

    reg5 = Register.square(2, spacing=2)
    coords_ = np.array([(-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=float)
    assert np.all(np.array(reg5._coords) == coords_)

    reg6 = Register.triangular_lattice(2, 2, spacing=4)
    coords_ = np.array([(-3, -np.sqrt(3)), (1, -np.sqrt(3)),
                        (-1, np.sqrt(3)), (3, np.sqrt(3))])
    assert np.all(np.array(reg6._coords) == coords_)


def test_rotation():
    with pytest.raises(NotImplementedError):
        reg_ = Register.from_coordinates([(1, 0, 0), (0, 1, 4)])
        reg_.rotate(20)
    reg = Register.square(2, spacing=np.sqrt(2))
    reg.rotate(45)
    coords_ = np.array([(0, -1), (1, 0), (-1, 0), (0, 1)], dtype=float)
    assert np.all(np.isclose(reg._coords, coords_))


def test_drawing():
    with pytest.raises(NotImplementedError, match="register layouts in 2D."):
        reg_ = Register.from_coordinates([(1, 0, 0), (0, 1, 4)])
        reg_.draw()
    reg = Register.triangular_lattice(3, 8)
    with patch('matplotlib.pyplot.show'):
        reg.draw()

    reg = Register.rectangle(1, 8)
    with patch('matplotlib.pyplot.show'):
        reg.draw(blockade_radius=5, draw_half_radius=True, draw_graph=True)

    with pytest.raises(ValueError, match="to draw the graph."):
        reg.draw(draw_graph=True)

    with pytest.raises(ValueError, match="'blockade_radius' to draw."):
        reg.draw(draw_half_radius=True)

    reg = Register.square(1)
    with pytest.raises(NotImplementedError, match="Needs more than one atom"):
        reg.draw(blockade_radius=5, draw_half_radius=True)
