import pytest
import numpy as np
import matplotlib.pyplot as plt

from pulser import Register


def test_creation():
    coords = [(0, 0), (1, 0)]
    ids = ['q0', 'q1']
    qubits = dict(zip(ids, coords))
    with pytest.raises(TypeError):
        Register(coords)
        Register(ids)

    reg1 = Register(qubits)
    reg2 = Register.from_coordinates(coords, center=False, prefix='q')
    assert reg1.qubits == reg2.qubits

    reg3 = Register.from_coordinates(coords, prefix='foo')
    coords_ = np.array([(-0.5, 0), (0.5, 0)])
    assert reg3._ids == ['foo0', 'foo1']
    assert np.all(reg3._coords == coords_)

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
    reg = Register.square(2, spacing=np.sqrt(2))
    reg.rotate(45)
    coords_ = np.array([(0, -1), (1, 0), (-1, 0), (0, 1)], dtype=float)
    assert np.all(np.isclose(reg._coords, coords_))


# def test_drawing():
#     reg = Register.triangular_lattice(3, 8)
#     reg.draw()
