import pytest

from pulser import Register
from pulser.devices import Chadoq2


def test_init():
    with pytest.raises(TypeError):
        Chadoq2([1, 2, 4])
        Chadoq2('q1')
        Chadoq2(('a', 'b', 'c'))

    qubits = dict(enumerate([(0, 0), (10, 0)]))
    dev1 = Chadoq2(qubits)
    reg = Register(qubits)
    dev2 = Chadoq2(reg)
    assert dev1.qubits == dev2.qubits
    assert isinstance(dev1.channels, dict)
    assert dev2.supported_bases == {'digital', 'ground-rydberg'}


def test_check_array():
    with pytest.raises(ValueError, match='Too many atoms'):
        Chadoq2(Register.square(50))

    coords = [(100, 0), (-100, 0)]
    with pytest.raises(ValueError, match='at most 50um away from the center'):
        Chadoq2(Register.from_coordinates(coords))

    with pytest.raises(ValueError, match='must be 2D vectors'):
        coords += [(-10, 4, 0)]
        Chadoq2(dict(enumerate(coords)))

    with pytest.raises(ValueError, match="don't respect the minimal distance"):
        Chadoq2(Register.triangular_lattice(3, 4, spacing=3.9))

    Chadoq2(Register.rectangle(5, 10, spacing=5))
