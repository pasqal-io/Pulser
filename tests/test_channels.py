import pytest

from pulser.channels import Raman, Rydberg


def test_init():
    with pytest.raises(ValueError, match="set retarget time for local"):
        Raman('local', 50, 10)

    with pytest.raises(ValueError, match="Can't set retarget time for global"):
        Rydberg('global', 50, 10, retarget_time=10)

    with pytest.raises(ValueError, match="'global' or 'local'"):
        Raman('total', 20, 1)
        Rydberg(2, 1, 0)

    with pytest.raises(ValueError, match="detuning has to be positive"):
        Raman('global', -1, 1)

    with pytest.raises(ValueError, match="amplitude has to be positive"):
        Rydberg('global', 1, 0)

    Raman('local', 10, 2, retarget_time=1000)
    Rydberg('global', 50, 2.5)


def test_repr():
    raman = Raman('local', 10, 2, retarget_time=1000)
    r1 = ("Raman(local, Max Absolute Detuning: 10 MHz, Max Amplitude: 2 MHz,"
          " Target time: 1000 ns, Basis: 'digital')")
    assert raman.__str__() == r1

    ryd = Rydberg('global', 50, 2.5)
    r2 = ("Rydberg(global, Max Absolute Detuning: 50 MHz, "
          "Max Amplitude: 2.5 MHz, Basis: 'ground-rydberg')")
    assert ryd.__str__() == r2
