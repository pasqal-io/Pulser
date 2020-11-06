import pytest
import numpy as np

from pulser import Sequence, Pulse, Register
from pulser.devices import Chadoq2

reg = Register.triangular_lattice(4, 7, spacing=5, prefix='q')
device = Chadoq2(reg)


def test_init():
    with pytest.raises(TypeError, match='has to be a PasqalDevice'):
        Sequence(Chadoq2)
        Sequence(np.random.rand(10, 2))

    seq = Sequence(device)
    assert seq.qubit_info == reg.qubits
    assert seq.declared_channels == {}
    assert seq.available_channels.keys() == device.channels.keys()


def test_channel_declaration():
    seq = Sequence(device)
    seq.declare_channel('ch0', 'rydberg_global')
    seq.declare_channel('ch1', 'raman_local')
    with pytest.raises(ValueError, match="No channel"):
        seq.declare_channel('ch2', 'raman')
    with pytest.raises(ValueError, match="has already been added"):
        seq.declare_channel('ch2', 'rydberg_global')
    with pytest.raises(ValueError, match="name is already in use"):
        seq.declare_channel('ch0', 'raman_local')

    chs = {'rydberg_global', 'raman_local'}
    assert set(seq.available_channels) == {ch for ch in device.channels
                                           if ch not in chs}


def test_target():
    seq = Sequence(device)
    seq.declare_channel('ch0', 'raman_local')
    with pytest.raises(ValueError, match='name of a declared channel'):
        seq.target('q0', 'ch1')
    with pytest.raises(ValueError, match='qubits have to belong'):
        seq.target(0, 'ch0')
        seq.target('0', 'ch0')
