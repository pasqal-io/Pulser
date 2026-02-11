import numpy as np
import pytest

from pulser import MockDevice, Pulse, Sequence
from pulser._hamiltonian_data import HamiltonianData
from pulser.register import Register, Register3D
from pulser_simulation.hamiltonian import Hamiltonian


@pytest.fixture
def reg2d():
    q_dict = {
        "q0": np.array([-4.0, 0.0]),
        "q1": np.array([0.0, 4.0]),
    }

    return Register(q_dict)


@pytest.fixture
def reg3d():
    q_dict = {
        "q0": np.array([-4.0, 0.0, 0.0]),
        "q1": np.array([0.0, 4.0, 0.0]),
    }
    return Register3D(q_dict)


def test_register_2d(reg2d):
    # this should not error, see #940
    sampling_rate = 0.5
    duration = 10
    seq = Sequence(reg2d, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "raman_local", initial_target="q0")
    seq.declare_channel("ch2", "raman_local", initial_target="q1")

    pulse1 = Pulse.ConstantPulse(duration, 0, 0, 0)
    # Added twice to check the fluctuation doesn't change from pulse to pulse
    seq.add(pulse1, "ch0")
    seq.add(pulse1, "ch0")
    # The two local channels target alternating qubits on the same basis
    seq.add(pulse1, "ch1", protocol="no-delay")
    seq.add(pulse1, "ch2", protocol="no-delay")
    data = HamiltonianData.from_sequence(seq)
    for traj, noisy_samples, _ in data.noisy_samples:
        Hamiltonian(
            noisy_samples,
            traj,
            data.basis_data,
            data.lindblad_data,
            sampling_rate,
        )


def test_register_3d(reg3d):
    # this should not error, see #940
    sampling_rate = 0.5
    duration = 10
    seq = Sequence(reg3d, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "raman_local", initial_target="q0")
    seq.declare_channel("ch2", "raman_local", initial_target="q1")

    pulse1 = Pulse.ConstantPulse(duration, 0, 0, 0)
    # Added twice to check the fluctuation doesn't change from pulse to pulse
    seq.add(pulse1, "ch0")
    seq.add(pulse1, "ch0")
    # The two local channels target alternating qubits on the same basis
    seq.add(pulse1, "ch1", protocol="no-delay")
    seq.add(pulse1, "ch2", protocol="no-delay")
    data = HamiltonianData.from_sequence(seq)
    for traj, noisy_samples, _ in data.noisy_samples:
        Hamiltonian(
            noisy_samples,
            traj,
            data.basis_data,
            data.lindblad_data,
            sampling_rate,
        )
