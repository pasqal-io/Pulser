import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pulser
import pulser.math as pm
from pulser.hamiltonian_data.hamiltonian_data import (
    HamiltonianData,
    _generate_detuning_fluctuations,
    _noisy_register,
    _register_sigma_xy_z,
)
from pulser.sampler import sample

from .test_sequence_sampler import seq_rydberg, seq_with_SLM


def test_sigma_register_xy_z():
    temperature = 15.0
    trap_waist = 1.0
    trap_depth = 150.0
    sigma_xy, sigma_z = _register_sigma_xy_z(
        temperature, trap_waist, trap_depth
    )
    assert 0.158 == pytest.approx(sigma_xy, abs=1e-2)
    assert 0.826 == pytest.approx(sigma_z, abs=1e-2)


@pytest.mark.parametrize(
    "register2D",
    [
        True,
        False,
    ],
)
def test_noisy_register(register2D) -> None:
    """Testing noisy_register function in case 2D and 3D register."""
    if register2D:
        qdict = {
            "q0": pm.AbstractArray(np.array([-15.0, 0.0])),
            "q1": pm.AbstractArray(np.array([-5.0, 0.0])),
            "q2": pm.AbstractArray(np.array([5.0, 0.0])),
            "q3": pm.AbstractArray(np.array([15.0, 0.0])),
        }
    else:
        qdict = {
            "q0": pm.AbstractArray(np.array([-15.0, 0.0, 0.0])),
            "q1": pm.AbstractArray(np.array([-5.0, 0.0, 0.0])),
            "q2": pm.AbstractArray(np.array([5.0, 0.0, 0.0])),
            "q3": pm.AbstractArray(np.array([15.0, 0.0, 0.0])),
        }

    # Predefined deterministic noise
    fake_normal_xy_noise = np.array(
        [
            [0.1, -0.1],
            [0.2, -0.2],
            [0.3, -0.3],
            [0.5, -0.5],
        ]
    )
    fake_normal_z_noise = np.array([0.05, 0.07, 0.09, 0.11])

    mock_noise_model = MagicMock(
        spec=pulser.NoiseModel
    )  # generic NoiseModel class
    with patch(
        "pulser.hamiltonian_data.hamiltonian_data._register_sigma_xy_z",
        return_value=(0.13, 0.8),
    ):
        with patch("numpy.random.normal") as mock_normal:
            # moke the noise generation
            mock_normal.side_effect = [
                fake_normal_xy_noise,
                fake_normal_z_noise,
            ]
            result = _noisy_register(qdict, mock_noise_model)
    expected_positions = {
        "q0": np.array([-15.0 + 0.1, 0.0 - 0.1, 0.05]),
        "q1": np.array([-5.0 + 0.2, 0.0 - 0.2, 0.07]),
        "q2": np.array([5.0 + 0.3, 0.0 - 0.3, 0.09]),
        "q3": np.array([15.0 + 0.5, 0.0 - 0.5, 0.11]),
    }
    for q in qdict:
        np.testing.assert_array_almost_equal(
            result.qubits[q], expected_positions[q]
        )


@pytest.mark.parametrize(
    "register2D",
    [
        True,
        False,
    ],
)
def test_noisy_register_torch(register2D):
    torch = pytest.importorskip("torch")
    if register2D:
        qdict = {
            "q0": pm.AbstractArray(
                torch.tensor([-15.0, 0.0], requires_grad=True)
            ),
            "q1": pm.AbstractArray(
                torch.tensor([-5.0, 0.0], requires_grad=True)
            ),
            "q2": pm.AbstractArray(
                torch.tensor([5.0, 0.0], requires_grad=True)
            ),
            "q3": pm.AbstractArray(
                torch.tensor([15.0, 0.0], requires_grad=True)
            ),
        }
    else:
        qdict = {
            "q0": pm.AbstractArray(
                torch.tensor([-15.0, 0.0, 0.0], requires_grad=True)
            ),
            "q1": pm.AbstractArray(
                torch.tensor([-5.0, 0.0, 0.0], requires_grad=True)
            ),
            "q2": pm.AbstractArray(
                torch.tensor([5.0, 0.0, 0.0], requires_grad=True)
            ),
            "q3": pm.AbstractArray(
                torch.tensor([15.0, 0.0, 0.0], requires_grad=True)
            ),
        }

    # Predefined deterministic noise
    fake_normal_xy_noise = torch.tensor(
        [
            [0.1, -0.1],
            [0.2, -0.2],
            [0.3, -0.3],
            [0.5, -0.5],
        ]
    )
    fake_normal_z_noise = torch.tensor([[0.05], [0.07], [0.09], [0.11]])

    mock_noise_model = MagicMock(
        spec=pulser.NoiseModel
    )  # generic NoiseModel class
    with patch(
        "pulser.hamiltonian_data.hamiltonian_data._register_sigma_xy_z",
        return_value=(0.13, 0.8),
    ):
        with patch("torch.normal") as mock_normal:
            # moke the noise generation
            mock_normal.side_effect = [
                fake_normal_xy_noise,
                fake_normal_z_noise,
            ]
            result = _noisy_register(qdict, mock_noise_model)
    expected_positions = {
        "q0": torch.tensor(
            [-15.0 + 0.1, 0.0 - 0.1, 0.05], dtype=pm.torch.float64
        ),
        "q1": torch.tensor(
            [-5.0 + 0.2, 0.0 - 0.2, 0.07], dtype=pm.torch.float64
        ),
        "q2": torch.tensor(
            [5.0 + 0.3, 0.0 - 0.3, 0.09], dtype=pm.torch.float64
        ),
        "q3": torch.tensor(
            [15.0 + 0.5, 0.0 - 0.5, 0.11], dtype=pm.torch.float64
        ),
    }
    for q in qdict:
        assert torch.allclose(result.qubits[q]._array, expected_positions[q])
    g = torch.autograd.grad(result.qubits["q0"]._array[0], qdict["q0"]._array)[
        0
    ]
    assert g.shape == qdict["q0"]._array.shape


def test_init_errors():
    seq = seq_with_SLM("rydberg_global")
    seq_samples = sample(seq)
    register = pulser.Register.square(3, spacing=6)
    with pytest.raises(
        TypeError,
        match=(
            "The provided sequence has to be a "
            "valid SequenceSamples instance."
        ),
    ):
        HamiltonianData(None, None, None, None)

    with pytest.raises(
        TypeError, match="The device must be a Device or BaseDevice."
    ):
        HamiltonianData(seq_samples, None, None, None)

    with pytest.raises(
        ValueError, match="Samples use SLM mask but device does not have one."
    ):
        HamiltonianData(seq_samples, seq.register, pulser.AnalogDevice, None)

    with pytest.raises(
        ValueError,
        match=(
            "The ids of qubits targeted in SLM "
            "mask should be defined in register."
        ),
    ):
        HamiltonianData(
            seq_samples, register, pulser.DigitalAnalogDevice, None
        )

    with pytest.raises(
        ValueError,
        match=(
            "The ids of qubits targeted in Local "
            "channels should be defined in register."
        ),
    ):
        HamiltonianData(
            sample(seq_rydberg()), register, pulser.DigitalAnalogDevice, None
        )

    seq = pulser.Sequence(register, pulser.AnalogDevice)
    seq.declare_channel("ch0", "rydberg_global")
    with pytest.raises(ValueError, match="SequenceSamples is empty."):
        HamiltonianData(sample(seq), None, None, None)

    seq = seq_with_SLM("mw_global")
    seq_samples = sample(seq)
    with pytest.raises(
        ValueError,
        match="Bases used in samples should be supported by device.",
    ):
        HamiltonianData(
            seq_samples, seq.register, pulser.DigitalAnalogDevice, None
        )


def test_from_sequence():
    seq = seq_with_SLM("rydberg_global")
    noise_model = pulser.NoiseModel(detuning_sigma=0.5, runs=1)
    samples = 5  # something that implements __eq__
    with unittest.mock.patch(
        "pulser.hamiltonian_data.hamiltonian_data.sampler.sample"
    ) as mock_sample:
        with unittest.mock.patch(
            "pulser.hamiltonian_data.hamiltonian_data.HamiltonianData.__init__"
        ) as mock_init:
            mock_init.return_value = None
            mock_sample.return_value = samples
            HamiltonianData.from_sequence(seq, noise_model=noise_model)
            mock_init.assert_called_once_with(
                samples, seq.register, seq.device, noise_model
            )

    with pytest.raises(
        TypeError,
        match=(
            "The provided sequence has to be "
            "a valid pulser.Sequence instance."
        ),
    ):
        HamiltonianData.from_sequence(None)

    seq._building = False
    with pytest.raises(
        ValueError,
        match=(
            "The provided sequence needs to be built to be simulated. "
            r"Call `Sequence.build\(\)` with the necessary parameters."
        ),
    ):
        HamiltonianData.from_sequence(seq)

    seq._building = True
    sched = seq._schedule
    seq._schedule = None
    with pytest.raises(
        ValueError, match="The provided sequence has no declared channels."
    ):
        HamiltonianData.from_sequence(seq)
    seq._schedule = sched

    with pytest.raises(
        NotImplementedError,
        match=(
            "Simulation of sequences combining an SLM mask "
            "and output modulation is not supported."
        ),
    ):
        HamiltonianData.from_sequence(seq, with_modulation=True)

    with pytest.raises(
        ValueError,
        match="No instructions given for the channels in the sequence.",
    ):
        seq2 = pulser.Sequence(seq.register, seq.device)
        seq2.declare_channel("ch0", "rydberg_global")
        HamiltonianData.from_sequence(seq2)

    ham = HamiltonianData.from_sequence(seq, noise_model=noise_model)
    noiseless = ham.samples.to_nested_dict(all_local=True)
    full = ham.noisy_samples.to_nested_dict()
    diff = (
        noiseless["Local"]["ground-rydberg"]["superman"]["det"]
        - full["Local"]["ground-rydberg"]["superman"]["det"]
    )
    assert not np.any(np.isclose(diff, np.zeros_like(diff)))
    assert np.all(
        np.isclose(
            noiseless["Local"]["ground-rydberg"]["batman"]["det"]
            - full["Local"]["ground-rydberg"]["batman"]["det"],
            diff,
        )
    )


def test_register():
    seq = seq_with_SLM("rydberg_global")
    ham = HamiltonianData.from_sequence(seq)
    assert ham.register == seq.register


def test_bad_atoms():
    seq = seq_with_SLM("rydberg_global")
    noise = pulser.NoiseModel(state_prep_error=1.0, runs=1)
    ham = HamiltonianData.from_sequence(seq, noise_model=noise)
    for key in seq.register.qubit_ids:
        assert ham.bad_atoms[key]


@pytest.mark.parametrize("channel_type", ["rydberg_global", "mw_global"])
def test_interaction_matrix(channel_type):
    q_dict = {
        "batman": [-4.0, 0.0],
        "superman": [4.0, 0.0],
    }
    reg = pulser.Register(q_dict)
    seq = pulser.Sequence(reg, pulser.MockDevice)

    seq.declare_channel("ch0", channel_type)

    seq.add(
        pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(200, np.pi / 5), 0.0, 0.0
        ),
        "ch0",
    )
    ham = HamiltonianData.from_sequence(seq)

    if channel_type == "rydberg_global":
        interaction_size = ham._device.interaction_coeff / 8**6
        assert np.allclose(
            ham._interaction_matrix,
            np.array([[0.0, interaction_size], [interaction_size, 0.0]]),
        )
    elif channel_type == "mw_global":
        interaction_size = ham._device.interaction_coeff_xy / 8**3
        assert np.allclose(
            ham._interaction_matrix,
            np.array([[0.0, interaction_size], [interaction_size, 0.0]]),
        )
    else:
        assert False


@pytest.mark.parametrize("channel_type", ["rydberg_global", "mw_global"])
def test_interaction_matrix_torch(channel_type):
    torch = pytest.importorskip("torch")
    q_dict = {
        "batman": torch.tensor(
            [-4.0, 0.0], dtype=torch.float64, requires_grad=True
        ),
        "superman": torch.tensor(
            [4.0, 0.0], dtype=torch.float64, requires_grad=True
        ),
    }
    reg = pulser.Register(q_dict)
    seq = pulser.Sequence(reg, pulser.MockDevice)

    seq.declare_channel("ch0", channel_type)

    seq.add(
        pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(200, np.pi / 5), 0.0, 0.0
        ),
        "ch0",
    )
    ham = HamiltonianData.from_sequence(seq)
    if channel_type == "rydberg_global":
        interaction_size = ham._device.interaction_coeff / 8**6
        assert torch.allclose(
            ham._interaction_matrix,
            torch.tensor(
                [[0.0, interaction_size], [interaction_size, 0.0]],
                dtype=torch.float64,
            ),
        )
        gr = torch.autograd.grad(
            ham._interaction_matrix[0, 1], q_dict["superman"]
        )
        assert torch.allclose(
            gr[0],
            torch.tensor(
                [-6 * ham._device.interaction_coeff / 8**7, 0.0],
                dtype=torch.float64,
            ),
        )
    elif channel_type == "mw_global":
        interaction_size = ham._device.interaction_coeff_xy / 8**3
        assert torch.allclose(
            ham._interaction_matrix,
            torch.tensor(
                [[0.0, interaction_size], [interaction_size, 0.0]],
                dtype=torch.float64,
            ),
        )
        gr = torch.autograd.grad(
            ham._interaction_matrix[0, 1], q_dict["superman"]
        )
        assert torch.allclose(
            gr[0],
            torch.tensor(
                [-3 * ham._device.interaction_coeff_xy / 8**4, 0.0],
                dtype=torch.float64,
            ),
        )
    else:
        assert False


def test_noisy_interaction_matrix():
    q_dict = {
        "batman": [-4.0, 0.0],
        "superman": [4.0, 0.0],
    }
    reg = pulser.Register(q_dict)
    seq = pulser.Sequence(reg, pulser.AnalogDevice)

    seq.declare_channel("ch0", "rydberg_global")

    seq.add(
        pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(200, np.pi / 5), 0.0, 0.0
        ),
        "ch0",
    )
    noise = pulser.NoiseModel(state_prep_error=1.0, runs=1)
    ham = HamiltonianData.from_sequence(seq, noise_model=noise)
    assert np.allclose(
        ham.noisy_interaction_matrix._array,
        np.zeros_like(ham._interaction_matrix),
    )


def test_noisy_interaction_matrix_torch():
    torch = pytest.importorskip("torch")
    q_dict = {
        "batman": torch.tensor([-4.0, 0.0], dtype=torch.float64),
        "superman": torch.tensor([4.0, 0.0], dtype=torch.float64),
    }
    reg = pulser.Register(q_dict)
    seq = pulser.Sequence(reg, pulser.AnalogDevice)

    seq.declare_channel("ch0", "rydberg_global")

    seq.add(
        pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(200, np.pi / 5), 0.0, 0.0
        ),
        "ch0",
    )
    noise = pulser.NoiseModel(state_prep_error=1.0, runs=1)
    ham = HamiltonianData.from_sequence(seq, noise_model=noise)
    assert torch.allclose(
        ham.noisy_interaction_matrix._array,
        torch.zeros_like(ham._interaction_matrix),
    )


def test_noise_hf_detuning_generation():
    def original_formula_gen_noise(psd, freqs, times, phases):
        """Compute δ_hf(t).

        Args:
            psd : in Hz²/Hz
            freqs : in Hz
            times : in µs, is converted to seconds inside
            phases : phase offsets
        """
        hf_detun = np.zeros_like(times)
        times *= 1e-6  # µsec -> sec
        for i, s in enumerate(psd[1:]):
            df = freqs[i + 1] - freqs[i]
            hf_detun += (
                2.0
                * np.pi
                * 1e-6
                * np.sqrt(2 * df * s)
                * np.cos(2 * np.pi * (freqs[i + 1] * times + phases[i]))
            )
        return hf_detun

    psd = [1, 2, 3]
    freqs = [3, 4, 5]
    times = np.arange(0, 10, 0.1)
    phases = np.random.uniform(size=(2,))

    noise_m = pulser.NoiseModel(
        detuning_hf_psd=psd, detuning_hf_freqs=freqs, runs=1
    )
    hf_det = _generate_detuning_fluctuations(noise_m, 0.0, phases, times)
    hd_det_expected = original_formula_gen_noise(psd, freqs, times, phases)

    assert np.allclose(hf_det, hd_det_expected)
    assert hf_det.size == times.size
