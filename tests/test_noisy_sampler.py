import unittest

import numpy as np
import pytest

import pulser
from pulser.sampler import sample
from pulser.sampler.noisy_sampler import HamiltonianData

from .test_sequence_sampler import seq_rydberg, seq_with_SLM


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
        "pulser.sampler.noisy_sampler.sampler.sample"
    ) as mock_sample:
        with unittest.mock.patch(
            "pulser.sampler.noisy_sampler.HamiltonianData.__init__"
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
    noiseless = ham.samples_obj.to_nested_dict(all_local=True)
    full = ham.noisy_samples
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


def test_noisy_samples_obj():
    seq = seq_with_SLM("rydberg_global")
    ham = HamiltonianData.from_sequence(seq)
    with pytest.raises(NotImplementedError):
        ham.noisy_samples_obj


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
            ham.interaction_matrix,
            np.array([[0.0, interaction_size], [interaction_size, 0.0]]),
        )
    elif channel_type == "mw_global":
        interaction_size = ham._device.interaction_coeff_xy / 8**3
        assert np.allclose(
            ham.interaction_matrix,
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
            ham.interaction_matrix,
            torch.tensor(
                [[0.0, interaction_size], [interaction_size, 0.0]],
                dtype=torch.float64,
            ),
        )
        gr = torch.autograd.grad(
            ham.interaction_matrix[0, 1], q_dict["superman"]
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
            ham.interaction_matrix,
            torch.tensor(
                [[0.0, interaction_size], [interaction_size, 0.0]],
                dtype=torch.float64,
            ),
        )
        gr = torch.autograd.grad(
            ham.interaction_matrix[0, 1], q_dict["superman"]
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
        ham.noisy_interaction_matrix, np.zeros_like(ham.interaction_matrix)
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
        ham.noisy_interaction_matrix, torch.zeros_like(ham.interaction_matrix)
    )
