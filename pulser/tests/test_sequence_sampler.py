from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import pulser
from pulser.channels import Rydberg
from pulser.devices import Device, MockDevice
from pulser.pulse import Pulse
from pulser.sampler import sample
from pulser.sampler.sampler import _write_dict
from pulser.sampler.samples import QubitSamples
from pulser.waveforms import BlackmanWaveform, RampWaveform


def test_corner_cases():
    """Test corner cases of helper functions."""
    with pytest.raises(
        ValueError,
        match="ndarrays amp, det and phase must have the same length.",
    ):
        _ = QubitSamples(
            amp=np.array([1.0]),
            det=np.array([1.0]),
            phase=np.array([1.0, 1.0]),
            qubit="q0",
        )

    reg = pulser.Register.square(1, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    N, M = 10, 11
    samples_dict = {
        "a": [QubitSamples(np.zeros(N), np.zeros(N), np.zeros(N), "q0")],
        "b": [QubitSamples(np.zeros(M), np.zeros(M), np.zeros(M), "q0")],
    }
    with pytest.raises(
        ValueError, match="All the samples do not share the same duration."
    ):
        _write_dict(seq, samples_dict, {})


def test_one_pulse_sampling():
    """Test the sample function on a one-pulse sequence."""
    reg = pulser.Register.square(1, prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    N = 1000
    amp_wf = BlackmanWaveform(N, np.pi)
    det_wf = RampWaveform(N, -np.pi / 2, np.pi / 2)
    phase = 1.234
    seq.add(Pulse(amp_wf, det_wf, phase), "ch0")
    seq.measure()

    got = sample(seq)["Global"]["ground-rydberg"]
    want = (amp_wf.samples, det_wf.samples, np.ones(N) * phase)
    for i, key in enumerate(["amp", "det", "phase"]):
        np.testing.assert_array_equal(got[key], want[i])


def check_same_samples_as_sim(seq: pulser.Sequence):
    """Check against the legacy sample extraction in the simulation module."""
    got = sample(seq)
    want = pulser.Simulation(seq).samples

    for basis in want["Global"]:
        for qty in want["Global"][basis]:
            np.testing.assert_array_equal(
                got["Global"][basis][qty],
                want["Global"][basis][qty],
            )
    for basis in want["Local"]:
        for qubit in want["Local"][basis]:
            for qty in want["Local"][basis][qubit]:
                np.testing.assert_array_equal(
                    got["Local"][basis][qubit][qty],
                    want["Local"][basis][qubit][qty],
                )


def test_table_sequence(seqs):
    """A table-driven test designed to be extended easily."""
    for seq in seqs:
        check_same_samples_as_sim(seq)


def test_inXY() -> None:
    """Test sequence in XY mode."""
    pulse = Pulse(
        BlackmanWaveform(200, np.pi / 2),
        RampWaveform(200, -np.pi / 2, np.pi / 2),
        0.0,
    )

    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "mw_global")
    seq.add(pulse, "ch0")
    seq.measure(basis="XY")

    sim = pulser.Simulation(seq)

    want = sim.samples
    got = sample(seq)

    for qty in ["amp", "det", "phase"]:
        np.testing.assert_array_equal(
            got["Global"]["XY"][qty], want["Global"]["XY"][qty]
        )

    for q in seq._qids:
        for qty in ["amp", "det", "phase"]:
            try:
                np.testing.assert_array_equal(
                    got["Local"]["XY"][q][qty], want["Local"]["XY"][q][qty]
                )
            except KeyError:
                np.testing.assert_array_equal(
                    got["Local"]["XY"][q][qty],
                    np.zeros(len(got["Local"]["XY"][q][qty])),
                )


def test_modulation(mod_seq: pulser.Sequence) -> None:
    """Test sampling for modulated channels."""
    N = mod_seq.get_duration()
    chan = mod_seq.declared_channels["ch0"]
    blackman = np.clip(np.blackman(N), 0, np.inf)
    input = (np.pi / 2) / (np.sum(blackman) / N) * blackman

    want = chan.modulate(input)
    got = sample(mod_seq, modulation=True)["Global"]["ground-rydberg"]["amp"]

    np.testing.assert_array_equal(got, want)


@pytest.fixture
def seqs(seq_rydberg) -> list[pulser.Sequence]:
    seqs: list[pulser.Sequence] = []

    pulse = Pulse(
        BlackmanWaveform(200, np.pi / 2),
        RampWaveform(200, -np.pi / 2, np.pi / 2),
        0.0,
    )

    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "raman_global")
    seq.add(pulse, "ch0")
    seq.measure()
    seqs.append(deepcopy(seq))

    seqs.append(seq_rydberg)

    return seqs


@pytest.fixture
def seq_rydberg() -> pulser.Sequence:
    reg = pulser.Register.from_coordinates(
        np.array([[0.0, 0.0], [2.0, 0.0]]), prefix="q"
    )
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "rydberg_local", initial_target="q0")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(100, np.pi / 8), 0.0, 0.0),
        "ch0",
    )
    seq.delay(20, "ch0")
    seq.add(
        Pulse.ConstantAmplitude(0.0, BlackmanWaveform(100, np.pi / 8), 0.0),
        "ch0",
    )
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(100, np.pi / 8), 0.0, 0.0),
        "ch1",
    )
    seq.target("q1", "ch1")
    seq.add(
        Pulse.ConstantAmplitude(1.0, BlackmanWaveform(100, np.pi / 8), 0.0),
        "ch1",
    )
    seq.target(["q0", "q1"], "ch1")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(100, np.pi / 8), 0.0, 0.0),
        "ch1",
    )
    seq.measure()
    return seq


@pytest.fixture
def mod_seq(mod_device: Device) -> pulser.Sequence:
    reg = pulser.Register.from_coordinates(np.array([[0.0, 0.0]]), prefix="q")
    seq = pulser.Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi / 2), 0.0, 0.0),
        "ch0",
    )
    seq.measure()
    return seq


@pytest.fixture
def mod_device() -> Device:
    return Device(
        name="ModDevice",
        dimensions=3,
        rydberg_level=70,
        max_atom_num=2000,
        max_radial_distance=1000,
        min_atom_distance=1,
        _channels=(
            (
                "rydberg_global",
                Rydberg(
                    "Global",
                    1000,
                    200,
                    clock_period=1,
                    min_duration=1,
                    mod_bandwidth=4.0,  # MHz
                ),
            ),
            (
                "rydberg_local",
                Rydberg(
                    "Local",
                    2 * np.pi * 20,
                    2 * np.pi * 10,
                    max_targets=2,
                    phase_jump_time=0,
                    fixed_retarget_t=0,
                    min_retarget_interval=220,
                    mod_bandwidth=4.0,
                ),
            ),
        ),
    )
