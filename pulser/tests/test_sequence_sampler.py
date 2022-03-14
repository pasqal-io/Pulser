import numpy as np
import pytest

import pulser
from pulser.devices import MockDevice
from pulser.pulse import Pulse
from pulser.sampler.noises import doppler
from pulser.sampler.sampler import sample
from pulser.simulation.simconfig import SimConfig
from pulser.waveforms import BlackmanWaveform

# from deepdiff import DeepDiff


def test_sequence_sampler(seq):
    """Check against the legacy sample extraction in the simulation module."""
    samples = sample(seq)
    sim = pulser.Simulation(seq)

    # Exclude the digital basis, since filled with zero vs empty.
    # it's just a way to check the coherence
    global_keys = [
        ("Global", basis, qty)
        for basis in ["ground-rydberg"]
        for qty in ["amp", "det", "phase"]
    ]
    local_keys = [
        ("Local", basis, qubit, qty)
        for basis in ["ground-rydberg"]
        for qubit in seq._qids
        for qty in ["amp", "det", "phase"]
    ]

    for k in global_keys:
        np.testing.assert_array_equal(
            samples[k[0]][k[1]][k[2]], sim.samples[k[0]][k[1]][k[2]]
        )

    for k in local_keys:
        np.testing.assert_array_equal(
            samples[k[0]][k[1]][k[2]][k[3]],
            sim.samples[k[0]][k[1]][k[2]][k[3]],
        )

    # print(DeepDiff(samples, sim.samples))
    # The deepdiff shows that there is no dict for unused basis in sim.samples,
    # where it's a zero dict for _sequence_sampler.sample


def test_doppler_noise(seq):

    MASS = 1.45e-25  # kg
    KB = 1.38e-23  # J/K
    KEFF = 8.7  # µm^-1
    doppler_sigma = KEFF * np.sqrt(KB * 50.0e-6 / MASS)

    local_noises = [doppler(seq.register, doppler_sigma, seed=0)]
    samples = sample(seq, local_noises=local_noises)
    got = samples["Local"]["ground-rydberg"]["q0"]["det"]

    np.random.seed(0)
    sim = pulser.Simulation(seq)
    sim.add_config(SimConfig("doppler"))
    sim._extract_samples()
    want = sim.samples["Local"]["ground-rydberg"]["q0"]["det"]

    np.testing.assert_array_equal(got, want)


@pytest.fixture
def seq() -> pulser.Sequence:
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
