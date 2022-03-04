import numpy as np

from pulser import Pulse, Register, Sequence, Simulation
from pulser.channels import Rydberg
from pulser.devices import Device
from pulser.waveforms import BlackmanWaveform


def modulated_mock_device() -> Device:
    """A Mock device with only one modulated channel."""
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
                    mod_bandwidth=4.00,  # MHz
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
                    mod_bandwidth=4,
                ),
            ),
        ),
    )


def one_atom_register() -> Register:
    return Register.from_coordinates(np.array([(0, 0)]), prefix="atom")


def two_atom_register() -> Register:
    return Register.from_coordinates(np.array([(0, 0), (2, 0)]), prefix="atom")


def simple_sequence() -> Sequence:
    seq = Sequence(one_atom_register(), modulated_mock_device())
    seq.declare_channel("ch0", "rydberg_global")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi), 0.0, 0.0), "ch0"
    )
    seq.measure()
    return seq


def complex_sequence() -> Sequence:
    seq = Sequence(two_atom_register(), modulated_mock_device())
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "rydberg_local", initial_target="atom0")

    def blackman_amp_pulse(duration: int) -> Pulse:
        return Pulse.ConstantDetuning(
            BlackmanWaveform(duration, np.pi), 0.0, 0.0
        )

    seq.add(blackman_amp_pulse(1000), "ch0")
    seq.target("atom0", "ch1")
    seq.add(blackman_amp_pulse(500), "ch1")
    seq.target("atom1", "ch1")
    seq.add(blackman_amp_pulse(500), "ch1")
    seq.add(blackman_amp_pulse(500), "ch1")
    seq.add(blackman_amp_pulse(1000), "ch0")

    seq.measure()
    return seq


def test_simulation_modulation():
    seq = complex_sequence()
    sim = Simulation(
        seq,
        samples_modulation=True,
    )

    assert sim._tot_duration == seq.get_duration() + 2 * max(
        [ch.rise_time for ch in seq.declared_channels.values()]
    )

    # The modulated blackman should give approximately the same final state
    sim_2 = Simulation(seq)
    res = sim.run()
    res_2 = sim_2.run()
    np.testing.assert_allclose(
        res.get_final_state(), res_2.get_final_state(), atol=1e-3
    )
