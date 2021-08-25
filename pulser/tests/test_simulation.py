# Copyright 2020 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import patch

from collections import Counter

import numpy as np
import pytest
import qutip

from pulser import Sequence, Pulse, Register
from pulser.devices import Chadoq2, MockDevice
from pulser.waveforms import BlackmanWaveform, RampWaveform, ConstantWaveform
from pulser.simulation import SimConfig, Simulation

q_dict = {
    "control1": np.array([-4.0, 0.0]),
    "target": np.array([0.0, 4.0]),
    "control2": np.array([4.0, 0.0]),
}
reg = Register(q_dict)

duration = 1000
pi = Pulse.ConstantDetuning(BlackmanWaveform(duration, np.pi), 0.0, 0)
twopi = Pulse.ConstantDetuning(BlackmanWaveform(duration, 2 * np.pi), 0.0, 0)
pi_Y = Pulse.ConstantDetuning(
    BlackmanWaveform(duration, np.pi), 0.0, -np.pi / 2
)

seq = Sequence(reg, Chadoq2)

# Declare Channels
seq.declare_channel("ryd", "rydberg_local", "control1")
seq.declare_channel("raman", "raman_local", "control1")

d = 0  # Pulse Duration

# Prepare state 'hhh':
seq.add(pi_Y, "raman")
seq.target("target", "raman")
seq.add(pi_Y, "raman")
seq.target("control2", "raman")
seq.add(pi_Y, "raman")
d += 3

prep_state = qutip.tensor([qutip.basis(3, 2) for _ in range(3)])

# Write CCZ sequence:
seq.add(pi, "ryd", protocol="wait-for-all")
seq.target("control2", "ryd")
seq.add(pi, "ryd")
seq.target("target", "ryd")
seq.add(twopi, "ryd")
seq.target("control2", "ryd")
seq.add(pi, "ryd")
seq.target("control1", "ryd")
seq.add(pi, "ryd")
d += 5

# Add a ConstantWaveform part to testout the drawing procedure
seq.add(Pulse.ConstantPulse(duration, 1, 0, 0), "ryd")
d += 1


def test_initialization_and_construction_of_hamiltonian():
    fake_sequence = {"pulse1": "fake", "pulse2": "fake"}
    with pytest.raises(TypeError, match="sequence has to be a valid"):
        Simulation(fake_sequence)
    sim = Simulation(seq, sampling_rate=0.011)
    assert sim._seq == seq
    assert sim._qdict == seq.qubit_info
    assert sim._size == len(seq.qubit_info)
    assert sim._tot_duration == duration * d
    assert sim._qid_index == {"control1": 0, "target": 1, "control2": 2}

    with pytest.raises(ValueError, match="too small, less than"):
        Simulation(seq, sampling_rate=0.0001)
    with pytest.raises(ValueError, match="`sampling_rate`"):
        Simulation(seq, sampling_rate=5)
    with pytest.raises(ValueError, match="`sampling_rate`"):
        Simulation(seq, sampling_rate=-1)

    assert sim._sampling_rate == 0.011
    assert len(sim._times) == int(sim._sampling_rate * sim._tot_duration)

    assert isinstance(sim._hamiltonian, qutip.QobjEvo)
    # Checks adapt() method:
    assert bool(set(sim._hamiltonian.tlist).intersection(sim._times))
    for qobjevo in sim._hamiltonian.ops:
        for sh in qobjevo.qobj.shape:
            assert sh == sim.dim ** sim._size


def test_extraction_of_sequences():
    sim = Simulation(seq)
    for channel in seq.declared_channels:
        addr = seq.declared_channels[channel].addressing
        basis = seq.declared_channels[channel].basis

        if addr == "Global":
            for slot in seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    samples = sim.samples[addr][basis]
                    assert (
                        samples["amp"][slot.ti : slot.tf]
                        == slot.type.amplitude.samples
                    ).all()
                    assert (
                        samples["det"][slot.ti : slot.tf]
                        == slot.type.detuning.samples
                    ).all()
                    assert (
                        samples["phase"][slot.ti : slot.tf] == slot.type.phase
                    ).all()

        elif addr == "Local":
            for slot in seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    for qubit in slot.targets:  # TO DO: multiaddressing??
                        samples = sim.samples[addr][basis][qubit]
                        assert (
                            samples["amp"][slot.ti : slot.tf]
                            == slot.type.amplitude.samples
                        ).all()
                        assert (
                            samples["det"][slot.ti : slot.tf]
                            == slot.type.detuning.samples
                        ).all()
                        assert (
                            samples["phase"][slot.ti : slot.tf]
                            == slot.type.phase
                        ).all()


def test_building_basis_and_projection_operators():
    # All three levels:
    sim = Simulation(seq, sampling_rate=0.01)
    assert sim.basis_name == "all"
    assert sim.dim == 3
    assert sim.basis == {
        "r": qutip.basis(3, 0),
        "g": qutip.basis(3, 1),
        "h": qutip.basis(3, 2),
    }
    assert (
        sim.op_matrix["sigma_rr"]
        == qutip.basis(3, 0) * qutip.basis(3, 0).dag()
    )
    assert (
        sim.op_matrix["sigma_gr"]
        == qutip.basis(3, 1) * qutip.basis(3, 0).dag()
    )
    assert (
        sim.op_matrix["sigma_hg"]
        == qutip.basis(3, 2) * qutip.basis(3, 1).dag()
    )

    # Check local operator building method:
    with pytest.raises(ValueError, match="Duplicate atom"):
        sim._build_operator("sigma_gg", "target", "target")

    # Global ground-rydberg
    seq2 = Sequence(reg, Chadoq2)
    seq2.declare_channel("global", "rydberg_global")
    seq2.add(pi, "global")
    sim2 = Simulation(seq2, sampling_rate=0.01)
    assert sim2.basis_name == "ground-rydberg"
    assert sim2.dim == 2
    assert sim2.basis == {"r": qutip.basis(2, 0), "g": qutip.basis(2, 1)}
    assert (
        sim2.op_matrix["sigma_rr"]
        == qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    )
    assert (
        sim2.op_matrix["sigma_gr"]
        == qutip.basis(2, 1) * qutip.basis(2, 0).dag()
    )

    # Digital
    seq2b = Sequence(reg, Chadoq2)
    seq2b.declare_channel("local", "raman_local", "target")
    seq2b.add(pi, "local")
    sim2b = Simulation(seq2b, sampling_rate=0.01)
    assert sim2b.basis_name == "digital"
    assert sim2b.dim == 2
    assert sim2b.basis == {"g": qutip.basis(2, 0), "h": qutip.basis(2, 1)}
    assert (
        sim2b.op_matrix["sigma_gg"]
        == qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    )
    assert (
        sim2b.op_matrix["sigma_hg"]
        == qutip.basis(2, 1) * qutip.basis(2, 0).dag()
    )

    # Local ground-rydberg
    seq2c = Sequence(reg, Chadoq2)
    seq2c.declare_channel("local_ryd", "rydberg_local", "target")
    seq2c.add(pi, "local_ryd")
    sim2c = Simulation(seq2c, sampling_rate=0.01)
    assert sim2c.basis_name == "ground-rydberg"
    assert sim2c.dim == 2
    assert sim2c.basis == {"r": qutip.basis(2, 0), "g": qutip.basis(2, 1)}
    assert (
        sim2c.op_matrix["sigma_rr"]
        == qutip.basis(2, 0) * qutip.basis(2, 0).dag()
    )
    assert (
        sim2c.op_matrix["sigma_gr"]
        == qutip.basis(2, 1) * qutip.basis(2, 0).dag()
    )


def test_empty_sequences():
    seq = Sequence(reg, Chadoq2)
    with pytest.raises(ValueError, match="no declared channels"):
        Simulation(seq)
    with pytest.raises(ValueError, match="No instructions given"):
        seq.declare_channel("test", "rydberg_local", "target")
        seq.declare_channel("test2", "rydberg_global")
        Simulation(seq)
    seqMW = Sequence(reg, MockDevice)
    with pytest.raises(NotImplementedError):
        seqMW.declare_channel("ch0", "mw_global")
        seqMW.add(
            Pulse.ConstantDetuning(RampWaveform(1500, 0.0, 2.0), 0.0, 0.0),
            "ch0",
        )
        Simulation(seqMW)


def test_get_hamiltonian():
    simple_reg = Register.from_coordinates([[10, 0], [0, 0]], prefix="atom")
    detun = 1.0
    rise = Pulse.ConstantDetuning(RampWaveform(1500, 0.0, 2.0), detun, 0.0)
    simple_seq = Sequence(simple_reg, Chadoq2)
    simple_seq.declare_channel("ising", "rydberg_global")
    simple_seq.add(rise, "ising")

    simple_sim = Simulation(simple_seq, sampling_rate=0.01)
    with pytest.raises(ValueError, match="less than or equal to"):
        simple_sim.get_hamiltonian(1650)
    with pytest.raises(ValueError, match="greater than or equal to"):
        simple_sim.get_hamiltonian(-10)
    # Constant detuning, so |rr><rr| term is C_6/r^6 - 2*detuning for any time
    simple_ham = simple_sim.get_hamiltonian(143)
    assert simple_ham[0, 0] == Chadoq2.interaction_coeff / 10 ** 6 - 2 * detun

    np.random.seed(123)
    simple_sim_noise = Simulation(
        simple_seq, config=SimConfig(noise="doppler", temperature=20000)
    )
    simple_ham_noise = simple_sim_noise.get_hamiltonian(144)
    assert np.isclose(
        simple_ham_noise.full(),
        np.array(
            [
                [
                    4.0683997 + 0.0j,
                    0.09606404 + 0.0j,
                    0.09606404 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.09606404 + 0.0j,
                    12.03082372 + 0.0j,
                    0.0 + 0.0j,
                    0.09606404 + 0.0j,
                ],
                [
                    0.09606404 + 0.0j,
                    0.0 + 0.0j,
                    -12.97113702 + 0.0j,
                    0.09606404 + 0.0j,
                ],
                [0.0 + 0.0j, 0.09606404 + 0.0j, 0.09606404 + 0.0j, 0.0 + 0.0j],
            ]
        ),
    ).all()


def test_single_atom_simulation():
    one_reg = Register.from_coordinates([(0, 0)], "atom")
    one_seq = Sequence(one_reg, Chadoq2)
    one_seq.declare_channel("ch0", "rydberg_global")
    one_seq.add(
        Pulse.ConstantDetuning(ConstantWaveform(16, 1.0), 1.0, 0), "ch0"
    )
    one_sim = Simulation(one_seq)
    one_res = one_sim.run()
    assert one_res._size == one_sim._size
    one_sim = Simulation(one_seq, evaluation_times="Minimal")
    one_resb = one_sim.run()
    assert one_resb._size == one_sim._size


def test_add_max_step_and_delays():
    reg = Register.from_coordinates([(0, 0)])
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ch", "rydberg_global")
    seq.delay(1500, "ch")
    seq.add(Pulse.ConstantDetuning(BlackmanWaveform(600, np.pi), 0, 0), "ch")
    seq.delay(2000, "ch")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(600, np.pi / 2), 0, 0), "ch"
    )
    sim = Simulation(seq)
    res_large_max_step = sim.run(max_step=1)
    res_auto_max_step = sim.run()
    r = qutip.basis(2, 0)
    occ_large = res_large_max_step.expect([r.proj()])[0]
    occ_auto = res_auto_max_step.expect([r.proj()])[0]
    assert np.isclose(occ_large[-1], 0, 1e-4)
    assert np.isclose(occ_auto[-1], 0.5, 1e-4)


def test_run():
    sim = Simulation(seq, sampling_rate=0.01)
    sim.set_config(SimConfig("SPAM", eta=0.0))
    with patch("matplotlib.pyplot.show"):
        sim.draw(draw_phase_area=True)
    bad_initial = np.array([1.0])
    good_initial_array = np.r_[1, np.zeros(sim.dim ** sim._size - 1)]
    good_initial_qobj = qutip.tensor(
        [qutip.basis(sim.dim, 0) for _ in range(sim._size)]
    )
    good_initial_qobj_no_dims = qutip.basis(sim.dim ** sim._size, 2)

    with pytest.raises(
        ValueError, match="Incompatible shape of initial state"
    ):
        sim.initial_state = bad_initial

    with pytest.raises(
        ValueError, match="Incompatible shape of initial state"
    ):
        sim.initial_state = qutip.Qobj(bad_initial)

    sim.initial_state = good_initial_array
    sim.run()
    sim.initial_state = good_initial_qobj
    sim.run()
    sim.initial_state = good_initial_qobj_no_dims
    sim.run()
    seq.measure("ground-rydberg")
    sim.run()
    assert sim._seq._measurement == "ground-rydberg"

    sim.set_config(SimConfig("SPAM", eta=0.1))
    with pytest.raises(
        NotImplementedError,
        match="Can't combine state preparation errors with an initial state "
        "different from the ground.",
    ):
        sim.run()


def test_eval_times():
    with pytest.raises(
        ValueError, match="evaluation_times float must be between 0 " "and 1."
    ):
        sim = Simulation(seq, sampling_rate=1.0)
        sim.evaluation_times = 3.0
    with pytest.raises(ValueError, match="Wrong evaluation time label."):

        sim = Simulation(seq, sampling_rate=1.0)
        sim.evaluation_times = 123
    with pytest.raises(ValueError, match="Wrong evaluation time label."):
        sim = Simulation(seq, sampling_rate=1.0)
        sim.evaluation_times = "Best"

    with pytest.raises(
        ValueError,
        match="Provided evaluation-time list contains " "negative values.",
    ):
        sim = Simulation(seq, sampling_rate=1.0)
        sim.evaluation_times = [-1, 0, sim._times[-2]]

    with pytest.raises(
        ValueError,
        match="Provided evaluation-time list extends "
        "further than sequence duration.",
    ):
        sim = Simulation(seq, sampling_rate=1.0)
        sim.evaluation_times = [0, sim._times[-1] + 10]

    sim = Simulation(seq, sampling_rate=1.0)
    sim.evaluation_times = "Full"
    assert sim.evaluation_times == "Full"
    np.testing.assert_almost_equal(sim._eval_times_array, sim._times)

    sim = Simulation(seq, sampling_rate=1.0)
    sim.evaluation_times = "Minimal"
    np.testing.assert_almost_equal(
        sim._eval_times_array, np.array([sim._times[0], sim._times[-1]])
    )

    sim = Simulation(seq, sampling_rate=1.0)
    sim.evaluation_times = [0, sim._times[-3], sim._times[-1]]
    np.testing.assert_almost_equal(
        sim._eval_times_array, np.array([0, sim._times[-3], sim._times[-1]])
    )

    sim = Simulation(seq, sampling_rate=1.0)
    sim.evaluation_times = [sim._times[-10], sim._times[-3]]
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        np.array([0, sim._times[-10], sim._times[-3], sim._times[-1]]),
    )

    sim = Simulation(seq, sampling_rate=1.0)
    sim.evaluation_times = 0.4
    np.testing.assert_almost_equal(
        sim._times[
            np.linspace(
                0, len(sim._times) - 1, int(0.4 * len(sim._times)), dtype=int
            )
        ],
        sim._eval_times_array,
    )


def test_config():
    np.random.seed(123)
    reg = Register.from_coordinates([(0, 0), (0, 5)], prefix="q")
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 2500
    pulse = Pulse.ConstantPulse(duration, np.pi, 0.0 * 2 * np.pi, 0)
    seq.add(pulse, "ch0")
    sim = Simulation(seq, config=SimConfig(noise="SPAM"))
    sim.reset_config()
    assert sim.config == SimConfig()
    sim.show_config()
    with pytest.raises(ValueError, match="not a valid"):
        sim.set_config("bad_config")
    clean_ham = sim.get_hamiltonian(123)
    new_cfg = SimConfig(noise="doppler", temperature=10000)
    sim.set_config(new_cfg)
    assert sim.config == new_cfg
    noisy_ham = sim.get_hamiltonian(123)
    assert (
        noisy_ham[0, 0] != clean_ham[0, 0]
        and noisy_ham[3, 3] == clean_ham[3, 3]
    )
    sim.set_config(SimConfig(noise="amplitude"))
    noisy_amp_ham = sim.get_hamiltonian(123)
    assert (
        noisy_amp_ham[0, 0] == clean_ham[0, 0]
        and noisy_amp_ham[0, 1] != clean_ham[0, 1]
    )


def test_noise():
    sim2 = Simulation(
        seq, sampling_rate=0.01, config=SimConfig(noise=("doppler"))
    )
    sim2.run()
    with pytest.raises(NotImplementedError, match="Cannot include"):
        sim2.set_config(SimConfig(noise="dephasing"))
        sim2.run()
    assert sim2.config.spam_dict == {
        "eta": 0.005,
        "epsilon": 0.01,
        "epsilon_prime": 0.05,
    }


def test_dephasing():
    np.random.seed(123)
    reg = Register.from_coordinates([(0, 0)], prefix="q")
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 2500
    pulse = Pulse.ConstantPulse(duration, np.pi, 0.0 * 2 * np.pi, 0)
    seq.add(pulse, "ch0")
    sim = Simulation(
        seq, sampling_rate=0.01, config=SimConfig(noise="dephasing")
    )
    assert sim.run().sample_final_state() == Counter({"0": 482, "1": 518})
    assert len(sim._collapse_ops) != 0
    with pytest.warns(UserWarning, match="first-order"):
        reg = Register.from_coordinates([(0, 0), (0, 10)], prefix="q")
        seq2 = Sequence(reg, Chadoq2)
        seq2.declare_channel("ch0", "rydberg_global")
        duration = 2500
        pulse = Pulse.ConstantPulse(duration, np.pi, 0.0 * 2 * np.pi, 0)
        seq2.add(pulse, "ch0")
        sim = Simulation(
            seq2,
            sampling_rate=0.01,
            config=SimConfig(noise="dephasing", dephasing_prob=0.5),
        )


def test_add_config():
    reg = Register.from_coordinates([(0, 0)], prefix="q")
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 2500
    pulse = Pulse.ConstantPulse(duration, np.pi, 0.0 * 2 * np.pi, 0)
    seq.add(pulse, "ch0")
    sim = Simulation(
        seq, sampling_rate=0.01, config=SimConfig(noise="SPAM", eta=0.5)
    )
    with pytest.raises(ValueError, match="is not a valid"):
        sim.add_config("bad_cfg")
    sim.add_config(
        SimConfig(noise=("dephasing", "SPAM", "doppler"), temperature=20000)
    )
    assert "dephasing" in sim.config.noise and "SPAM" in sim.config.noise
    assert sim.config.eta == 0.5
    assert sim.config.temperature == 20000.0e-6
    sim.set_config(SimConfig(noise="dephasing", laser_waist=175.0))
    sim.add_config(SimConfig(noise=("SPAM", "amplitude"), laser_waist=172.0))
    assert "amplitude" in sim.config.noise and "SPAM" in sim.config.noise
    assert sim.config.laser_waist == 172.0


def test_cuncurrent_pulses():
    reg = Register({"q0": (0, 0)})
    seq = Sequence(reg, Chadoq2)

    seq.declare_channel("ch_local", "rydberg_local", initial_target="q0")
    seq.declare_channel("ch_global", "rydberg_global")

    pulse = Pulse.ConstantPulse(20, 10, 0, 0)

    seq.add(pulse, "ch_local")
    seq.add(pulse, "ch_global", protocol="no-delay")

    # Clean simulation
    sim_no_noise = Simulation(seq)

    # Noisy simulation
    sim_with_noise = Simulation(seq)
    config_doppler = SimConfig(noise=("doppler"))
    sim_with_noise.set_config(config_doppler)

    for t in sim_no_noise._times:
        ham_no_noise = sim_no_noise.get_hamiltonian(t)
        ham_with_noise = sim_with_noise.get_hamiltonian(t)
        assert ham_no_noise[0, 1] == ham_with_noise[0, 1]
