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

import dataclasses
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
import qutip

from pulser import Pulse, Register, Sequence
from pulser.devices import AnalogDevice, DigitalAnalogDevice, MockDevice
from pulser.register.register_layout import RegisterLayout
from pulser.sampler import sampler
from pulser.waveforms import BlackmanWaveform, ConstantWaveform, RampWaveform
from pulser_simulation import QutipEmulator, SimConfig


@pytest.fixture
def reg():
    q_dict = {
        "control1": np.array([-4.0, 0.0]),
        "target": np.array([0.0, 4.0]),
        "control2": np.array([4.0, 0.0]),
    }
    return Register(q_dict)


duration = 1000
pi_pulse = Pulse.ConstantDetuning(BlackmanWaveform(duration, np.pi), 0.0, 0)
twopi_pulse = Pulse.ConstantDetuning(
    BlackmanWaveform(duration, 2 * np.pi), 0.0, 0
)
pi_Y_pulse = Pulse.ConstantDetuning(
    BlackmanWaveform(duration, np.pi), 0.0, -np.pi / 2
)


@pytest.fixture
def seq_digital(reg):
    seq = Sequence(reg, DigitalAnalogDevice)
    # Declare Channels
    seq.declare_channel("raman", "raman_local", "control1")

    # Prepare state 'hhh':
    seq.add(pi_Y_pulse, "raman")
    seq.target("target", "raman")
    seq.add(pi_Y_pulse, "raman")
    seq.target("control2", "raman")
    seq.add(pi_Y_pulse, "raman")
    return seq


@pytest.fixture
def seq(seq_digital):
    # Write CCZ sequence:
    with pytest.warns(
        UserWarning, match="Building a non-parametrized sequence"
    ):
        seq = seq_digital.build()
    seq.declare_channel("ryd", "rydberg_local", "control1")
    seq.add(pi_pulse, "ryd", protocol="wait-for-all")
    seq.target("control2", "ryd")
    seq.add(pi_pulse, "ryd")
    seq.target("target", "ryd")
    seq.add(twopi_pulse, "ryd")
    seq.target("control2", "ryd")
    seq.add(pi_pulse, "ryd")
    seq.target("control1", "ryd")
    seq.add(pi_pulse, "ryd")

    # Add a ConstantWaveform part to testout the drawing procedure
    seq.add(Pulse.ConstantPulse(duration, 1, 0, 0), "ryd")
    return seq


@pytest.fixture
def matrices():
    pauli = {}
    pauli["I"] = qutip.qeye(2)
    pauli["X"] = qutip.sigmax()
    pauli["Y"] = qutip.sigmay()
    pauli["Z"] = qutip.sigmaz()
    pauli["I3"] = qutip.qeye(3)
    pauli["Z3"] = qutip.Qobj([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    return pauli


def test_initialization_and_construction_of_hamiltonian(seq, mod_device):
    fake_sequence = {"pulse1": "fake", "pulse2": "fake"}
    with pytest.raises(TypeError, match="sequence has to be a valid"):
        QutipEmulator.from_sequence(fake_sequence)
    with pytest.raises(TypeError, match="sequence has to be a valid"):
        QutipEmulator(fake_sequence, Register.square(2), mod_device)
    # Simulation cannot be run on a register not defining "control1"
    with pytest.raises(
        ValueError,
        match="The ids of qubits targeted in Local channels",
    ):
        QutipEmulator(
            sampler.sample(seq),
            Register(
                {
                    "target": np.array([0.0, 0.0]),
                    "control2": np.array([1.0, 0.0]),
                }
            ),
            MockDevice,
        )
    sim = QutipEmulator.from_sequence(seq, sampling_rate=0.011)
    sampled_seq = sampler.sample(seq)
    ext_sampled_seq = sampled_seq.extend_duration(sampled_seq.max_duration + 1)
    assert np.all(
        [
            np.equal(
                sim.samples_obj.channel_samples[ch].amp,
                ext_sampled_seq.channel_samples[ch].amp,
            )
            for ch in sampled_seq.channels
        ]
    )
    assert np.all(
        [
            np.equal(
                sim.samples_obj.channel_samples[ch].det,
                ext_sampled_seq.channel_samples[ch].det,
            )
            for ch in sampled_seq.channels
        ]
    )
    assert np.all(
        [
            np.equal(
                sim.samples_obj.channel_samples[ch].phase,
                ext_sampled_seq.channel_samples[ch].phase,
            )
            for ch in sampled_seq.channels
        ]
    )
    assert Register(sim._hamiltonian._qdict) == Register(seq.qubit_info)
    assert sim._hamiltonian._size == len(seq.qubit_info)
    assert sim._tot_duration == 9000  # seq has 9 pulses of 1µs
    assert sim._hamiltonian._qid_index == {
        "control1": 0,
        "target": 1,
        "control2": 2,
    }

    with pytest.raises(ValueError, match="too small, less than"):
        QutipEmulator.from_sequence(seq, sampling_rate=0.0001)
    with pytest.raises(ValueError, match="`sampling_rate`"):
        QutipEmulator.from_sequence(seq, sampling_rate=5)
    with pytest.raises(ValueError, match="`sampling_rate`"):
        QutipEmulator.from_sequence(seq, sampling_rate=-1)

    assert sim._sampling_rate == 0.011
    assert len(sim.sampling_times) == int(
        sim._sampling_rate * sim._tot_duration
    )

    assert isinstance(sim._hamiltonian._hamiltonian, qutip.QobjEvo)

    assert not seq.is_parametrized()
    with pytest.warns(UserWarning, match="returns a copy of itself"):
        seq_copy = seq.build()  # Take a copy of the sequence
    x = seq_copy.declare_variable("x")
    seq_copy.add(Pulse.ConstantPulse(x, 1, 0, 0), "ryd")
    assert seq_copy.is_parametrized()
    with pytest.raises(ValueError, match="needs to be built"):
        QutipEmulator.from_sequence(seq_copy)

    layout = RegisterLayout([[0, 0], [10, 10]])
    mapp_reg = layout.make_mappable_register(1)
    seq_ = Sequence(mapp_reg, DigitalAnalogDevice)
    assert seq_.is_register_mappable() and not seq_.is_parametrized()
    with pytest.raises(ValueError, match="needs to be built"):
        QutipEmulator.from_sequence(seq_)


def test_extraction_of_sequences(seq):
    sim = QutipEmulator.from_sequence(seq)
    for channel in seq.declared_channels:
        addr = seq.declared_channels[channel].addressing
        basis = seq.declared_channels[channel].basis

        if addr == "Global":
            for slot in seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    samples = sim._hamiltonian.samples[addr][basis]
                    assert np.all(
                        samples["amp"][slot.ti : slot.tf]
                        == slot.type.amplitude.samples
                    )
                    assert np.all(
                        samples["det"][slot.ti : slot.tf]
                        == slot.type.detuning.samples
                    )
                    assert np.all(
                        samples["phase"][slot.ti : slot.tf] == slot.type.phase
                    )

        elif addr == "Local":
            for slot in seq._schedule[channel]:
                if isinstance(slot.type, Pulse):
                    for qubit in slot.targets:  # TO DO: multiaddressing??
                        samples = sim._hamiltonian.samples[addr][basis][qubit]
                        assert np.all(
                            samples["amp"][slot.ti : slot.tf]
                            == slot.type.amplitude.samples
                        )
                        assert np.all(
                            samples["det"][slot.ti : slot.tf]
                            == slot.type.detuning.samples
                        )
                        assert np.all(
                            samples["phase"][slot.ti : slot.tf]
                            == slot.type.phase
                        )


@pytest.mark.parametrize("leakage", [False, True])
def test_building_basis_and_projection_operators(seq, reg, leakage, matrices):
    # All three levels:
    def _config(dim):
        return (
            SimConfig(
                ("leakage", "eff_noise"),
                eff_noise_opers=[qutip.qeye(dim)],
                eff_noise_rates=[0.0],
            )
            if leakage
            else SimConfig()
        )

    dim = 3 + leakage
    sim = QutipEmulator.from_sequence(
        seq, sampling_rate=0.01, config=_config(dim)
    )
    assert sim.basis_name == "all" + ("_with_error" if leakage else "")
    assert sim.dim == dim
    basis_dict = {
        "r": qutip.basis(dim, 0),
        "g": qutip.basis(dim, 1),
        "h": qutip.basis(dim, 2),
    }
    if leakage:
        basis_dict["x"] = qutip.basis(dim, 3)
    assert sim.basis == basis_dict
    assert (
        sim._hamiltonian.op_matrix["sigma_rr"]
        == qutip.basis(dim, 0) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim._hamiltonian.op_matrix["sigma_gr"]
        == qutip.basis(dim, 1) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim._hamiltonian.op_matrix["sigma_hg"]
        == qutip.basis(dim, 2) * qutip.basis(dim, 1).dag()
    )
    if leakage:
        assert (
            sim._hamiltonian.op_matrix["sigma_xr"]
            == qutip.basis(dim, 3) * qutip.basis(dim, 0).dag()
        )
    # Check local operator building method:
    with pytest.raises(ValueError, match="Duplicate atom"):
        sim.build_operator([("sigma_gg", ["target", "target"])])
    with pytest.raises(ValueError, match="not a valid operator"):
        sim.build_operator([("wrong", ["target"])])
    with pytest.raises(ValueError, match="Invalid qubit names: {'wrong'}"):
        sim.build_operator([("sigma_gg", ["wrong"])])

    # Check building operator with one operator
    op_standard = sim.build_operator([("sigma_gg", ["target"])])
    op_one = sim.build_operator(("sigma_gg", ["target"]))
    assert (op_standard - op_one).norm() < 1e-10

    # Global ground-rydberg
    seq2 = Sequence(reg, DigitalAnalogDevice)
    seq2.declare_channel("global", "rydberg_global")
    pi_pls = Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi), 0.0, 0)
    seq2.add(pi_pls, "global")
    dim = 2 + leakage
    sim2 = QutipEmulator.from_sequence(
        seq2, sampling_rate=0.01, config=_config(dim)
    )
    assert sim2.basis_name == "ground-rydberg" + (
        "_with_error" if leakage else ""
    )
    assert sim2.dim == dim
    basis_dict = {"r": qutip.basis(dim, 0), "g": qutip.basis(dim, 1)}
    if leakage:
        basis_dict["x"] = qutip.basis(dim, 2)
    assert sim2.basis == basis_dict
    assert (
        sim2._hamiltonian.op_matrix["sigma_rr"]
        == qutip.basis(dim, 0) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim2._hamiltonian.op_matrix["sigma_gr"]
        == qutip.basis(dim, 1) * qutip.basis(dim, 0).dag()
    )
    if leakage:
        assert (
            sim2._hamiltonian.op_matrix["sigma_xr"]
            == qutip.basis(dim, 2) * qutip.basis(dim, 0).dag()
        )
    # Digital
    seq2b = Sequence(reg, DigitalAnalogDevice)
    seq2b.declare_channel("local", "raman_local", "target")
    seq2b.add(pi_pls, "local")
    sim2b = QutipEmulator.from_sequence(
        seq2b, sampling_rate=0.01, config=_config(dim)
    )
    assert sim2b.basis_name == "digital" + ("_with_error" if leakage else "")
    assert sim2b.dim == dim
    basis_dict = {"g": qutip.basis(dim, 0), "h": qutip.basis(dim, 1)}
    if leakage:
        basis_dict["x"] = qutip.basis(dim, 2)
    assert sim2b.basis == basis_dict
    assert (
        sim2b._hamiltonian.op_matrix["sigma_gg"]
        == qutip.basis(dim, 0) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim2b._hamiltonian.op_matrix["sigma_hg"]
        == qutip.basis(dim, 1) * qutip.basis(dim, 0).dag()
    )
    if leakage:
        assert (
            sim2b._hamiltonian.op_matrix["sigma_xh"]
            == qutip.basis(dim, 2) * qutip.basis(dim, 1).dag()
        )

    # Local ground-rydberg
    seq2c = Sequence(reg, DigitalAnalogDevice)
    seq2c.declare_channel("local_ryd", "rydberg_local", "target")
    seq2c.add(pi_pls, "local_ryd")
    sim2c = QutipEmulator.from_sequence(
        seq2c, sampling_rate=0.01, config=_config(dim)
    )
    assert sim2c.basis_name == "ground-rydberg" + (
        "_with_error" if leakage else ""
    )
    assert sim2c.dim == dim
    basis_dict = {"r": qutip.basis(dim, 0), "g": qutip.basis(dim, 1)}
    if leakage:
        basis_dict["x"] = qutip.basis(dim, 2)
    assert sim2c.basis == basis_dict
    assert (
        sim2c._hamiltonian.op_matrix["sigma_rr"]
        == qutip.basis(dim, 0) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim2c._hamiltonian.op_matrix["sigma_gr"]
        == qutip.basis(dim, 1) * qutip.basis(dim, 0).dag()
    )
    if leakage:
        assert (
            sim2c._hamiltonian.op_matrix["sigma_xg"]
            == qutip.basis(dim, 2) * qutip.basis(dim, 1).dag()
        )
    # Global XY
    seq2 = Sequence(reg, MockDevice)
    seq2.declare_channel("global", "mw_global")
    seq2.add(pi_pls, "global")
    # seq2 cannot be run on DigitalAnalogDevice because it does not support mw
    with pytest.raises(
        ValueError,
        match="Bases used in samples should be supported by device.",
    ):
        QutipEmulator(sampler.sample(seq2), seq2.register, DigitalAnalogDevice)
    sim2 = QutipEmulator.from_sequence(
        seq2, sampling_rate=0.01, config=_config(dim)
    )
    assert sim2.basis_name == "XY" + ("_with_error" if leakage else "")
    assert sim2.dim == dim
    basis_dict = {"u": qutip.basis(dim, 0), "d": qutip.basis(dim, 1)}
    if leakage:
        basis_dict["x"] = qutip.basis(dim, 2)
    assert sim2.basis == basis_dict
    assert (
        sim2._hamiltonian.op_matrix["sigma_uu"]
        == qutip.basis(dim, 0) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim2._hamiltonian.op_matrix["sigma_du"]
        == qutip.basis(dim, 1) * qutip.basis(dim, 0).dag()
    )
    assert (
        sim2._hamiltonian.op_matrix["sigma_ud"]
        == qutip.basis(dim, 0) * qutip.basis(dim, 1).dag()
    )
    if leakage:
        assert (
            sim2._hamiltonian.op_matrix["sigma_ux"]
            == qutip.basis(dim, 0) * qutip.basis(dim, 2).dag()
        )


def test_empty_sequences(reg):
    seq = Sequence(reg, MockDevice)
    with pytest.raises(ValueError, match="no declared channels"):
        QutipEmulator.from_sequence(seq)
    seq.declare_channel("ch0", "mw_global")
    with pytest.raises(ValueError, match="No instructions given"):
        QutipEmulator.from_sequence(seq)
    with pytest.raises(ValueError, match="SequenceSamples is empty"):
        QutipEmulator(sampler.sample(seq), seq.register, seq.device)

    seq = Sequence(reg, MockDevice)
    seq.declare_channel("test", "raman_local", "target")
    seq.declare_channel("test2", "rydberg_global")
    with pytest.raises(ValueError, match="No instructions given"):
        QutipEmulator.from_sequence(seq)

    seq.delay(100, "test")
    emu = QutipEmulator.from_sequence(seq, config=SimConfig(noise="SPAM"))
    assert not emu._hamiltonian.samples["Global"]
    for basis in emu._hamiltonian.samples["Local"]:
        for q in emu._hamiltonian.samples["Local"][basis]:
            for qty_values in emu._hamiltonian.samples["Local"][basis][
                q
            ].values():
                np.testing.assert_equal(qty_values, 0)


def test_get_hamiltonian():
    simple_reg = Register.from_coordinates([[10, 0], [0, 0]], prefix="atom")
    detun = 1.0
    rise = Pulse.ConstantDetuning(RampWaveform(1500, 0.0, 2.0), detun, 0.0)
    simple_seq = Sequence(simple_reg, DigitalAnalogDevice)
    simple_seq.declare_channel("ising", "rydberg_global")
    simple_seq.add(rise, "ising")

    simple_sim = QutipEmulator.from_sequence(simple_seq, sampling_rate=0.01)
    with pytest.raises(ValueError, match="less than or equal to"):
        simple_sim.get_hamiltonian(1650)
    with pytest.raises(ValueError, match="greater than or equal to"):
        simple_sim.get_hamiltonian(-10)
    # Constant detuning, so |rr><rr| term is C_6/r^6 - 2*detuning for any time
    simple_ham = simple_sim.get_hamiltonian(143)
    assert np.isclose(
        simple_ham[0, 0],
        DigitalAnalogDevice.interaction_coeff / 10**6 - 2 * detun,
    )

    np.random.seed(123)
    simple_sim_noise = QutipEmulator.from_sequence(
        simple_seq, config=SimConfig(noise="doppler", temperature=20000)
    )
    simple_ham_noise = simple_sim_noise.get_hamiltonian(144)
    np.testing.assert_allclose(
        simple_ham_noise.full(),
        np.array(
            [
                [
                    4.47984523 + 0.0j,
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
    )


def test_single_atom_simulation():
    one_reg = Register.from_coordinates([(0, 0)], "atom")
    one_seq = Sequence(one_reg, DigitalAnalogDevice)
    one_seq.declare_channel("ch0", "rydberg_global")
    one_seq.add(
        Pulse.ConstantDetuning(ConstantWaveform(16, 1.0), 1.0, 0), "ch0"
    )
    one_sim = QutipEmulator.from_sequence(one_seq)
    one_res = one_sim.run()
    assert one_res._size == one_sim._hamiltonian._size
    one_sim = QutipEmulator.from_sequence(one_seq, evaluation_times="Minimal")
    one_resb = one_sim.run()
    assert one_resb._size == one_sim._hamiltonian._size


def test_add_max_step_and_delays():
    reg = Register.from_coordinates([(0, 0)])
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ch", "rydberg_global")
    seq.delay(1500, "ch")
    seq.add(Pulse.ConstantDetuning(BlackmanWaveform(600, np.pi), 0, 0), "ch")
    seq.delay(2000, "ch")
    seq.add(
        Pulse.ConstantDetuning(BlackmanWaveform(600, np.pi / 2), 0, 0), "ch"
    )
    sim = QutipEmulator.from_sequence(seq)
    res_large_max_step = sim.run(max_step=1)
    res_auto_max_step = sim.run()
    r = qutip.basis(2, 0)
    occ_large = res_large_max_step.expect([r.proj()])[0]
    occ_auto = res_auto_max_step.expect([r.proj()])[0]
    assert np.isclose(occ_large[-1], 0, 1e-4)
    assert np.isclose(occ_auto[-1], 0.5, 1e-4)


def test_run(seq, patch_plt_show):
    sim = QutipEmulator.from_sequence(seq, sampling_rate=0.01)
    sim.set_config(SimConfig("SPAM", eta=0.0))
    with patch("matplotlib.pyplot.savefig"):
        sim.draw(draw_phase_area=True, fig_name="my_fig.pdf")
    bad_initial = np.array([1.0])
    good_initial_array = np.r_[
        1, np.zeros(sim.dim**sim._hamiltonian._size - 1)
    ]
    good_initial_qobj = qutip.tensor(
        [qutip.basis(sim.dim, 0) for _ in range(sim._hamiltonian._size)]
    )
    good_initial_qobj_no_dims = qutip.basis(sim.dim**sim._hamiltonian._size, 2)

    with pytest.raises(
        ValueError, match="Incompatible shape of initial state"
    ):
        sim.set_initial_state(bad_initial)

    with pytest.raises(
        ValueError, match="Incompatible shape of initial state"
    ):
        sim.set_initial_state(qutip.Qobj(bad_initial))

    sim.set_initial_state(good_initial_array)
    sim.run()
    sim.set_initial_state(good_initial_qobj)
    sim.run()
    sim.set_initial_state(good_initial_qobj_no_dims)
    sim.run()
    seq.measure("ground-rydberg")
    sim = QutipEmulator.from_sequence(seq, sampling_rate=0.01)
    sim.set_initial_state(good_initial_qobj_no_dims)

    sim.run()
    assert sim.samples_obj._measurement == "ground-rydberg"

    sim.run(progress_bar=True)
    sim.run(progress_bar=False)
    sim.run(progress_bar=None)
    with pytest.raises(
        ValueError,
        match="`progress_bar` must be a bool.",
    ):
        sim.run(progress_bar=1)

    sim.set_config(SimConfig("SPAM", eta=0.1))
    with pytest.raises(
        NotImplementedError,
        match="Can't combine state preparation errors with an initial state "
        "different from the ground.",
    ):
        sim.run()


def test_eval_times(seq):
    with pytest.raises(
        ValueError, match="evaluation_times float must be between 0 " "and 1."
    ):
        sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
        sim.set_evaluation_times(3.0)
    with pytest.raises(ValueError, match="Wrong evaluation time label."):
        sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
        sim.set_evaluation_times(123)
    with pytest.raises(ValueError, match="Wrong evaluation time label."):
        sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
        sim.set_evaluation_times("Best")

    with pytest.raises(
        ValueError,
        match="Provided evaluation-time list contains " "negative values.",
    ):
        sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
        sim.set_evaluation_times([-1, 0, sim.sampling_times[-2]])

    with pytest.raises(
        ValueError,
        match="Provided evaluation-time list extends "
        "further than sequence duration.",
    ):
        sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
        sim.set_evaluation_times([0, sim.sampling_times[-1] + 10])

    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    sim.set_evaluation_times("Full")
    assert sim._eval_times_instruction == "Full"
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        sim.sampling_times,
    )

    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    sim.set_evaluation_times("Minimal")
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        np.array([sim.sampling_times[0], sim._tot_duration / 1000]),
    )

    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    sim.set_evaluation_times(
        [
            0,
            sim.sampling_times[-3],
            sim._tot_duration / 1000,
        ]
    )
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        np.array([0, sim.sampling_times[-3], sim._tot_duration / 1000]),
    )

    sim.set_evaluation_times([])
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        np.array([0, sim._tot_duration / 1000]),
    )

    sim.set_evaluation_times(0.0001)
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        np.array([0, sim._tot_duration / 1000]),
    )

    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    sim.set_evaluation_times([sim.sampling_times[-10], sim.sampling_times[-3]])
    np.testing.assert_almost_equal(
        sim._eval_times_array,
        np.array(
            [
                0,
                sim.sampling_times[-10],
                sim.sampling_times[-3],
                sim._tot_duration / 1000,
            ]
        ),
    )

    sim = QutipEmulator.from_sequence(seq, sampling_rate=1.0)
    sim.set_evaluation_times(0.4)
    np.testing.assert_almost_equal(
        sim.sampling_times[
            np.linspace(
                0,
                len(sim.sampling_times) - 1,
                int(0.4 * len(sim.sampling_times)),
                dtype=int,
            )
        ],
        sim._eval_times_array,
    )


def test_config(matrices):
    np.random.seed(123)
    reg = Register.from_coordinates([(0, 0), (0, 5)], prefix="q")
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 2500
    pulse = Pulse.ConstantPulse(duration, np.pi, 0.0 * 2 * np.pi, 0)
    seq.add(pulse, "ch0")
    sim = QutipEmulator.from_sequence(seq, config=SimConfig(noise="SPAM"))
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
    assert sim._initial_state == qutip.tensor(
        [qutip.basis(2, 1) for _ in range(2)]
    )
    # Currently in ground state => initial state is extended without warning
    sim.set_config(
        SimConfig(
            noise=("leakage", "eff_noise"),
            eff_noise_opers=[matrices["Z3"]],
            eff_noise_rates=[0.1],
        )
    )
    assert sim._initial_state == qutip.tensor(
        [qutip.basis(3, 1) for _ in range(2)]
    )
    # Otherwise initial state is set to ground-state
    sim.set_initial_state(qutip.tensor([qutip.basis(3, 0) for _ in range(2)]))
    with pytest.warns(
        UserWarning,
        match="Current initial state's dimension does not match new dim",
    ):
        sim.set_config(SimConfig(noise="SPAM", eta=0.5))
    assert sim._initial_state == qutip.tensor(
        [qutip.basis(2, 1) for _ in range(2)]
    )


def test_noise(seq, matrices):
    np.random.seed(3)
    sim2 = QutipEmulator.from_sequence(
        seq, sampling_rate=0.01, config=SimConfig(noise=("SPAM"), eta=0.9)
    )
    assert sim2.run().sample_final_state() == Counter(
        {"000": 857, "110": 73, "100": 70}
    )
    with pytest.raises(NotImplementedError, match="Cannot include"):
        sim2.set_config(SimConfig(noise="depolarizing"))
    assert sim2.config.spam_dict == {
        "eta": 0.9,
        "epsilon": 0.01,
        "epsilon_prime": 0.05,
    }
    assert sim2._hamiltonian.samples["Global"] == {}
    assert any(sim2._hamiltonian._bad_atoms.values())
    for basis in ("ground-rydberg", "digital"):
        for t in sim2._hamiltonian._bad_atoms:
            if not sim2._hamiltonian._bad_atoms[t]:
                continue
            for qty in ("amp", "det", "phase"):
                assert np.all(
                    sim2._hamiltonian.samples["Local"][basis][t][qty] == 0.0
                )


def test_noise_with_zero_epsilons(seq, matrices):
    np.random.seed(3)
    sim = QutipEmulator.from_sequence(seq, sampling_rate=0.01)

    sim2 = QutipEmulator.from_sequence(
        seq,
        sampling_rate=0.01,
        config=SimConfig(
            noise=("SPAM"), eta=0.0, epsilon=0.0, epsilon_prime=0.0
        ),
    )
    assert sim2.config.noise == ()

    assert sim.run().sample_final_state() == sim2.run().sample_final_state()


@pytest.mark.parametrize(
    "noise, result, n_collapse_ops",
    [
        ("dephasing", {"0": 586, "1": 414}, 1),
        ("relaxation", {"0": 586, "1": 414}, 1),
        ("eff_noise", {"0": 586, "1": 414}, 1),
        ("depolarizing", {"0": 581, "1": 419}, 3),
        (("dephasing", "depolarizing", "relaxation"), {"0": 582, "1": 418}, 5),
        (("eff_noise", "dephasing"), {"0": 587, "1": 413}, 2),
        (("eff_noise", "leakage"), {"0": 586, "1": 414}, 1),
    ],
)
def test_noises_rydberg(matrices, noise, result, n_collapse_ops):
    np.random.seed(123)
    reg = Register.from_coordinates([(0, 0)], prefix="q")
    # Test with Rydberg Sequence
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 2500
    pulse = Pulse.ConstantPulse(duration, np.pi, 0, 0)
    seq.add(pulse, "ch0")
    sim = QutipEmulator.from_sequence(
        seq,
        sampling_rate=0.01,
        config=SimConfig(
            noise=noise,
            eff_noise_opers=[
                (
                    qutip.Qobj([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
                    if "leakage" in noise
                    else matrices["Z"]
                )
            ],
            eff_noise_rates=[0.1 if "leakage" in noise else 0.025],
        ),
    )
    res = sim.run()
    res_samples = res.sample_final_state()
    assert res_samples == Counter(result)
    assert len(sim._hamiltonian._collapse_ops) == n_collapse_ops
    trace_2 = np.trace((res.states[-1] ** 2).full())
    assert trace_2 < 1 and not np.isclose(trace_2, 1)
    if "leakage" in noise:
        state = res.get_final_state()
        assert np.all(np.isclose(state[2, :], np.zeros_like(state[2, :])))
        assert np.all(np.isclose(state[:, 2], np.zeros_like(state[:, 2])))


def test_relaxation_noise():
    seq = Sequence(Register({"q0": (0, 0)}), MockDevice)
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi), 0, 0), "ryd")
    seq.delay(10000, "ryd")

    sim = QutipEmulator.from_sequence(seq)
    sim.add_config(SimConfig(noise="relaxation", relaxation_rate=0.1))
    res = sim.run()
    start_samples = res.sample_state(1)
    ryd_pop = start_samples["1"]
    assert ryd_pop > start_samples.get("0", 0)
    # The Rydberg state population gradually decays
    for t_ in range(2, 10):
        new_ryd_pop = res.sample_state(t_)["1"]
        assert new_ryd_pop < ryd_pop
        ryd_pop = new_ryd_pop


deph_res = {"111": 978, "110": 11, "011": 6, "101": 5}
depo_res = {
    "111": 821,
    "110": 61,
    "011": 59,
    "101": 48,
    "100": 5,
    "001": 3,
    "010": 3,
}
deph_depo_res = {
    "111": 806,
    "110": 65,
    "011": 63,
    "101": 52,
    "100": 6,
    "001": 4,
    "010": 3,
    "000": 1,
}
eff_deph_res = {"111": 958, "110": 19, "011": 12, "101": 11}


@pytest.mark.parametrize(
    "noise, result, n_collapse_ops",
    [
        ("dephasing", deph_res, 1),
        ("eff_noise", deph_res, 1),
        ("depolarizing", depo_res, 3),
        (("dephasing", "depolarizing"), deph_depo_res, 4),
        (("eff_noise", "dephasing"), eff_deph_res, 2),
        (("eff_noise", "leakage"), deph_res, 1),
        (("eff_noise", "leakage", "dephasing"), eff_deph_res, 2),
    ],
)
def test_noises_digital(matrices, noise, result, n_collapse_ops, seq_digital):
    np.random.seed(123)
    # Test with Digital Sequence
    sim = QutipEmulator.from_sequence(
        seq_digital,  # resulting state should be hhh
        sampling_rate=0.01,
        config=SimConfig(
            noise=noise,
            hyperfine_dephasing_rate=0.05,
            eff_noise_opers=[
                (
                    qutip.Qobj([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                    if "leakage" in noise
                    else matrices["Z"]
                )
            ],
            eff_noise_rates=[0.1 if "leakage" in noise else 0.025],
        ),
    )

    with pytest.raises(
        ValueError,
        match="'relaxation' noise requires addressing of the 'ground-rydberg'",
    ):
        sim.set_config(SimConfig(noise="relaxation"))

    res = sim.run()
    res_samples = res.sample_final_state()
    assert res_samples == Counter(result)
    assert len(sim._hamiltonian._collapse_ops) == n_collapse_ops * len(
        seq_digital.register.qubits
    )
    trace_2 = np.trace((res.states[-1] ** 2).full())
    assert trace_2 < 1 and not np.isclose(trace_2, 1)
    if "leakage" in noise:
        state = res.get_final_state()
        assert np.all(np.isclose(state[2, :], np.zeros_like(state[2, :])))
        assert np.all(np.isclose(state[:, 2], np.zeros_like(state[:, 2])))


res_deph_relax = {
    "000": 412,
    "010": 230,
    "001": 176,
    "100": 174,
    "101": 7,
    "011": 1,
}


@pytest.mark.parametrize(
    "noise, result, n_collapse_ops",
    [
        ("dephasing", {"111": 958, "110": 19, "011": 12, "101": 11}, 2),
        ("eff_noise", {"111": 958, "110": 19, "011": 12, "101": 11}, 2),
        (
            "relaxation",
            {"000": 421, "010": 231, "001": 172, "100": 171, "101": 5},
            1,
        ),
        (("dephasing", "relaxation"), res_deph_relax, 3),
        (
            ("eff_noise", "dephasing"),
            {"111": 922, "110": 33, "011": 23, "101": 21, "100": 1},
            4,
        ),
        (
            ("eff_noise", "leakage"),
            {"111": 958, "110": 19, "011": 12, "101": 11},
            2,
        ),
    ],
)
def test_noises_all(matrices, reg, noise, result, n_collapse_ops, seq):
    # Test with Digital+Rydberg Sequence
    if "relaxation" in noise:
        # Bring the states to ggg
        seq.target("control1", "raman")
        seq.add(pi_Y_pulse, "raman")
        seq.target("target", "raman")
        seq.add(pi_Y_pulse, "raman")
        seq.target("control2", "raman")
        seq.add(pi_Y_pulse, "raman")
        # Apply a 2pi pulse on ggg
        seq.declare_channel("ryd_glob", "rydberg_global")
        seq.add(twopi_pulse, "ryd_glob")
        # Measure in the rydberg basis
        seq.measure()
    if "leakage" in noise:
        deph_op = qutip.Qobj(
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        hyp_deph_op = qutip.Qobj(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        )
    else:
        deph_op = qutip.Qobj([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        hyp_deph_op = qutip.Qobj([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    sim = QutipEmulator.from_sequence(
        seq,  # resulting state should be hhh
        sampling_rate=0.01,
        config=SimConfig(
            noise=noise,
            dephasing_rate=0.1,
            hyperfine_dephasing_rate=0.1,
            relaxation_rate=1.0,
            eff_noise_opers=[deph_op, hyp_deph_op],
            eff_noise_rates=[0.2, 0.2],
        ),
    )
    with pytest.raises(
        ValueError,
        match="Incompatible shape for effective noise operator n°0.",
    ):
        # Only raised if 'eff_noise' in noise
        sim.set_config(
            SimConfig(
                noise=("eff_noise",),
                eff_noise_opers=[matrices["Z"]],
                eff_noise_rates=[1.0],
            )
        )

    with pytest.raises(
        NotImplementedError,
        match="Cannot include depolarizing noise in all-basis.",
    ):
        sim.set_config(SimConfig(noise="depolarizing"))

    assert len(sim._hamiltonian._collapse_ops) == n_collapse_ops * len(
        seq.register.qubits
    )
    np.random.seed(123)
    res = sim.run()
    res_samples = res.sample_final_state()
    assert res_samples == Counter(result)
    trace_2 = np.trace((res.states[-1] ** 2).full())
    assert trace_2 < 1 and not np.isclose(trace_2, 1)
    if "leakage" in noise:
        state = res.get_final_state()
        assert np.all(np.isclose(state[3, :], np.zeros_like(state[3, :])))
        assert np.all(np.isclose(state[:, 3], np.zeros_like(state[:, 3])))


def test_add_config(matrices):
    reg = Register.from_coordinates([(0, 0)], prefix="q")
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 2500
    pulse = Pulse.ConstantPulse(duration, np.pi, 0.0 * 2 * np.pi, 0)
    seq.add(pulse, "ch0")
    sim = QutipEmulator.from_sequence(
        seq, sampling_rate=0.01, config=SimConfig(noise="SPAM", eta=0.5)
    )
    with pytest.raises(ValueError, match="is not a valid"):
        sim.add_config("bad_cfg")
    config = SimConfig(
        noise=(
            "SPAM",
            "doppler",
            "eff_noise",
        ),
        eff_noise_opers=[matrices["I"], matrices["X"]],
        eff_noise_rates=[0.4, 0.6],
        temperature=20000,
    )
    sim.add_config(config)
    assert (
        "doppler" in sim.config.noise
        and "SPAM" in sim.config.noise
        and "eff_noise" in sim.config.noise
    )
    assert sim.config.eta == 0.5
    assert sim.config.temperature == 20000.0e-6
    sim.set_config(SimConfig(noise="doppler", laser_waist=175.0))
    sim.add_config(
        SimConfig(
            noise=("SPAM", "amplitude", "dephasing"),
            laser_waist=172.0,
            amp_sigma=1e-2,
        )
    )
    assert (
        "amplitude" in sim.config.noise
        and "dephasing" in sim.config.noise
        and "SPAM" in sim.config.noise
    )
    assert sim.config.laser_waist == 172.0
    assert sim.config.amp_sigma == 1e-2
    sim.set_config(SimConfig(noise="SPAM", eta=0.5))
    sim.add_config(SimConfig(noise="depolarizing"))
    assert "depolarizing" in sim.config.noise
    assert sim._initial_state == qutip.basis(2, 1)
    # Currently in ground state => initial state is extended without warning
    sim.add_config(
        SimConfig(
            noise=("leakage", "eff_noise"),
            eff_noise_opers=[matrices["Z3"]],
            eff_noise_rates=[0.1],
        )
    )
    assert sim._initial_state == qutip.basis(3, 1)
    # Otherwise initial state is set to ground-state
    sim.set_config(SimConfig(noise="SPAM", eta=0.5))
    sim.set_initial_state(qutip.basis(2, 0))
    with pytest.warns(
        UserWarning,
        match="Current initial state's dimension does not match new dim",
    ):
        sim.add_config(
            SimConfig(
                noise=("leakage", "eff_noise"),
                eff_noise_opers=[matrices["Z3"]],
                eff_noise_rates=[0.1],
            )
        )
    assert sim._initial_state == qutip.basis(3, 1)


def test_concurrent_pulses():
    reg = Register({"q0": (0, 0)})
    seq = Sequence(reg, DigitalAnalogDevice)

    seq.declare_channel("ch_local", "rydberg_local", initial_target="q0")
    seq.declare_channel("ch_global", "rydberg_global")

    pulse = Pulse.ConstantPulse(20, 10, 0, 0)

    seq.add(pulse, "ch_local")
    seq.add(pulse, "ch_global", protocol="no-delay")

    # Clean simulation
    sim_no_noise = QutipEmulator.from_sequence(seq)

    # Noisy simulation
    sim_with_noise = QutipEmulator.from_sequence(seq)
    config_doppler = SimConfig(noise=("doppler"))
    sim_with_noise.set_config(config_doppler)

    for t in sim_no_noise.evaluation_times:
        ham_no_noise = sim_no_noise.get_hamiltonian(t)
        ham_with_noise = sim_with_noise.get_hamiltonian(t)
        assert ham_no_noise[0, 1] == ham_with_noise[0, 1]


def test_get_xy_hamiltonian():
    simple_reg = Register.from_coordinates(
        [[0, 10], [10, 0], [0, 0]], prefix="atom"
    )
    detun = 1.0
    amp = 3.0
    rise = Pulse.ConstantPulse(1500, amp, detun, 0.0)
    simple_seq = Sequence(simple_reg, MockDevice)
    simple_seq.declare_channel("ch0", "mw_global")
    simple_seq.set_magnetic_field(0, 1.0, 0.0)
    simple_seq.add(rise, "ch0")

    assert np.isclose(np.linalg.norm(simple_seq.magnetic_field[0:2]), 1)

    simple_sim = QutipEmulator.from_sequence(simple_seq, sampling_rate=0.03)
    with pytest.raises(
        ValueError, match="less than or equal to the sequence duration"
    ):
        simple_sim.get_hamiltonian(1650)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        simple_sim.get_hamiltonian(-10)
    # Constant detuning, so |ud><du| term is C_3/r^3 - 2*detuning for any time
    simple_ham = simple_sim.get_hamiltonian(143)
    assert simple_ham[1, 2] == MockDevice.interaction_coeff_xy / 10**3
    assert (
        np.abs(
            simple_ham[1, 4] - (-2 * MockDevice.interaction_coeff_xy / 10**3)
        )
        < 1e-10
    )
    assert simple_ham[0, 1] == 0.5 * amp
    # |udd><udd| -> 1 atom in |u>
    assert simple_ham[3, 3] == -detun
    # |udd><udd| -> 3 atom in |u>
    assert simple_ham[0, 0] == -3 * detun
    # |ddd><ddd| -> no atom in |u>
    assert simple_ham[7, 7] == 0


def test_run_xy():
    simple_reg = Register.from_coordinates([[10, 0], [0, 0]], prefix="atom")
    detun = 1.0
    amp = 3.0
    rise = Pulse.ConstantPulse(1500, amp, detun, 0.0)
    simple_seq = Sequence(simple_reg, MockDevice)
    simple_seq.declare_channel("ch0", "mw_global")
    simple_seq.add(rise, "ch0")

    sim = QutipEmulator.from_sequence(simple_seq, sampling_rate=0.01)

    good_initial_array = np.r_[
        1, np.zeros(sim.dim**sim._hamiltonian._size - 1)
    ]
    good_initial_qobj = qutip.tensor(
        [qutip.basis(sim.dim, 0) for _ in range(sim._hamiltonian._size)]
    )
    sim.set_initial_state(good_initial_array)
    assert sim.initial_state == good_initial_qobj
    sim.run()
    sim.set_initial_state(good_initial_qobj)
    sim.run()

    assert not sim.samples_obj._measurement
    simple_seq.measure(basis="XY")
    sim = QutipEmulator.from_sequence(simple_seq, sampling_rate=0.01)
    sim.run()
    assert sim.samples_obj._measurement == "XY"


res1 = {"0000": 950, "0100": 19, "0001": 21, "0010": 10}
res2 = {"0000": 944, "0010": 15, "1000": 33, "0100": 8}
res3 = {"0000": 950, "0100": 19, "0010": 10, "0001": 21}
res4 = {"0000": 951, "0100": 19, "1000": 30}


@pytest.mark.parametrize(
    "masked_qubit, noise, result, n_collapse_ops",
    [
        (None, "dephasing", res1, 1),
        (None, "eff_noise", res1, 1),
        (None, "leakage", res1, 1),
        (None, "depolarizing", res2, 3),
        ("atom0", "dephasing", res3, 1),
        ("atom1", "dephasing", res4, 1),
    ],
)
def test_noisy_xy(matrices, masked_qubit, noise, result, n_collapse_ops):
    np.random.seed(15092021)
    simple_reg = Register.square(2, prefix="atom")
    detun = 1.0
    amp = 3.0
    rise = Pulse.ConstantPulse(100, amp, detun, 0.0)
    seq = Sequence(simple_reg, MockDevice)
    seq.declare_channel("ch0", "mw_global")
    if masked_qubit is not None:
        seq.config_slm_mask([masked_qubit])
    seq.add(rise, "ch0")

    sim = QutipEmulator.from_sequence(seq, sampling_rate=0.1)
    with pytest.raises(
        NotImplementedError, match="mode 'XY' does not support simulation of"
    ):
        sim.set_config(SimConfig(("SPAM", "doppler")))
    with pytest.raises(ValueError, match="is not a valid"):
        sim._hamiltonian.set_config(SimConfig(("SPAM", "doppler")))
    with pytest.raises(
        NotImplementedError, match="mode 'XY' does not support simulation of"
    ):
        sim._hamiltonian.set_config(
            SimConfig(("SPAM", "doppler")).to_noise_model()
        )
    with pytest.raises(
        NotImplementedError, match="simulation of noise types: amplitude"
    ):
        sim.add_config(SimConfig("amplitude"))

    # SPAM simulation is implemented:
    sim.set_config(
        SimConfig(
            (
                ("SPAM", noise)
                if noise != "leakage"
                else ("SPAM", "leakage", "eff_noise")
            ),
            eta=0.4,
            eff_noise_opers=[
                matrices["Z"] if noise != "leakage" else matrices["Z3"]
            ],
            eff_noise_rates=[0.025],
        )
    )
    assert sim._hamiltonian._bad_atoms == {
        "atom0": True,
        "atom1": False,
        "atom2": True,
        "atom3": False,
    }
    assert (
        len(sim._hamiltonian._collapse_ops) // len(simple_reg.qubits)
        == n_collapse_ops
    )
    assert sim.run().sample_final_state() == Counter(result)


def test_mask_nopulses():
    """Check interaction between SLM mask and a simulation with no pulses."""
    reg = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    for channel_type in ["mw_global", "rydberg_global"]:
        seq_empty = Sequence(reg, MockDevice)
        if channel_type == "mw_global":
            seq_empty.set_magnetic_field(0, 1.0, 0.0)
        seq_empty.declare_channel("ch", channel_type)
        seq_empty.delay(duration=100, channel="ch")
        masked_qubits = ["q2"]
        seq_empty.config_slm_mask(masked_qubits)
        sim_empty = QutipEmulator.from_sequence(seq_empty)

        assert seq_empty._slm_mask_time == []
        assert sampler.sample(seq_empty)._slm_mask.end == 0
        assert sim_empty.samples_obj._slm_mask.end == 0


def test_mask_equals_remove_xy():
    """Check that masking is equivalent to removing the masked qubits in XY.

    A global MW pulse acting on three qubits of which one is masked, should be
    equivalent to acting on a register with only the two unmasked qubits.
    """
    reg_three = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    reg_two = Register({"q0": (0, 0), "q1": (10, 10)})
    pulse = Pulse.ConstantPulse(100, 10, 0, 0)

    # Masked simulation
    seq_masked = Sequence(reg_three, MockDevice)
    seq_masked.set_magnetic_field(0, 1.0, 0.0)
    seq_masked.declare_channel("ch_masked", "mw_global")
    masked_qubits = ["q2"]
    seq_masked.config_slm_mask(masked_qubits)
    seq_masked.add(pulse, "ch_masked")
    sim_masked = QutipEmulator.from_sequence(seq_masked)
    # Simulation cannot be run on a device not having an SLM mask
    with pytest.raises(
        ValueError,
        match="Samples use SLM mask but device does not have one.",
    ):
        QutipEmulator(sampler.sample(seq_masked), reg_three, AnalogDevice)
    # Simulation cannot be run on a register not defining "q2"
    with pytest.raises(
        ValueError,
        match="The ids of qubits targeted in SLM mask",
    ):
        QutipEmulator(sampler.sample(seq_masked), reg_two, MockDevice)
    # Simulation on reduced register
    seq_two = Sequence(reg_two, MockDevice)
    seq_two.set_magnetic_field(0, 1.0, 0.0)
    seq_two.declare_channel("ch_two", "mw_global")
    seq_two.add(pulse, "ch_two")
    sim_two = QutipEmulator.from_sequence(seq_two)

    # Check equality
    for t in sim_two.sampling_times:
        ham_masked = sim_masked.get_hamiltonian(t)
        ham_two = sim_two.get_hamiltonian(t)
        assert ham_masked == qutip.tensor(ham_two, qutip.qeye(2))


def test_mask_two_pulses_xy():
    """Similar to test_mask_equals_remove, but with more pulses afterwards.

    Three XY global pulses act on a three qubit register, with one qubit masked
    during the first pulse.
    """
    reg_three = Register({"q0": (0, 0), "q1": (10, 10), "q2": (-10, -10)})
    reg_two = Register({"q0": (0, 0), "q1": (10, 10)})
    pulse = Pulse.ConstantPulse(100, 10, 0, 0)
    no_pulse = Pulse.ConstantPulse(100, 0, 0, 0)

    # Masked simulation
    seq_masked = Sequence(reg_three, MockDevice)
    seq_masked.declare_channel("ch_masked", "mw_global")
    masked_qubits = ["q2"]
    seq_masked.config_slm_mask(masked_qubits)
    seq_masked.add(pulse, "ch_masked")  # First pulse: masked
    seq_masked.add(pulse, "ch_masked")  # Second pulse: unmasked
    seq_masked.add(pulse, "ch_masked")  # Third pulse: unmasked
    sim_masked = QutipEmulator.from_sequence(seq_masked)

    # Unmasked simulation on full register
    seq_three = Sequence(reg_three, MockDevice)
    seq_three.declare_channel("ch_three", "mw_global")
    seq_three.add(no_pulse, "ch_three")
    seq_three.add(pulse, "ch_three")
    seq_three.add(pulse, "ch_three")
    sim_three = QutipEmulator.from_sequence(seq_three)

    # Unmasked simulation on reduced register
    seq_two = Sequence(reg_two, MockDevice)
    seq_two.declare_channel("ch_two", "mw_global")
    seq_two.add(pulse, "ch_two")
    seq_two.add(no_pulse, "ch_two")
    seq_two.add(no_pulse, "ch_two")
    sim_two = QutipEmulator.from_sequence(seq_two)

    ti = seq_masked._slm_mask_time[0]
    tf = seq_masked._slm_mask_time[1]
    for t in sim_masked.sampling_times:
        ham_masked = sim_masked.get_hamiltonian(t)
        ham_three = sim_three.get_hamiltonian(t)
        ham_two = sim_two.get_hamiltonian(t)
        if ti <= t <= tf:
            assert ham_masked == qutip.tensor(ham_two, qutip.qeye(2))
        else:
            assert ham_masked == ham_three


def test_mask_local_channel():
    seq_ = Sequence(
        Register.square(2, prefix="q"),
        MockDevice,
    )
    seq_.declare_channel("rydberg_global", "rydberg_global")
    pulse = Pulse.ConstantPulse(1000, 10, 0, 0)
    seq_.config_slm_mask(["q0", "q3"])
    seq_.add(pulse, "rydberg_global")

    seq_.declare_channel("raman_local", "raman_local", initial_target="q0")
    pulse2 = Pulse.ConstantPulse(1000, 10, -5, np.pi)
    seq_.add(pulse2, "raman_local", protocol="no-delay")

    assert seq_._slm_mask_time == [0, 1000]
    assert seq_._slm_mask_targets == {"q0", "q3"}
    sim = QutipEmulator.from_sequence(seq_)
    assert np.array_equal(
        sim._hamiltonian.samples["Global"]["ground-rydberg"]["amp"],
        np.concatenate((pulse.amplitude.samples, [0])),
    )
    assert np.array_equal(
        sim._hamiltonian.samples["Global"]["ground-rydberg"]["det"],
        np.concatenate((pulse.detuning.samples, [0])),
    )
    assert np.all(
        sim._hamiltonian.samples["Global"]["ground-rydberg"]["phase"] == 0.0
    )
    qubits = ["q0", "q1", "q2", "q3"]
    masked_qubits = ["q0", "q3"]
    for q in qubits:
        if q in masked_qubits:
            assert np.array_equal(
                sim._hamiltonian.samples["Local"]["ground-rydberg"][q]["det"],
                np.concatenate((-10 * pulse.amplitude.samples, [0])),
            )
        else:
            assert np.all(
                sim._hamiltonian.samples["Local"]["ground-rydberg"][q]["det"]
                == 0.0
            )
        assert np.all(
            sim._hamiltonian.samples["Local"]["ground-rydberg"][q]["amp"]
            == 0.0
        )
        assert np.all(
            sim._hamiltonian.samples["Local"]["ground-rydberg"][q]["phase"]
            == 0.0
        )

    assert np.array_equal(
        sim._hamiltonian.samples["Local"]["digital"]["q0"]["amp"],
        np.concatenate((pulse2.amplitude.samples, [0])),
    )
    assert np.array_equal(
        sim._hamiltonian.samples["Local"]["digital"]["q0"]["det"],
        np.concatenate((pulse2.detuning.samples, [0])),
    )
    assert np.all(
        np.isclose(
            sim._hamiltonian.samples["Local"]["digital"]["q0"]["phase"],
            np.concatenate((np.pi * np.ones(1000), [0])),
        )
    )


def test_effective_size_intersection():
    simple_reg = Register.square(2, prefix="atom")
    rise = Pulse.ConstantPulse(1500, 0, 0, 0)
    for channel_type in ["mw_global", "rydberg_global"]:
        np.random.seed(15092021)
        seq = Sequence(simple_reg, MockDevice)
        seq.declare_channel("ch0", channel_type)
        seq.add(rise, "ch0")
        seq.config_slm_mask(["atom0"])

        sim = QutipEmulator.from_sequence(seq, sampling_rate=0.01)
        sim.set_config(SimConfig("SPAM", eta=0.4))
        assert sim._hamiltonian._bad_atoms == {
            "atom0": True,
            "atom1": False,
            "atom2": True,
            "atom3": False,
        }
        assert sim.get_hamiltonian(0) != 0 * sim.build_operator(
            [("I", "global")]
        )


@pytest.mark.parametrize(
    "channel_type",
    [
        "mw_global",
        "rydberg_global",
        "raman_global",
    ],
)
def test_effective_size_disjoint(channel_type):
    simple_reg = Register.square(2, prefix="atom")
    amp = 1
    rise = Pulse.ConstantPulse(1500, amp, 0, 0)
    np.random.seed(15092021)
    seq = Sequence(simple_reg, MockDevice)
    seq.declare_channel("ch0", channel_type)
    seq.add(rise, "ch0")
    seq.config_slm_mask(["atom1"])
    assert seq._slm_mask_time == [0, 1500]
    sim = QutipEmulator.from_sequence(seq, sampling_rate=0.01)
    sim.set_config(SimConfig("SPAM", eta=0.4))
    assert sim._hamiltonian._bad_atoms == {
        "atom0": True,
        "atom1": False,
        "atom2": True,
        "atom3": False,
    }
    if channel_type == "mw_global":
        assert sim.get_hamiltonian(0) == 0.5 * amp * sim.build_operator(
            [(qutip.sigmax(), ["atom3"])]
        )
    else:
        basis = (
            "ground-rydberg" if channel_type == "rydberg_global" else "digital"
        )
        assert np.array_equal(
            sim._hamiltonian.samples["Local"][basis]["atom1"]["amp"],
            np.concatenate((rise.amplitude.samples, [0])),
        )
        assert np.array_equal(
            sim._hamiltonian.samples["Local"][basis]["atom3"]["amp"],
            np.concatenate((rise.amplitude.samples, [0])),
        )
        # SLM
        assert np.all(
            sim._hamiltonian.samples["Local"]["ground-rydberg"]["atom1"]["det"]
            == -10 * np.concatenate((rise.amplitude.samples, [0]))
        )
        if channel_type == "raman_global":
            assert np.all(
                sim._hamiltonian.samples["Local"][basis]["atom1"]["det"] == 0.0
            )
        assert np.all(
            sim._hamiltonian.samples["Local"][basis]["atom3"]["det"] == 0.0
        )
        for q in ["atom1", "atom3"]:
            assert np.all(
                sim._hamiltonian.samples["Local"][basis][q]["phase"] == 0.0
            )


@pytest.mark.parametrize(
    "propagation_dir", (None, (1, 0, 0), (0, 1, 0), (0, 0, 1))
)
def test_simulation_with_modulation(
    mod_device, reg, propagation_dir, patch_plt_show
):
    channels = mod_device.channels
    channels["rydberg_global"] = dataclasses.replace(
        channels["rydberg_global"], propagation_dir=propagation_dir
    )
    mod_device = dataclasses.replace(
        mod_device, channel_objects=tuple(channels.values()), channel_ids=None
    )
    seq = Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.config_slm_mask({"control1"})
    pulse1 = Pulse.ConstantPulse(120, 1, 0, 2.0)
    seq.add(pulse1, "ch0")

    with pytest.raises(
        NotImplementedError,
        match="Simulation of sequences combining an SLM mask and output "
        "modulation is not supported.",
    ):
        QutipEmulator.from_sequence(seq, with_modulation=True)

    seq = Sequence(reg, mod_device)
    seq.declare_channel("ch0", "rydberg_global")
    seq.declare_channel("ch1", "raman_local", initial_target="target")
    seq.add(pulse1, "ch1")
    seq.target("control1", "ch1")
    seq.add(pulse1, "ch1")
    seq.add(pulse1, "ch0")
    ch1_obj = seq.declared_channels["ch1"]
    pulse1_mod_samples = ch1_obj.modulate(pulse1.amplitude.samples).as_array()
    mod_dt = pulse1.duration + pulse1.fall_time(ch1_obj)
    assert pulse1_mod_samples.size == mod_dt

    sim_config = SimConfig(("amplitude", "doppler"))
    sim = QutipEmulator.from_sequence(
        seq, with_modulation=True, config=sim_config
    )

    assert (
        sim._hamiltonian.samples["Global"] == {}
    )  # All samples stored in local
    raman_samples = sim._hamiltonian.samples["Local"]["digital"]
    # Local pulses
    for qid, time_slice in [
        ("target", slice(0, mod_dt)),
        ("control1", slice(mod_dt, 2 * mod_dt)),
    ]:
        np.testing.assert_allclose(
            raman_samples[qid]["amp"][time_slice],
            pulse1_mod_samples,
            atol=1e-2,
        )
        np.testing.assert_equal(
            raman_samples[qid]["det"][time_slice],
            sim._hamiltonian._doppler_detune[qid],
        )
        np.testing.assert_allclose(
            raman_samples[qid]["phase"][time_slice], float(pulse1.phase)
        )

    def pos_factor(qid):
        if propagation_dir is None or propagation_dir == (0, 1, 0):
            # Optical axis long y, x dicates the distance
            r = reg.qubits[qid].as_array()[0]
        elif propagation_dir == (1, 0, 0):
            # Optical axis long x, y dicates the distance
            r = reg.qubits[qid].as_array()[1]
        else:
            # Optical axis long z, use distance to origin
            assert propagation_dir == (0, 0, 1)
            r = np.linalg.norm(reg.qubits[qid].as_array())

        w0 = sim_config.laser_waist
        return np.exp(-((r / w0) ** 2))

    # Global pulse
    time_slice = slice(2 * mod_dt, 3 * mod_dt)
    rydberg_samples = sim._hamiltonian.samples["Local"]["ground-rydberg"]
    noise_amp_base = rydberg_samples["target"]["amp"][time_slice] / (
        pulse1_mod_samples * pos_factor("target")
    )
    for qid in reg.qubit_ids:
        np.testing.assert_allclose(
            rydberg_samples[qid]["amp"][time_slice],
            pulse1_mod_samples * noise_amp_base * pos_factor(qid),
        )
        np.testing.assert_equal(
            rydberg_samples[qid]["det"][time_slice],
            sim._hamiltonian._doppler_detune[qid],
        )
        np.testing.assert_allclose(
            rydberg_samples[qid]["phase"][time_slice], float(pulse1.phase)
        )
    # Drawing with modulation
    sim.draw()
