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
from copy import deepcopy

from collections import Counter

import numpy as np
import pytest
import qutip
from qutip.piqs import isdiagonal

from pulser import Sequence, Pulse, Register
from pulser.devices import Chadoq2
from pulser.waveforms import BlackmanWaveform
from pulser.simulation import Simulation, SimConfig
from pulser.simulation.simresults import CoherentResults, NoisyResults

np.random.seed(123)
q_dict = {
    "A": np.array([0.0, 0.0]),
    "B": np.array([0.0, 10.0]),
}
reg = Register(q_dict)

duration = 1000
pi = Pulse.ConstantDetuning(BlackmanWaveform(duration, np.pi), 0.0, 0)

seq = Sequence(reg, Chadoq2)

# Declare Channels
seq.declare_channel("ryd", "rydberg_global")
seq.add(pi, "ryd")
seq_no_meas = deepcopy(seq)
seq_no_meas_noisy = deepcopy(seq)
seq.measure("ground-rydberg")

sim = Simulation(seq)
cfg_noisy = SimConfig(noise=("SPAM", "doppler", "amplitude"))
sim_noisy = Simulation(seq, config=cfg_noisy)
results = sim.run()
results_noisy = sim_noisy.run()

state = qutip.tensor([qutip.basis(2, 0), qutip.basis(2, 0)])
ground = qutip.tensor([qutip.basis(2, 1), qutip.basis(2, 1)])


def test_initialization():
    with pytest.raises(ValueError, match="`basis_name` must be"):
        CoherentResults(state, 2, "bad_basis", None, [0])
    with pytest.raises(ValueError, match="`meas_basis` must be"):
        CoherentResults(
            state, 1, "ground-rydberg", [0], "wrong_measurement_basis"
        )
    with pytest.raises(ValueError, match="`basis_name` must be"):
        NoisyResults(state, 2, "bad_basis", [0], 123)
    with pytest.raises(
        ValueError, match="only values of 'epsilon' and 'epsilon_prime'"
    ):
        CoherentResults(
            state,
            1,
            "ground-rydberg",
            [0],
            "ground-rydberg",
            cfg_noisy.spam_dict,
        )

    assert results._dim == 2
    assert results._size == 2
    assert results._basis_name == "ground-rydberg"
    assert results._meas_basis == "ground-rydberg"
    assert results.states[0] == ground


def test_get_final_state():
    with pytest.raises(TypeError, match="Can't reduce"):
        results.get_final_state(reduce_to_basis="digital")
    assert (
        results.get_final_state(
            reduce_to_basis="ground-rydberg", ignore_global_phase=False
        )
        == results.states[-1].tidyup()
    )
    assert np.all(
        np.isclose(
            np.abs(results.get_final_state().full()),
            np.abs(results.states[-1].full()),
        )
    )

    seq_ = Sequence(reg, Chadoq2)
    seq_.declare_channel("ryd", "rydberg_global")
    seq_.declare_channel("ram", "raman_local", initial_target="A")
    seq_.add(pi, "ram")
    seq_.add(pi, "ram")
    seq_.add(pi, "ryd")

    sim_ = Simulation(seq_)
    results_ = sim_.run()

    with pytest.raises(ValueError, match="'reduce_to_basis' must be"):
        results_.get_final_state(reduce_to_basis="all")

    with pytest.raises(TypeError, match="Can't reduce to chosen basis"):
        results_.get_final_state(reduce_to_basis="digital")

    h_states = results_.get_final_state(
        reduce_to_basis="digital", tol=1, normalize=False
    ).eliminate_states([0])
    assert h_states.norm() < 3e-6

    assert np.all(
        np.isclose(
            np.abs(
                results_.get_final_state(
                    reduce_to_basis="ground-rydberg"
                ).full()
            ),
            np.abs(results.states[-1].full()),
            atol=1e-5,
        )
    )


def test_get_final_state_noisy():
    np.random.seed(123)
    seq_ = Sequence(reg, Chadoq2)
    seq_.declare_channel("ram", "raman_local", initial_target="A")
    seq_.add(pi, "ram")
    noisy_config = SimConfig(noise=("SPAM", "doppler"))
    sim_noisy = Simulation(seq_, config=noisy_config)
    res3 = sim_noisy.run()
    res3._meas_basis = "digital"
    final_state = res3.get_final_state()
    assert isdiagonal(final_state)
    res3._meas_basis = "ground-rydberg"
    assert (
        final_state[0, 0] == 0.06666666666666667 + 0j
        and final_state[2, 2] == 0.9333333333333333 + 0j
    )
    assert res3.states[-1] == final_state
    assert res3.results[-1] == Counter(
        {"10": 0.9333333333333333, "00": 0.06666666666666667}
    )


def test_get_state_float_time():
    with pytest.raises(IndexError, match="is absent from"):
        results.get_state(-1.0)
    with pytest.raises(IndexError, match="is absent from"):
        mean = (results._sim_times[-1] + results._sim_times[-2]) / 2
        diff = (results._sim_times[-1] - results._sim_times[-2]) / 2
        results.get_state(mean, t_tol=diff / 2)
    state = results.get_state(mean, t_tol=3 * diff / 2)
    assert state == results.get_state(results._sim_times[-2])
    assert np.isclose(
        state.full(),
        np.array(
            [
                [0.79602211 + 0.0j],
                [0.02417478 - 0.37829574j],
                [0.02417478 - 0.37829574j],
                [-0.27423657 - 0.06131009j],
            ]
        ),
    ).all()


def test_expect():
    with pytest.raises(TypeError, match="must be a list"):
        results.expect("bad_observable")
    with pytest.raises(TypeError, match="Incompatible type"):
        results.expect(["bad_observable"])
    with pytest.raises(ValueError, match="Incompatible shape"):
        results.expect([np.array(3)])
    reg_single = Register.from_coordinates([(0, 0)], prefix="q")
    seq_single = Sequence(reg_single, Chadoq2)
    seq_single.declare_channel("ryd", "rydberg_global")
    seq_single.add(pi, "ryd")
    sim_single = Simulation(seq_single)
    results_single = sim_single.run()
    op = [qutip.basis(2, 0).proj()]
    exp = results_single.expect(op)[0]
    assert np.isclose(exp[-1], 1)
    assert len(exp) == duration
    np.testing.assert_almost_equal(
        results_single._calc_pseudo_density(-1).full(),
        np.array([[1, 0], [0, 0]]),
    )

    config = SimConfig(noise="SPAM", eta=0)
    sim_single.set_config(config)
    sim_single.evaluation_times = "Minimal"
    results_single = sim_single.run()
    exp = results_single.expect(op)[0]
    assert len(exp) == 2
    assert isinstance(results_single, CoherentResults)
    assert results_single._meas_errors == {
        "epsilon": config.epsilon,
        "epsilon_prime": config.epsilon_prime,
    }
    # Probability of measuring 1 = probability of false positive
    assert np.isclose(exp[0], config.epsilon)
    # Probability of measuring 1 = 1 - probability of false negative
    assert np.isclose(exp[-1], 1 - config.epsilon_prime)
    np.testing.assert_almost_equal(
        results_single._calc_pseudo_density(-1).full(),
        np.array([[1 - config.epsilon_prime, 0], [0, config.epsilon_prime]]),
    )
    seq3dim = Sequence(reg, Chadoq2)
    seq3dim.declare_channel("ryd", "rydberg_global")
    seq3dim.declare_channel("ram", "raman_local", initial_target="A")
    seq3dim.add(pi, "ram")
    seq3dim.add(pi, "ryd")
    sim3dim = Simulation(seq3dim)
    exp3dim = sim3dim.run().expect(
        [qutip.tensor(qutip.basis(3, 0).proj(), qutip.qeye(3))]
    )
    assert np.isclose(exp3dim[0][-1], 1.89690200e-14)


def test_expect_noisy():
    np.random.seed(123)
    bad_op = qutip.tensor([qutip.qeye(2), qutip.sigmap()])
    with pytest.raises(ValueError, match="non-diagonal"):
        results_noisy.expect([bad_op])
    op = qutip.tensor([qutip.qeye(2), qutip.basis(2, 0).proj()])
    assert np.isclose(results_noisy.expect([op])[0][-1], 0.7733333333333334)


def test_plot():
    op = qutip.tensor([qutip.qeye(2), qutip.basis(2, 0).proj()])
    results_noisy.plot(op)
    results_noisy.plot(op, error_bars=False)
    results.plot(op)


def test_sample_final_state():
    np.random.seed(123)
    sim_no_meas = Simulation(seq_no_meas, config=SimConfig(runs=1))
    results_no_meas = sim_no_meas.run()
    assert results_no_meas.sample_final_state() == Counter(
        {"00": 77, "01": 140, "10": 167, "11": 616}
    )
    with pytest.raises(NotImplementedError, match="dimension > 3"):
        results_large_dim = deepcopy(results)
        results_large_dim._dim = 7
        results_large_dim.sample_final_state()

    sampling = results.sample_final_state(1234)
    assert len(sampling) == 4  # Check that all states were observed.

    results._meas_basis = "digital"
    sampling0 = results.sample_final_state(N_samples=911)
    assert sampling0 == {"00": 911}
    seq_no_meas.declare_channel("raman", "raman_local", "B")
    seq_no_meas.add(pi, "raman")
    res_3level = Simulation(seq_no_meas).run()
    # Raman pi pulse on one atom will not affect other,
    # even with global pi on rydberg
    assert len(res_3level.sample_final_state()) == 2
    res_3level._meas_basis = "ground-rydberg"
    sampling_three_levelB = res_3level.sample_final_state()
    # Rydberg will affect both:
    assert len(sampling_three_levelB) == 4


def test_sample_final_state_noisy():
    np.random.seed(123)
    assert results_noisy.sample_final_state(N_samples=1234) == Counter(
        {"11": 787, "10": 219, "01": 176, "00": 52}
    )
    res_3level = Simulation(
        seq_no_meas_noisy, config=SimConfig(noise=("SPAM", "doppler"), runs=10)
    )
    final_state = res_3level.run().states[-1]
    assert np.isclose(
        final_state.full(),
        np.array(
            [
                [0.62 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.16 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.18 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.04 + 0.0j],
            ]
        ),
    ).all()
