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
from collections import Counter
from typing import cast

import numpy as np
import pytest
import qutip
from qutip.piqs.piqs import isdiagonal

from pulser import AnalogDevice, NoiseModel, Pulse, Register, Sequence
from pulser.devices import DigitalAnalogDevice, MockDevice
from pulser.waveforms import (
    BlackmanWaveform,
    CompositeWaveform,
    ConstantWaveform,
)
from pulser_simulation import QutipEmulator
from pulser_simulation.simresults import CoherentResults, NoisyResults


@pytest.fixture
def reg():
    q_dict = {
        "A": np.array([0.0, 0.0]),
        "B": np.array([0.0, 10.0]),
    }
    return Register(q_dict)


@pytest.fixture
def pi_pulse():
    return Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi), 0.0, 0)


@pytest.fixture
def seq_no_meas(reg, pi_pulse):
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(pi_pulse, "ryd")
    return seq


@pytest.fixture
def sim(seq_no_meas):
    seq_no_meas.measure("ground-rydberg")
    np.random.seed(123)
    return QutipEmulator.from_sequence(seq_no_meas)


@pytest.fixture
def results_noisy(seq_no_meas):
    np.random.seed(123)
    with pytest.warns(
        DeprecationWarning, match="Setting samples_per_run different to 1 is"
    ):
        sim = QutipEmulator.from_sequence(
            seq_no_meas,
            noise_model=NoiseModel(
                runs=15,
                samples_per_run=5,
                temperature=50.0,
                state_prep_error=0.005,
                p_false_pos=0.01,
                p_false_neg=0.05,
                amp_sigma=1e-3,
                laser_waist=175.0,
            ),
        )
    return sim.run()


@pytest.fixture
def results(sim):
    return sim.run()


@pytest.mark.parametrize(
    ["basis", "exp_basis"],
    [
        ("ground-rydberg_with_error", "ground-rydberg"),
        ("digital_with_error", "digital"),
        ("all_with_error", "digital"),
        ("all", "digital"),
        ("XY_with_error", "XY"),
    ],
)
def test_initialization(results, basis, exp_basis):
    rr_state = qutip.tensor([qutip.basis(2, 0), qutip.basis(2, 0)])
    with pytest.raises(ValueError, match="`basis_name` must be"):
        CoherentResults(rr_state, 2, "bad_basis", None, [0])
    if "all" in basis:
        with pytest.raises(
            ValueError,
            match="`meas_basis` must be 'ground-rydberg' or 'digital'.",
        ):
            CoherentResults(rr_state, 1, basis, None, "XY")
    else:
        with pytest.raises(
            ValueError,
            match=f"`meas_basis` associated to basis_name '{basis}' must be",
        ):
            CoherentResults(rr_state, 1, basis, [0], "wrong_measurement_basis")
    with pytest.raises(
        ValueError, match="only values of 'epsilon' and 'epsilon_prime'"
    ):
        CoherentResults(
            rr_state,
            1,
            basis,
            [0],
            exp_basis,
            {"eta": 0.1, "epsilon": 0.0, "epsilon_prime": 0.4},
        )

    assert results._dim == 2
    assert results._size == 2
    assert results._basis_name == "ground-rydberg"
    assert results._meas_basis == "ground-rydberg"
    assert results.states[0] == qutip.tensor(
        [qutip.basis(2, 1), qutip.basis(2, 1)]
    )


@pytest.mark.parametrize(
    ["basis", "exp_basis"],
    [
        ("ground-rydberg_with_error", "ground-rydberg"),
        ("digital_with_error", "digital"),
        ("all_with_error", "digital"),
        ("all", "digital"),
        ("XY_with_error", "XY"),
    ],
)
def test_init_noisy(basis, exp_basis):
    state = qutip.tensor([qutip.basis(2, 0), qutip.basis(2, 0)])
    with pytest.raises(ValueError, match="`basis_name` must be"):
        NoisyResults(state, 2, "bad_basis", [0], 123)
    assert NoisyResults(state, 2, basis, [0], 100)._basis_name == exp_basis


@pytest.mark.parametrize("noisychannel", [True, False])
def test_get_final_state(
    noisychannel, seq_no_meas, sim: QutipEmulator, results, reg, pi_pulse
):
    if noisychannel:
        sim = QutipEmulator(
            sim.samples_obj,
            register=seq_no_meas.register,
            device=seq_no_meas.device,
            noise_model=NoiseModel(dephasing_rate=0.01),
        )
    _results = sim.run()
    assert isinstance(_results, CoherentResults)
    final_state = _results.get_final_state()
    assert final_state.isoper if noisychannel else final_state.isket
    with pytest.raises(TypeError, match="Can't reduce"):
        _results.get_final_state(reduce_to_basis="digital")
    assert (
        _results.get_final_state(
            reduce_to_basis="ground-rydberg", ignore_global_phase=False
        )
        == _results.states[-1].tidyup()
    )
    # Get final state is last state in results
    assert np.all(
        np.isclose(
            np.abs(_results.get_final_state(ignore_global_phase=False).full()),
            np.abs(_results.states[-1].full()),
        )
    )
    # For atoms that are far enough there is no impact of global_phase
    # Density matrix states are not changed by global phase
    assert np.all(
        np.isclose(
            np.abs(_results.get_final_state(ignore_global_phase=True).full()),
            np.abs(_results.states[-1].full()),
        )
    )

    seq_ = Sequence(reg, DigitalAnalogDevice)
    seq_.declare_channel("ryd", "rydberg_global")
    seq_.declare_channel("ram", "raman_local", initial_target="A")
    seq_.add(pi_pulse, "ram")
    seq_.add(pi_pulse, "ram")
    seq_.add(pi_pulse, "ryd")

    sim_ = QutipEmulator.from_sequence(seq_)
    results_ = sim_.run()
    results_ = cast(CoherentResults, results_)

    with pytest.raises(ValueError, match="'reduce_to_basis' must be"):
        results_.get_final_state(reduce_to_basis="all")

    with pytest.raises(TypeError, match="Can't reduce to chosen basis"):
        results_.get_final_state(reduce_to_basis="digital")
    h_states = results_.get_final_state(
        reduce_to_basis="digital", tol=1, normalize=False
    ).full()[1:]
    assert np.linalg.norm(h_states) < 3e-6

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


@pytest.mark.filterwarnings("ignore:Setting samples_per_run different to 1 is")
def test_get_final_state_noisy(reg, pi_pulse):
    np.random.seed(123)
    seq_ = Sequence(reg, DigitalAnalogDevice)
    seq_.declare_channel("ram", "raman_local", initial_target="A")
    seq_.add(pi_pulse, "ram")
    sim_noisy = QutipEmulator.from_sequence(
        seq_,
        noise_model=NoiseModel(
            runs=15,
            samples_per_run=5,
            temperature=50.0,
            state_prep_error=0.005,
            p_false_pos=0.01,
            p_false_neg=0.05,
        ),
    )
    res3 = sim_noisy.run()
    res3._meas_basis = "digital"
    final_state = res3.get_final_state()
    assert isdiagonal(final_state)
    res3._meas_basis = "ground-rydberg"
    assert (
        final_state[0, 0] == 0.06666666666666668 + 0j
        and final_state[2, 2] == 0.9066666666666667 + 0j
    )
    assert res3.states[-1] == final_state
    assert res3.results[-1] == Counter(
        {
            "10": 0.9066666666666667,
            "00": 0.06666666666666668,
            "11": 0.026666666666666672,
        }
    )


def test_get_state_float_time(results):
    with pytest.raises(IndexError, match="is absent from"):
        results.get_state(-1.0)
    with pytest.raises(IndexError, match="is absent from"):
        mean = (results._sim_times[-1] + results._sim_times[-2]) / 2
        diff = (results._sim_times[-1] - results._sim_times[-2]) / 2
        results.get_state(mean, t_tol=diff / 2)
    state = results.get_state(mean, t_tol=3 * diff / 2)
    assert state == results.get_state(results._sim_times[-2])


def test_expect(results, pi_pulse, reg):
    with pytest.raises(TypeError, match="must be a list"):
        results.expect("bad_observable")
    with pytest.raises(TypeError, match="Incompatible type"):
        results.expect(["bad_observable"])
    with pytest.raises(ValueError, match="Incompatible shape"):
        results.expect([np.array(3)])
    reg_single = Register.from_coordinates([(0, 0)], prefix="q")
    seq_single = Sequence(reg_single, DigitalAnalogDevice)
    seq_single.declare_channel("ryd", "rydberg_global")
    seq_single.add(pi_pulse, "ryd")
    sim_single = QutipEmulator.from_sequence(seq_single)
    results_single = sim_single.run()
    op = [qutip.basis(2, 0).proj()]
    exp = results_single.expect(op)[0]
    assert np.isclose(exp[-1], 1)
    assert len(exp) == pi_pulse.duration + 1  # +1 for the final instant
    np.testing.assert_almost_equal(
        results_single._calc_pseudo_density(-1).full(),
        np.array([[1, 0], [0, 0]]),
    )

    noise_model = NoiseModel(p_false_pos=0.01, p_false_neg=0.05)
    sim_single = QutipEmulator.from_sequence(
        seq_single, noise_model=noise_model
    )
    sim_single.set_evaluation_times("Minimal")
    results_single = sim_single.run()
    exp = results_single.expect(op)[0]
    assert len(exp) == 2
    assert isinstance(results_single, CoherentResults)
    assert results_single._meas_errors == {
        "epsilon": noise_model.p_false_pos,
        "epsilon_prime": noise_model.p_false_neg,
    }
    # Probability of measuring 1 = probability of false positive
    assert np.isclose(exp[0], noise_model.p_false_pos)
    # Probability of measuring 1 = 1 - probability of false negative
    assert np.isclose(exp[-1], 1 - noise_model.p_false_neg)
    np.testing.assert_almost_equal(
        results_single._calc_pseudo_density(-1).full(),
        np.array(
            [[1 - noise_model.p_false_neg, 0], [0, noise_model.p_false_neg]]
        ),
    )
    seq3dim = Sequence(reg, DigitalAnalogDevice)
    seq3dim.declare_channel("ryd", "rydberg_global")
    seq3dim.declare_channel("ram", "raman_local", initial_target="A")
    seq3dim.add(pi_pulse, "ram")
    seq3dim.add(pi_pulse, "ryd")
    sim3dim = QutipEmulator.from_sequence(seq3dim)
    exp3dim = sim3dim.run().expect(
        [qutip.tensor(qutip.basis(3, 0).proj(), qutip.qeye(3))]
    )
    assert np.isclose(exp3dim[0][-1], 1.89690200e-14)


def test_expect_noisy(results_noisy):
    np.random.seed(123)
    bad_op = qutip.tensor([qutip.qeye(2), qutip.sigmap()])
    with pytest.raises(ValueError, match="non-diagonal"):
        results_noisy.expect([bad_op])
    op = qutip.tensor([qutip.qeye(2), qutip.basis(2, 0).proj()])
    assert np.isclose(results_noisy.expect([op])[0][-1], 0.8)


@pytest.mark.filterwarnings("ignore:Setting samples_per_run different to 1 is")
def test_plot(results_noisy, results):
    op = qutip.tensor([qutip.qeye(2), qutip.basis(2, 0).proj()])
    results_noisy.plot(op)
    results_noisy.plot(op, error_bars=False)
    results.plot(op)


def test_sim_without_measurement(seq_no_meas):
    assert not seq_no_meas.is_measured()
    sim_no_meas = QutipEmulator.from_sequence(seq_no_meas)
    results_no_meas = sim_no_meas.run()
    np.random.seed(123)
    assert results_no_meas.sample_final_state(1) == Counter({"11": 1})


def test_sample_final_state(results):
    sampling = results.sample_final_state(1234)
    assert len(sampling) == 4  # Check that all states were observed.

    # Switch the measurement basis in the result
    results[-1].matching_meas_basis = False
    sampling0 = results.sample_final_state(N_samples=911)
    assert sampling0 == {"00": 911}


def test_sample_final_state_three_level(seq_no_meas, pi_pulse):
    seq_no_meas.declare_channel("raman", "raman_local", "B")
    seq_no_meas.add(pi_pulse, "raman")
    res_3level = QutipEmulator.from_sequence(seq_no_meas).run()
    # Raman pi pulse on one atom will not affect other,
    # even with global pi on rydberg
    assert len(res_3level.sample_final_state()) == 2

    seq_no_meas.measure("ground-rydberg")
    res_3level_gb = QutipEmulator.from_sequence(seq_no_meas).run()
    sampling_three_levelB = res_3level_gb.sample_final_state()
    # Rydberg will affect both:
    assert len(sampling_three_levelB) == 4


@pytest.mark.filterwarnings("ignore:Setting samples_per_run different to 1 is")
def test_sample_final_state_noisy(seq_no_meas, results_noisy):
    np.random.seed(123)
    assert results_noisy.sample_final_state(N_samples=1234) == Counter(
        {"11": 626, "10": 154, "01": 351, "00": 103}
    )
    res_3level = QutipEmulator.from_sequence(
        seq_no_meas,
        noise_model=NoiseModel(
            runs=10,
            samples_per_run=5,
            temperature=50.0,
            state_prep_error=0.005,
            p_false_pos=0.01,
            p_false_neg=0.05,
        ),
    )
    final_state = res_3level.run().states[-1]
    assert np.isclose(
        final_state.full(),
        np.array(
            [
                [0.42 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.1 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.34 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.14 + 0.0j],
            ]
        ),
    ).all()


def test_results_xy(reg, pi_pulse):
    seq_ = Sequence(reg, MockDevice)

    # Declare Channels
    seq_.declare_channel("ch0", "mw_global")
    seq_.add(pi_pulse, "ch0")
    seq_.measure("XY")

    sim_ = QutipEmulator.from_sequence(seq_)
    results_ = sim_.run()

    assert results_._dim == 2
    assert results_._size == 2
    assert results_._basis_name == "XY"
    assert results_._meas_basis == "XY"
    # Checks the initial state
    assert results_.states[0] == qutip.tensor(
        [qutip.basis(2, 0), qutip.basis(2, 0)]
    )

    with pytest.raises(TypeError, match="Can't reduce a system in"):
        results_.get_final_state(reduce_to_basis="all")

    with pytest.raises(TypeError, match="Can't reduce a system in"):
        results_.get_final_state(reduce_to_basis="ground-rydberg")

    with pytest.raises(TypeError, match="Can't reduce a system in"):
        results_.get_final_state(reduce_to_basis="digital")

    state = results_.get_final_state(reduce_to_basis="XY")

    assert np.all(
        np.isclose(
            np.abs(state.full()), np.abs(results_.states[-1].full()), atol=1e-5
        )
    )

    # Check that measurement projectors are correct
    assert results_._meas_projector(0) == qutip.basis(2, 0).proj()
    assert results_._meas_projector(1) == qutip.basis(2, 1).proj()


def test_false_positive():
    """Breaks for pulser version < v0.18."""
    seq = Sequence(Register.square(2, 5, prefix="q"), AnalogDevice)
    seq.declare_channel("ryd_glob", "rydberg_global")
    seq.add(
        Pulse.ConstantDetuning(
            CompositeWaveform(
                ConstantWaveform(2500, 0.0),
                BlackmanWaveform(1000, np.pi),
                ConstantWaveform(500, 0.0),
            ),
            0,
            0,
        ),
        channel="ryd_glob",
    )
    sim = QutipEmulator.from_sequence(seq)
    assert sim.run().get_final_state() != sim.initial_state
