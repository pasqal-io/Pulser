# Copyright 2025 Pulser Development Team
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
from __future__ import annotations

import dataclasses
import math
from unittest.mock import patch

import numpy as np
import pytest
import qutip

import pulser
from pulser.backend.default_observables import (
    BitStrings,
    Energy,
    Occupation,
    StateResult,
)
from pulser.backend.observable import Callback
from pulser_simulation.qutip_backend import QutipBackendV2
from pulser_simulation.qutip_config import QutipConfig, Solver
from pulser_simulation.qutip_op import QutipOperator
from pulser_simulation.qutip_state import QutipState
from pulser_simulation.simulation import QutipEmulator


class CountCalls(Callback):
    def __init__(self):
        """Simple callback that counts how often it's been called.

        The count can be queried after the simulation from self.counter.
        """
        self.counter = 0

    def __call__(self, **kwargs):
        self.counter += 1


def sequence(device: pulser.devices.Device | None = None):
    Omega_max = 4 * 2 * math.pi
    U = Omega_max / 2
    delta_0 = -6 * U
    delta_f = 2 * U
    t_rise = 500
    t_fall = 1000
    t_sweep = int((delta_f - delta_0) / (2 * np.pi * 10) * 1000)

    R_interatomic = pulser.devices.MockDevice.rydberg_blockade_radius(U)
    reg = pulser.Register.rectangle(1, 2, R_interatomic, prefix="q")

    rise = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max,
        pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f),
        0.0,
    )
    fall = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
    )

    seq = pulser.Sequence(
        reg, device if device is not None else pulser.devices.MockDevice
    )
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")

    return seq


def test_callback():
    seq = sequence()

    config = QutipConfig(
        callbacks=[CountCalls()],
    )
    backend = QutipBackendV2(seq, config=config)
    backend.run()
    assert backend._config.callbacks[0].counter == seq.get_duration() + 1

    config = QutipConfig(
        callbacks=[CountCalls()],
        noise_model=pulser.NoiseModel(amp_sigma=0.1),
        n_trajectories=1,
    )
    backend = QutipBackendV2(seq, config=config)
    backend.run()
    assert backend._config.callbacks[0].counter == seq.get_duration() + 1


def test_qutip_backend_v2_energy():
    seq = sequence()
    with pytest.raises(
        TypeError, match="'config' must be an instance of 'EmulationConfig'"
    ):
        QutipBackendV2(seq, config="tralala")
    config = QutipConfig(
        default_evaluation_times="Full",
        observables=[
            StateResult(evaluation_times=[1.0]),
            Energy(evaluation_times=[0.001 * n for n in range(1001)]),
        ],
    )
    backend = QutipBackendV2(seq, config=config)
    results = backend.run()
    assert (
        results.get_result("energy", 0.0)
        == results.energy[0]
        == pytest.approx(0.0)
    )
    assert results.get_result("energy", 0.5) == pytest.approx(
        qutip.expect(
            backend._sim_obj.get_hamiltonian(seq.get_duration() // 2),
            results.state[len(results.state) // 2].to_qobj(),
        ),
        rel=1e-5,
    )
    assert (
        results.get_result("energy", 1.0)
        == results.energy[-1]
        == pytest.approx(
            qutip.expect(
                backend._sim_obj.get_hamiltonian(seq.get_duration()),
                results.state[-1].to_qobj(),
            )
        )
    )


def test_qutip_backend_v2_default_noise_model():
    noisy_device = dataclasses.replace(
        pulser.devices.MockDevice,
        default_noise_model=pulser.NoiseModel(
            dephasing_rate=0.01, temperature=50
        ),
    )

    config = QutipConfig(
        observables=[
            StateResult(evaluation_times=[1.0]),
        ],
        noise_model=pulser.NoiseModel(p_false_neg=0.1),
        prefer_device_noise_model=True,
        initial_state=QutipState(
            qutip.tensor([qutip.basis(2, 0) for _ in range(2)]),
            eigenstates=("r", "g"),
        ),
        n_trajectories=2,
    )

    backend = QutipBackendV2(sequence(noisy_device), config=config)

    # The QutipEmulator should use the device noise model as per the config
    assert backend._sim_obj._hamiltonian_data.noise_model.p_false_neg == 0.0
    assert backend._sim_obj._hamiltonian_data.noise_model.temperature == 50
    assert (
        backend._sim_obj._hamiltonian_data.noise_model.dephasing_rate == 0.01
    )

    # However, the config will contain the given noise model
    assert backend._config.noise_model.p_false_neg == 0.1

    backend.run()


def test_qutip_backend_v2_stochastic_noise():
    np.random.seed(123)

    def get_noise_model(samples_per_run: int) -> pulser.NoiseModel:
        return pulser.NoiseModel(
            temperature=50.0,
            p_false_neg=0.01,
            amp_sigma=1e-3,
            samples_per_run=samples_per_run,
        )

    config = QutipConfig(
        default_evaluation_times=(1.0,),
        observables=[
            StateResult(evaluation_times=[1.0]),
            Occupation(evaluation_times=[0.001 * n for n in range(1001)]),
        ],
        noise_model=get_noise_model(samples_per_run=1),
        n_trajectories=30,
    )
    seq = sequence()
    backend = QutipBackendV2(seq, config=config)

    # Check trajectories are passed to _sim_obj
    assert backend._sim_obj.n_trajectories == config.n_trajectories

    results = backend.run()

    # Same run with old API
    with pytest.warns(
        DeprecationWarning, match="Setting samples_per_run different to 1 is"
    ):
        qutip_emulator = QutipEmulator.from_sequence(
            seq,
            noise_model=get_noise_model(samples_per_run=100),
            n_trajectories=30,
        )
    results_old_api = qutip_emulator.run()

    times = results.get_result_times("occupation")
    occupation = np.array([x[0] for x in results.occupation])

    indices = np.searchsorted(
        results_old_api._sim_times,
        np.array([int(t * seq.get_duration()) * 1e-3 for t in times]),
    )

    occupation_old_api = results_old_api.expect(
        [qutip.tensor([qutip.basis(2, 0).proj(), qutip.qeye(2)])]
    )[0][indices]

    assert np.max(np.abs(occupation - occupation_old_api)) < 0.03


def test_qutip_backend_v2_eval_times_rounding():

    # This was originally used to reproduce a bug where the legacy evaluation
    # times went above the maximum duration due to a rounding error

    n_points = 100

    # Sweeping duration values in multiples of the clock period
    # In each case, trying to get 100 evaluation points
    for duration in range(400, 600, 4):
        reg = pulser.Register({"q0": (-5, 0), "q1": (5, 0)})
        seq = pulser.Sequence(reg, pulser.AnalogDevice)
        seq.declare_channel("rydberg_global", "rydberg_global")

        amp_wf = pulser.ConstantWaveform(duration, np.pi)
        det_wf = pulser.ConstantWaveform(duration, 0.0)
        seq.add(pulser.Pulse(amp_wf, det_wf, 0), "rydberg_global")

        evaluation_times = np.linspace(0, 1, n_points).tolist()

        obs = [pulser.backend.StateResult(evaluation_times=evaluation_times)]
        config = pulser.backend.EmulationConfig(observables=obs)

        backend = QutipBackendV2(seq, config=config)

        result = backend.run().state

        assert len(result) == n_points


@pytest.mark.parametrize("amp_sigma", [0.0, 1.0])
def test_leakage(amp_sigma):
    natoms = 2
    reg = pulser.Register.rectangle(1, natoms, spacing=1000.0, prefix="q")

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 500
    pulse = pulser.Pulse.ConstantPulse(duration, np.pi, 0.0, 0.0)
    seq.add(pulse, "ch0")

    # pulser convention of basis
    basisx = np.array([0.0, 0.0, 1.0]).reshape(3, 1)
    basisg = np.array([0.0, 1.0, 0.0]).reshape(3, 1)
    basisr = np.array([1.0, 0.0, 0.0]).reshape(3, 1)

    rate = 0.5
    eff_rate = [rate, rate]
    eff_ops = [basisx @ basisr.T, basisx @ basisg.T]  # |x><r| and |x><g|

    noise_model = pulser.NoiseModel(
        eff_noise_rates=eff_rate,
        eff_noise_opers=eff_ops,
        with_leakage=True,
        amp_sigma=amp_sigma,
    )

    eval_times = [1.0]
    qutip_config = QutipConfig(
        default_evaluation_times=eval_times,
        observables=[StateResult(evaluation_times=eval_times)],
        noise_model=noise_model,
        solver=Solver.MESOLVER,
        n_trajectories=1,
    )

    qutip_sim = QutipBackendV2(seq, config=qutip_config)
    result_qut = qutip_sim.run()
    eigenstates = ("r", "g", "x")

    both_leaked = QutipOperator(
        qutip.tensor(
            [qutip.Qobj(basisx @ basisx.T), qutip.Qobj(basisx @ basisx.T)]
        ),
        eigenstates,
    )

    p_no_leaked = np.zeros((3, 3))
    p_no_leaked[0, 0] = 1.0
    p_no_leaked[1, 1] = 1.0

    one_leaked = QutipOperator(
        qutip.tensor([qutip.Qobj(basisx @ basisx.T), qutip.Qobj(p_no_leaked)]),
        eigenstates,
    ) + QutipOperator(
        qutip.tensor([qutip.Qobj(p_no_leaked), qutip.Qobj(basisx @ basisx.T)]),
        eigenstates,
    )
    no_leaked = QutipOperator(
        qutip.tensor([qutip.Qobj(p_no_leaked), qutip.Qobj(p_no_leaked)]),
        eigenstates,
    )

    assert one_leaked.expect(result_qut.final_state) == pytest.approx(
        2
        * (1 - math.exp(-rate * duration / 1000))
        * math.exp(-rate * duration / 1000)
    )
    assert no_leaked.expect(result_qut.final_state) == pytest.approx(
        math.exp(-2 * rate * duration / 1000)
    )
    assert both_leaked.expect(result_qut.final_state) == pytest.approx(
        (1 - math.exp(-rate * duration / 1000)) ** 2
    )


def test_register_detuning_detection():
    natoms = 2
    reg = pulser.Register.rectangle(1, natoms, spacing=1000.0, prefix="q")

    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ch0", "rydberg_global")
    duration = 500
    pulse = pulser.Pulse.ConstantPulse(duration, np.pi, 0.0, 0.0)
    seq.add(pulse, "ch0")

    noise_model = pulser.NoiseModel(
        trap_depth=1.0,
        trap_waist=1.0,
        temperature=50.0,
        disable_doppler=True,
        detuning_sigma=5.0,
    )

    assert set(noise_model.noise_types) == {"register", "detuning"}

    eval_times = [1.0]
    qutip_config = QutipConfig(
        default_evaluation_times=eval_times,
        observables=[StateResult(evaluation_times=eval_times)],
        noise_model=noise_model,
        n_trajectories=10,
    )

    qutip_sim = QutipBackendV2(seq, config=qutip_config)
    result_qut = qutip_sim.run()
    assert result_qut.final_state._state.shape == (4, 4)  # density matrix


def test_aggregation():
    reg = pulser.Register({"q0": [-1e5, 0], "q1": [1e5, 0], "q2": [0, 1e5]})
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ryd", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(100, np.pi), 0.0, 0.0
        ),
        "ryd",
    )

    occup = Occupation(evaluation_times=[1.0])
    state = StateResult(evaluation_times=[1.0])
    bitstrings = BitStrings(evaluation_times=[1.0])

    qutip_config = QutipConfig(
        observables=(occup, state, bitstrings),
        n_trajectories=5,
        noise_model=pulser.NoiseModel(state_prep_error=1 / 3),
    )
    with patch(
        "pulser._hamiltonian_data.hamiltonian_data.np.random.uniform"
    ) as bad_atoms_mock:
        # The bad qubits for each trajectory (0,0,1,1,2 respectively)
        # and a 6th item for the noiseless hamiltonian
        bad_atoms_mock.side_effect = [
            np.array([0.1, 0.5, 0.6]),
            np.array([0.1, 0.5, 0.6]),
            np.array([0.5, 0.1, 0.6]),
            np.array([0.5, 0.1, 0.6]),
            np.array([0.5, 0.6, 0.1]),
            np.array([0.1, 0.2, 0.3]),
        ]
        qutip_backend = QutipBackendV2(seq, config=qutip_config)
        qutip_results = qutip_backend.run()

    expected_state = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert np.allclose(
        qutip_results.final_state._state.full(), expected_state, atol=1e-4
    )
    assert np.allclose(
        qutip_results.occupation[-1], np.array([0.6, 0.6, 0.8]), atol=1e-4
    )
    assert qutip_results.final_bitstrings == {
        "011": 2000,
        "101": 2000,
        "110": 1000,
    }
