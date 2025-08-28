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

import numpy as np
import pytest
import qutip

import pulser
from pulser.backend.default_observables import Energy, Occupation, StateResult
from pulser.backend.observable import Callback
from pulser_simulation.qutip_backend import QutipBackendV2
from pulser_simulation.qutip_config import QutipConfig
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
        noise_model=pulser.NoiseModel(
            amp_sigma=0.1, runs=1, samples_per_run=1
        ),
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
            dephasing_rate=0.01, temperature=50, runs=2, samples_per_run=1
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
    )

    backend = QutipBackendV2(sequence(noisy_device), config=config)

    assert backend._sim_obj._hamiltonian._config.p_false_neg == 0.0
    assert backend._sim_obj._hamiltonian._config.temperature == 50
    assert backend._sim_obj._hamiltonian._config.dephasing_rate == 0.01

    backend.run()


def test_qutip_backend_v2_stochastic_noise():
    np.random.seed(123)

    def get_noise_model(samples_per_run: int) -> pulser.NoiseModel:
        return pulser.NoiseModel(
            temperature=50.0,
            p_false_neg=0.01,
            amp_sigma=1e-3,
            runs=30,
            samples_per_run=samples_per_run,
        )

    config = QutipConfig(
        default_evaluation_times=(1.0,),
        observables=[
            StateResult(evaluation_times=[1.0]),
            Occupation(evaluation_times=[0.001 * n for n in range(1001)]),
        ],
        noise_model=get_noise_model(samples_per_run=1),
    )
    seq = sequence()
    backend = QutipBackendV2(seq, config=config)
    results = backend.run()

    # Same run with old API
    with pytest.warns(
        DeprecationWarning, match="Setting samples_per_run different to 1 is"
    ):
        qutip_emulator = QutipEmulator.from_sequence(
            seq, noise_model=get_noise_model(samples_per_run=100)
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
