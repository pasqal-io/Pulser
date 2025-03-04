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

import math

import numpy as np
import pytest
import qutip

import pulser
from pulser.backend.default_observables import Energy, StateResult
from pulser_simulation.qutip_backend import QutipBackendV2
from pulser_simulation.qutip_config import QutipConfig

Omega_max = 4 * 2 * math.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000
t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000


@pytest.fixture
def sequence():
    R_interatomic = pulser.devices.MockDevice.rydberg_blockade_radius(U)
    reg = pulser.Register.rectangle(1, 4, R_interatomic, prefix="q")

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

    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")

    return seq


def test_qutip_backend_energy(sequence):
    with pytest.raises(
        TypeError, match="'config' must be an instance of 'EmulationConfig'"
    ):
        QutipBackendV2(sequence, config="tralala")

    sim_config = QutipConfig(
        default_evaluation_times=(1.0,),
        observables=[
            StateResult(evaluation_times=[1.0]),
            Energy(evaluation_times=[0.001 * n for n in range(1001)]),
        ],
    )
    qutip_backend = QutipBackendV2(sequence, config=sim_config)
    results = qutip_backend.run()
    assert (
        results.get_result("energy", 0.0)
        == results.energy[0]
        == pytest.approx(0.0)
    )
    assert (
        results.get_result("energy", 1.0)
        == results.energy[-1]
        == pytest.approx(
            qutip.expect(
                qutip_backend._sim_obj.get_hamiltonian(
                    t_rise + t_sweep + t_fall
                ),
                results.state[-1].to_qobj(),
            )
        )
    )
