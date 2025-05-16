# Copyright 2023 Pulser Development Team
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

import numpy as np
import pytest
import qutip

import pulser
from pulser.devices import MockDevice
from pulser.register import SquareLatticeLayout
from pulser.waveforms import BlackmanWaveform
from pulser_simulation import SimConfig
from pulser_simulation.qutip_backend import QutipBackend
from pulser_simulation.qutip_result import QutipResult
from pulser_simulation.simresults import CoherentResults, NoisyResults


@pytest.fixture
def sequence():
    reg = pulser.Register({"q0": (0, 0)})
    seq = pulser.Sequence(reg, MockDevice)
    seq.declare_channel("raman_local", "raman_local", initial_target="q0")
    seq.add(
        pulser.Pulse.ConstantDetuning(BlackmanWaveform(1000, np.pi), 0, 0),
        "raman_local",
    )
    return seq


def test_qutip_backend(sequence):
    sim_config = SimConfig()
    with pytest.raises(TypeError, match="must be of type 'EmulatorConfig'"):
        QutipBackend(sequence, sim_config)

    qutip_backend = QutipBackend(sequence)
    results = qutip_backend.run()
    assert isinstance(results, CoherentResults)
    assert results[0].get_state() == qutip.basis(2, 0)

    final_result = results[-1]
    assert isinstance(final_result, QutipResult)
    final_state = final_result.get_state()
    assert final_state == results.get_final_state()
    np.testing.assert_allclose(final_state.full(), [[0], [1]], atol=1e-5)

    # Test mimic QPU
    with pytest.raises(TypeError, match="must be a real device"):
        QutipBackend(sequence, mimic_qpu=True)
    sequence = sequence.switch_device(pulser.DigitalAnalogDevice)
    with pytest.raises(ValueError, match="defined from a `RegisterLayout`"):
        QutipBackend(sequence, mimic_qpu=True)
    sequence = sequence.switch_register(
        SquareLatticeLayout(5, 5, 5).square_register(2)
    )
    QutipBackend(sequence, mimic_qpu=True)


def test_with_default_noise(sequence):
    spam_noise = pulser.NoiseModel(
        p_false_pos=0.1,
        p_false_neg=0.05,
        state_prep_error=0.1,
        runs=10,
        samples_per_run=1,
    )
    new_device = dataclasses.replace(
        MockDevice, default_noise_model=spam_noise
    )
    new_seq = sequence.switch_device(new_device)
    backend = QutipBackend(
        new_seq, config=pulser.EmulatorConfig(prefer_device_noise_model=True)
    )
    new_results = backend.run()
    assert isinstance(new_results, NoisyResults)
    assert backend._sim_obj.config == SimConfig.from_noise_model(spam_noise)


proj = [[0, 0], [0, 1]]


@pytest.mark.parametrize(
    "collapse_op", [qutip.sigmax(), qutip.Qobj(proj), np.array(proj), proj]
)
def test_collapse_op(sequence, collapse_op):
    noise_model = pulser.NoiseModel(
        eff_noise_opers=[collapse_op], eff_noise_rates=[0.1]
    )
    backend = QutipBackend(
        sequence, config=pulser.EmulatorConfig(noise_model=noise_model)
    )
    assert all(
        op.dtype == qutip.core.data.CSR
        for op in backend._sim_obj._hamiltonian._collapse_ops
    )
