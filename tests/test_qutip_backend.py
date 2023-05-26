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

import numpy as np
import pytest
import qutip

import pulser
from pulser.devices import MockDevice
from pulser.waveforms import BlackmanWaveform
from pulser_simulation import SimConfig
from pulser_simulation.qutip_backend import QutipBackend
from pulser_simulation.qutip_result import QutipResult
from pulser_simulation.simresults import CoherentResults


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
