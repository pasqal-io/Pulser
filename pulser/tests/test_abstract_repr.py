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
import json
from unittest.mock import patch

import numpy as np
import pytest
import jsonschema

from pulser import Pulse, Register, Sequence
from pulser.devices import Chadoq2
from pulser.waveforms import (
    BlackmanWaveform,
)

# from tutorials/advanced_features/Serialization.ipynb
qubits = {"control": (-2, 0), "target": (2, 0)}
reg = Register(qubits)

seq = Sequence(reg, Chadoq2)
seq.declare_channel("digital", "raman_local", initial_target="control")
seq.declare_channel("rydberg", "rydberg_local", initial_target="control")

half_pi_wf = BlackmanWaveform(200, np.pi / 2)

ry = Pulse.ConstantDetuning(amplitude=half_pi_wf, detuning=0, phase=-np.pi / 2)
ry_dag = Pulse.ConstantDetuning(
    amplitude=half_pi_wf, detuning=0, phase=np.pi / 2
)

seq.add(ry, "digital")
seq.target("target", "digital")
seq.add(ry_dag, "digital")

pi_wf = BlackmanWaveform(200, np.pi)
pi_pulse = Pulse.ConstantDetuning(pi_wf, 0, 0)

max_val = Chadoq2.rabi_from_blockade(8)
two_pi_wf = BlackmanWaveform.from_max_val(max_val, 2 * np.pi)
two_pi_pulse = Pulse.ConstantDetuning(two_pi_wf, 0, 0)

seq.align("digital", "rydberg")
seq.add(pi_pulse, "rydberg")
seq.target("target", "rydberg")
seq.add(two_pi_pulse, "rydberg")
seq.target("control", "rydberg")
seq.add(pi_pulse, "rydberg")

seq.align("digital", "rydberg")
seq.add(ry, "digital")
seq.measure("digital")

abstract = json.loads(seq.abstract_repr())


def test_schema():
    with open("pulser/json/abstract_repr_schema.json") as f:
        schema = json.load(f)

    jsonschema.validate(instance=abstract, schema=schema)


def test_values():
    assert set(abstract.keys()) == set(
        [
            "version",
            "device",
            "register",
            "parameters",
            "channels",
            "operations",
            "measurement",
        ]
    )
    assert abstract["device"] == "Chadoq2"
    assert abstract["register"] == {
        "control": {"x": -2.0, "y": 0.0},
        "target": {"x": 2.0, "y": 0.0},
    }
    assert abstract["channels"] == {
        "digital": {
            "hardware_channel": "raman_local",
            "initial_target": "control",
        },
        "rydberg": {
            "hardware_channel": "rydberg_local",
            "initial_target": "control",
        },
    }
    assert len(abstract["operations"]) == 13
    assert abstract["operations"][0] == {
        "op": "target",
        "channel": "digital",
        "target": "control",
    }
    assert abstract["operations"][2] == {
        "op": "pulse",
        "channel": "digital",
        "protocol": "min-delay",
        "amplitude": {
            "kind": "blackman",
            "duration": 200,
            "area": 1.5707963267948966,
        },
        "detuning": {"kind": "constant", "duration": 200, "value": 0.0},
        "phase": 4.71238898038469,
        "post_phase_shift": 0.0,
    }

    assert abstract["operations"][5] == {
        "op": "align",
        "channels": ["digital", "rydberg"],
    }
    assert abstract["measurement"] == "digital"
